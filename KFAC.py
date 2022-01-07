
import torch.nn as nn
import torch
from NN import Net
from torch.autograd import functional
from torch import matmul as matmul
from scipy.stats import matrix_normal
from torch import inverse
from utils import plot_decision_boundary


class KFAC(Net):
    """
    Class for KFAC method
    """
    def __init__(self, input_dim, hidden_dim, output_dim, momentum, l2_param, L, drop_out_rate):
        super().__init__(input_dim, hidden_dim, output_dim, drop_out_rate)

        # name of layers
        self.layers = {1: self.fc1, 2: self.fc2, 3: self.fc3}
        self.diff = {1: 'relu', 2: 'relu'}
        self.L = L
        self.dims = (input_dim, hidden_dim, output_dim)
        self.y_correct = None

        self.a = {i: [] for i in range(L)}  # activation dict running sum
        self.a_grad = {i: [] for i in range(1, L + 1)}
        self.h = {i: [] for i in range(1, L + 1)}  # pre-activation dict running sum
        self.grads = {i: 0 for i in range(1, L + 1)}  # gradients for each layer running sum
        self.dedh3 = 0

        self.l2_param = l2_param
        self.momentum = momentum

        self.relu2 = nn.ReLU()

        # Backward hooks
        self.relu.register_backward_hook(self.save_activation_grads(1))
        self.relu2.register_backward_hook(self.save_activation_grads(2))
        self.fc3.register_backward_hook(self.save_pre_activation_grads)

        # Forward hooks
        self.fc1.register_forward_hook(self.save_pre_activations(1))
        self.fc2.register_forward_hook(self.save_pre_activations(2))
        self.fc3.register_forward_hook(self.save_pre_activations(3))
        self.relu.register_forward_hook(self.save_activations(1))
        self.relu2.register_forward_hook(self.save_activations(2))

        # Matrices
        self.Q = self.MatrixContainer(L)
        self.H = self.MatrixContainer(L)
        self.W = self.MatrixContainer(L)

    def __setitem__(self, value, key=0):
        self.a[key] = value

    class MatrixContainer:
        # class to create matrx objects
        def __init__(self, L):
            self.data = {Lambda: 0 for Lambda in range(1, L + 1)}
            self.L = L

        def __getitem__(self, item):
            try:
                return self.data[item]
            except KeyError:
                print("The matrix index %s does not exist" % item)
                return None

        def __setitem__(self, key, value):
            self.data[key] = value



    def forward(self, x):
        # forward function. It is nessesary to set requires_grad = True.
        x.requires_grad = True

        h1 = self.fc1(x)
        h1 = self.dropout(h1)
        a1 = self.relu(h1)
        h2 = self.fc2(a1)
        h2 = self.dropout(h2)
        a2 = self.relu2(h2)  # update activation
        h3 = self.fc3(a2)

        return h3

    def tanh_ddf(self, x):
        # second derivative of tanh
        return -2 * torch.tanh(x) * (1 - torch.tanh(x) ** 2)

    def tanh_df(self, x):
        # derivative of tanh
        return 1 - torch.tanh(x) ** 2


    def relu_ddf(self, x):
        # second derivative of relu
        return torch.zeros(x.shape)

    def relu_df(self, x):
        """
        :param x: input tensor
        :return: derivative of relu w.r.t x
        Notice: if x = 0, the derivative is undefined. However, it will be defined as 0.
        """
        temp = [int(val) for val in x > 0]
        if 0 in x:
            print("The derivative of Relu is not defined for x = 0. However, this function returns 0 when x = 0")

        return torch.tensor(temp)


    def save_activation_grads(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """
            self.a_grad[Lambda] = (mod, out[0].detach())

        return hook

    def save_pre_activation_grads(self, mod, ind, out):
        """
        Hook structure
        :param mod: nn module
        :param ind: tensor
        :param out: tensor gradient
        """
        self.dedh3 = (mod, out[0].detach())

    def save_pre_activations(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """
            self.h[Lambda] = (mod, out[0].detach())

        return hook

    def save_activations(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """

            self.a[Lambda] = (mod, out[0].detach())
        return hook


    def cross_entropy_loss_binary(self, correct, rest):
        """
        :param correct: The score of the network related to the correct class
        :param rest: The score of the network related to the incorrect class
        :return: cross entropy loss
        """

        loss = -torch.log(torch.exp(correct) / (torch.exp(rest) + torch.exp(correct)))
        return loss

    def cross_entropy_loss(self, scores):
        """
        :param scores: Output scores of network. First element MUST be the score of the correct class
        :return: cross entropy loss
        """

        return -torch.log(torch.exp(scores[self.y_correct]) / (sum([torch.exp(i) for i in scores])))

    def compute_hessian(self, func, input):
        """
        :param func: function to be evaluated
        :param input: tuple or tensor with input to func. Elements must be torch tensors
        :return: hessian of one data point
        """
        return functional.hessian(func, input)

    def collect_values(self, data, optimizer, criterion, temp):
        loss_ave = 0
        """
        Collects running average Q and H for each layer
        """

        # collect MAP estimate of weights
        self.collectWstar()

        for i, (Xtrain, ytrain) in enumerate(data):

            self.y_correct = ytrain.item()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # store x = a0
            self.__setitem__([('a0', a0.detach()) for a0 in Xtrain][0])

            # Forward pass to get scores
            outputs = self(Xtrain)

            # Calculate Loss
            loss = criterion(outputs, ytrain)
            loss_ave = (loss_ave * i + loss.item()) / (i + 1)

            if i % 100 == 0:
                print('Number of data points: %s        loss: %s' % (i, loss_ave))

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Compute Q and H
            self.compute_Q(n=i)
            self.compute_H(Lambda=1, n=i)

        return loss_ave


    def compute_Q(self, n):
        """
        :param Lambda: Layer index
        :param n: number of datapoint iterated so far
        :return: Covariance matrix of the activations
        """
        for Lambda in self.Q.data.keys():
            self.Q[Lambda] = (self.Q[Lambda] * n +
                            torch.matmul(
                            torch.cat(
                            (torch.unsqueeze(self.a[Lambda - 1][1], 1),
                            torch.tensor([[1]])),
                            0),
                torch.cat((torch.unsqueeze(self.a[Lambda - 1][1], 0),
                           torch.tensor([[1]])), 1))) / (n + 1)

    def compute_D(self, func, Lambda):

        """
        :param ddf_lambda: second derivative of activation function
        :param Lambda: Layer index
        :return: Diagonal matrix with elements equal to the product of the second derivative of  the activation function
        w.r.t the pre-activation and the derivative of the cost function w.r.t the activation
        """

        if func == 'tanh':
            return self.tanh_ddf(self.h[Lambda][1]) * self.a_grad[Lambda][1] * torch.eye(len(self.h[Lambda][1]))
        elif func == 'relu':
            return self.relu_ddf(self.h[Lambda][1]) * self.a_grad[Lambda][1] * torch.eye(len(self.h[Lambda][1]))


    def compute_B(self, func, Lambda):
        """
        :param df_lambda: derivative of activation function
        :param Lambda: Layer index
        :return: Diagonal matrix with elements equal to the derivative of the activation function w.r.t the pre-activation
        """
        if func == 'tanh':
            return self.tanh_df(self.h[Lambda][1]) * torch.eye(len(self.h[Lambda][1]))

            # return torch.cat((self.tanh_df(self.h[Lambda][1]), torch.zeros(1)), 0) * torch.eye(
            #    len(self.h[Lambda][1]) + 1)

        elif func == 'relu':
            return self.relu_df(self.h[Lambda][1]) * torch.eye(len(self.h[Lambda][1]))

    def compute_H(self, Lambda, n):
        """
        Computes (for one point) and stores (running average over points) the pre-activation hessian for layer Lambda
        :param Lambda: Layer number¢
        :return:
        """
        if Lambda == self.L:  # base case
            self.H[Lambda] = (self.H[Lambda] * n + self.compute_hessian(self.cross_entropy_loss, self.h[Lambda][1])) / (
                    n + 1)
            return self.H[Lambda]

        diff_ = self.diff[Lambda]
        B = self.compute_B(diff_, Lambda)
        W = self.layers[Lambda + 1].weight.detach()  # ,torch.unsqueeze(self.layers[Lambda + 1].bias.detach(), 1)), 1)
        Wt = torch.transpose(W, 0, 1)
        D = self.compute_D(diff_, Lambda)

        # Lambda += 1

        self.H[Lambda] = (self.H[Lambda] * n + (matmul(B, matmul(Wt, matmul(
            self.compute_H(Lambda=Lambda + 1, n=n), matmul(W, B)))) + D)) / (n + 1)

        return self.H[Lambda]

    def collectWstar(self):
        for Lambda in self.W.data.keys():
            l = self.layers[Lambda].weight.detach()
            b = self.layers[Lambda].bias.detach()
            self.W[Lambda] = torch.cat((l, torch.unsqueeze(b, 1)), 1)

    def regularize_and_add_prior(self, tau, N):
        """
        :param tau: Precision of Gaussian prior
        :param N: size of dataset
        :return:
        """

        for Lambda in self.H.data.keys():
            sqrt_tau = tau ** (1 / 2)
            sqrt_N = N ** (1 / 2)

            IH = len(self.H[Lambda])
            self.H[Lambda] = self.H[Lambda] * sqrt_N + sqrt_tau * torch.eye(IH)

            IQ = len(self.Q[Lambda])
            self.Q[Lambda] = self.Q[Lambda] * sqrt_N + sqrt_tau * torch.eye(IQ)

    def sample_from_posterior(self):
        """
        :return: weight samples
        """
        # TODO: gør vægte til float, ikke doubles

        # Initialise sample dict
        samples = {Lambda: None for Lambda in range(1, self.L + 1)}

        for Lambda in samples.keys():
            # Invert covariances
            Q_inv = inverse(self.Q[Lambda])
            H_inv = inverse(self.H[Lambda])

            # Sample
            try:
                samples[Lambda] = matrix_normal.rvs(mean=self.W[Lambda], rowcov=H_inv, colcov=Q_inv, size=1)
            except:
                print('Linalg error')
                return None

        return samples


    def replace_network_weights(self, sampled_weights, dropout):


        # Replace network weights with sampled weights
        for idx in range(1, len(self.layers) + 1):
            w_idx = self.layers[idx].weight.shape

            self.layers[idx].weight.data = torch.tensor(sampled_weights[idx][:, :w_idx[1]]).float()
            self.layers[idx].bias.data = torch.squeeze(torch.tensor(sampled_weights[idx][:, w_idx[1]:])).float()

        return None



