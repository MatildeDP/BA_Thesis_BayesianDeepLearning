import torch
import torch.nn as nn
from NN import Net
from utils import plot_decision_boundary, Squared_matrix, l2_penalizer
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np


def update_running_moment(theta, theta_i, n):
    """
    params: theta: tensor
            theta_i: tensor
            n: int
    """
    new = (n * theta + theta_i) / (n + 1)
    return new

class Swag(Net):
    def __init__(self, input_dim, hidden_dim, output_dim, K, c, S, criterion, l2_param, drop_out_rate):
        super().__init__(input_dim, hidden_dim, output_dim, drop_out_rate)

        self.K = int(K)
        self.p = sum(p.numel() for p in self.parameters())
        self.c = c
        self.S = S

        self.criterion = criterion
        self.l2_param = l2_param

        self.D_hat = []
        self.theta_swa = torch.tensor([])
        self.sigma_vec = torch.tensor([])

        self.output_dim = output_dim

        self.dims = (input_dim, hidden_dim, output_dim)


    def get_ith_moments(self):
        # Extract flattened parameters
        params = nn.utils.parameters_to_vector(self.parameters())

        # Initialize moments (second is uncentered)
        theta1 = params.clone().detach()  # detach from original tensor
        theta2 = params.clone().detach() ** 2

        return theta1, theta2

    def updateD(self, theta1_i, theta1):
        if len(self.D_hat) == self.K:
            self.D_hat.pop(0)  # remove first item
        self.D_hat.append(torch.unsqueeze(theta1_i - theta1, 1))


    def train_swag(self, epoch, n, theta1, theta2, train_loader, test_loader, optimizer):
        """
        :param epoch: Current epoch number
        :param n: number of updates to theta1, theta2 and Dhat
        :param theta1: current value of first moment
        :param theta2: current value of second moment

        :return: n
                 theta1: Updated first moment
                 theta2: Updated second moment
                 loss_ave: Train loss for epoch, averaged over batches
        """
        loss_ave, loss_l2_ave = 0, 0
        #rule = [int((len(train_loader.dataset.y)/train_loader.batch_size)//i) for i in range(1,5)]
        for j, (Xtrain, ytrain) in enumerate(train_loader):


            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get scores
            outputs = self(Xtrain)

            # SGD update
            loss = self.criterion(outputs, ytrain)  # Cross entropy loss
            loss.backward()  # Gradients
            optimizer.step()  # Update parameters

            loss_l2 = l2_penalizer(self)*self.l2_param + loss.item()

            # Update loss average
            loss_ave = (loss_ave * j + loss.item()) / (j + 1)
            loss_l2_ave = (loss_l2_ave * j + loss_l2) / (j + 1)
            #if j == 0:
            #    print("Running loss average: %s" %loss_ave)

        # Update moments with frequency c
            if j == int((len(train_loader.dataset.y)/train_loader.batch_size)//2) or j ==int((len(train_loader.dataset.y)/train_loader.batch_size)//4):

                with torch.no_grad():

                    # Update n
                    n += 1

                    # Get ith moment
                    theta1_i, theta2_i = self.get_ith_moments()

                    # Update first moment
                    temp1 = update_running_moment(theta=theta1, theta_i=theta1_i, n=n)
                    theta1 = temp1

                    # Update second moment
                    temp2 = update_running_moment(theta=theta2, theta_i=theta2_i, n=n)
                    theta2 = temp2

                    # Update D_hat
                    self.updateD(theta1_i, theta1)

                self.theta_swa = theta1.clone().detach()
                self.sigma_vec = theta2 - theta1 ** 2

        #  Plot every 100th epoch
        #if epoch % 200 == 0 and len(self.D_hat) == self.K:
           # title = "Epoch: " + str(epoch) + " \n Moments computed " + str(n) + " times."


           # plot_decision_boundary(model=self, dataloader = test_loader, title ='%s' %epoch, predict_func='stochastic', S = 10, temp = 1)
        return n, loss_ave, optimizer, theta1, theta2, loss_l2_ave

    def sample_from_posterior(self):
        """
        input: m: SWA means in 1D
               Sigma_diag: sample vairances 1D
               D_hat: approximate sample covarinance 1D
        return: parameters in vector shape
        """

        indices = [list(np.arange(self.p)), list(np.arange(self.p))]
        Sigma_diag_squared = torch.sparse_coo_tensor(indices, Squared_matrix(abs(self.sigma_vec)), (self.p, self.p))
        z1 = torch.empty(self.p)
        z2 = MultivariateNormal(torch.zeros(self.K), scale_tril=torch.diag(torch.ones(self.K)))

        # Sample weight from posterior
        param_sample = self.theta_swa + 1 / (2 ** (1 / 2)) * torch.smm(Sigma_diag_squared, z1.normal_().unsqueeze(-1)).to_dense().squeeze() + 1 / (
                    (2 * (self.K - 1)) ** (1 / 2)) * torch.matmul(torch.cat(self.D_hat, dim=1), z2.sample())

        return param_sample

    def replace_network_weights(self, sampled_weights, dropout):
        """
        # In order to test during training time, this method defines a new network to test
        :param sampled_weights: sampled weight from distribution
        :return:
        """
        net = Net(self.dims[0], self.dims[1], self.dims[2], drop_out_rate = dropout)
        nn.utils.vector_to_parameters(sampled_weights, net.parameters())
        return net







