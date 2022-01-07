from utils import plot_decision_boundary
import torch
from NN import Net
from utils import l2_penalizer

class Deterministic_net(Net):
    """
    Model class for ANN
    """
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out_rate):
        super().__init__(input_dim, hidden_dim, output_dim, drop_out_rate)


    def train_net(self, train_loader, optimizer, criterion,l2, save_net = True, net_path = '', opti_path = ''):
        """
        :param train_loader: train data on torch dataloader
        :param optimizer: torch optimiser
        :param criterion: torch loss function
        :param l2: l2 regulariser parameter
        :param save_net: Bool
        :param net_path: path to save net to, if save_net == True
        :param opti_path: path to save optimiser to, if save_net == True
        :return:
        """
        loss_ave,loss_l2_ave = 0,0

        for i, (Xtrain, ytrain) in enumerate(train_loader):


            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get scores
            outputs = self(Xtrain)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, ytrain)
            loss_ave = (loss_ave * i + loss.item()) / (i + 1)

            loss_l2 = l2_penalizer(self)*l2 + loss
            loss_l2_ave = (loss_l2_ave * i + loss_l2) / (i + 1)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Update parameters
            optimizer.step()

        if save_net:
            torch.save(self.state_dict(),  net_path)
            torch.save(optimizer.state_dict(), opti_path)


        return optimizer, loss_ave, loss_l2_ave.item()

    def test_net(self, test_loader, criterion, temp, l2, freq=0, epoch='', plot=False, test = False):

        """
        :param test_loader: test data in toech dataloader structure
        :param criterion: nn loss function
        :param temp: temperature scala for calibration
        :param freq: How often decision plots should be plotted
        :param epoch: current epoch
        :param plot: whether to plot or not (bool)
        :return:
        """
        loss_ave, accuracy,loss_l2_ave = 0, 0, 0
        all_probs = []
        all_loss = []

        # Iterate through test dataset
        with torch.no_grad():
            for j, (Xtest, ytest) in enumerate(test_loader):


                probs, score, pred = self.predict(Xtest.float(), temp=temp)
                all_probs.append(probs)

                # Loss
                loss = criterion(score, ytest)
                all_loss.append(loss.item())
                loss_ave = (loss_ave * j + loss) / (j + 1)

                loss_l2 = l2_penalizer(self)*l2 + loss
                loss_l2_ave = (loss_l2_ave * j + loss_l2) / (j + 1)

                # Accuracy
                correct = (pred == ytest).sum() / len(ytest)
                accuracy = (accuracy * j + correct) / (j + 1)

                # plot decision boundary
                #if plot:
                #    if type(epoch) == str:
                #        plot_decision_boundary(self, Xtest, ytest, title="Decision boundary with test points")

                 #   elif epoch % freq == 0 and j == 0:
                 #       plot_decision_boundary(self, Xtest, ytest,
                #                               title="Decision boundary with test points: epoch %s" % epoch)

            all_probs = torch.cat(all_probs, dim=0)

        if test == True:
            return accuracy, loss_ave, all_probs, loss_l2_ave.item(), all_loss
        else:
            return accuracy, loss_ave, all_probs, loss_l2_ave.item()


    def test_net_out_dist(self, test_loader, temp):

        """
        :param test_loader: test data in toech dataloader structure
        :param criterion: nn loss function
        :param temp: temperature scala for calibration
        :param freq: How often decision plots should be plotted
        :param epoch: current epoch
        :param plot: whether to plot or not (bool)
        :return:
        """
        loss_ave, accuracy,loss_l2_ave = 0, 0, 0
        all_probs = []
        all_loss = []

        # Iterate through test dataset
        with torch.no_grad():
            for j, (Xtest, ytest) in enumerate(test_loader):


                probs, score, pred = self.predict(Xtest.float(), temp=temp)
                all_probs.append(probs)


            all_probs = torch.cat(all_probs, dim=0)


        return all_probs


