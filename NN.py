import torch
import torch.nn as nn



# TODO implement running average instead of storing lists for loss!!!


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out_rate):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out_rate)
        self.s = nn.Softmax(dim=1)



    def forward(self, x):
        """"
        :param x:
        :return:
        """

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)


        return x

    def predict(self, X, temp):

        """
        temp should only be different from 1 on test phase
        :param X:
        :param temp:
        :return:
        """

        score = self(X) # calibrating
        score = score/temp
        probs = self.s(score)
        pred = torch.max(probs, 1).indices

        return probs, score, pred


