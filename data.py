import torch
from sklearn.datasets import make_moons, load_breast_cancer
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class AbaloneDict:
    def __init__(self):
        self.d = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5,
                  9:6, 10:7, 11:8, 12:9, 13:10,
                  14:11, 15:12, 16:13, 17:14, 18:15,
                  19:16, 20:17,21:18}



    def __getitem__(self, key):
        return self.d[key.item()]


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class DataLoaderInput:
    """
    Data structure required for torch data iterable
    """

    def __init__(self, X, y, which_data):
        super().__init__()
        self.X = X
        self.y = y
        self.which_data = which_data
        self.abalonedict = AbaloneDict()

    def __getitem__(self, idx):


        if self.which_data == 'abalone':
            real_y = self.y[idx]
            y = self.abalonedict.__getitem__(real_y)
            x = self.X[idx].float()
            return x, y

        else:
            x = self.X[idx].float()
            y = self.y[idx]
            return x, y


    def __len__(self):
            return len(self.X)


class LoadDataSet:
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([1, 784])])



    def load_data_for_CV(self, n_samples=3000, noise=0.3):
        """
        This method is not only relevant for CV. It returns test and training data. For two moons it returns
        validation data and training data.

        :param n_samples: number of samples. Only nessesary for moons
        :param noise: noise to put over moons
        :return: data
        """
        if self.dataset == 'two_moons':
            train_data = make_moons(n_samples=n_samples, noise=noise, random_state=0)
            test_data = make_moons(n_samples=1000, noise=noise, random_state=1)

            Xtest, ytest = test_data
            Xtest, ytest = torch.tensor(Xtest, dtype=torch.float32), torch.tensor(ytest)

            Xtrain, ytrain = train_data
            Xtrain, ytrain = torch.tensor(Xtrain, dtype=torch.float32), torch.tensor(ytrain)


        elif self.dataset == "mnist":
            test_data = datasets.MNIST(
                root='data',
                train=False
            )
            train_data = datasets.MNIST(
                root='data',
                train=True,
                # transform= #transforms.Normalize((0.1307,), (0.3081,)),
                download=True)

            Xtest, ytest = test_data.data.reshape(-1, 28 * 28).float(), test_data.targets
            Xtrain, ytrain = train_data.data.reshape(-1, 28 * 28).float(), train_data.targets


        elif self.dataset == 'abalone':

            data = pd.read_csv("abalone/abalone.data", sep=",", names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'target'])
            val, count = np.unique(data['target'], return_counts = True)
            val = val[count > 1]
            data = data[data['target'].isin(val)]

            dummies =pd.get_dummies(data['x1'])
            data = data.drop(['x1'], axis=1)
            data['I'] = dummies['I']
            data['M'] = dummies['M']
            data['F'] = dummies['F']

            y  = torch.tensor(data['target'].values)
            X = torch.tensor(data[['x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','F', 'M', 'I']].values).float()

            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)


        elif self.dataset == 'cancer':
            X, y = load_breast_cancer(return_X_y = True)
            X = torch.tensor(X).float()
            y = torch.tensor(y)

            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)

        elif self.dataset == 'emnist':
            test_data = datasets.EMNIST(
                root='data',
                train=False,
                split = 'letters'
            )
            train_data = datasets.EMNIST(
                root='data',
                train=True,
                # transform= #transforms.Normalize((0.1307,), (0.3081,)),
                download=True,
                split= 'letters')

            Xtest, ytest = test_data.data.reshape(-1, 28 * 28).float(), test_data.targets-1 # substract 1 for 0 indexing
            Xtrain_, ytrain_ = train_data.data.reshape(-1, 28 * 28).float(), train_data.targets-1 # substract 1 for 0 indexing

            Xtrain, _, ytrain, _ = train_test_split(Xtrain_, ytrain_, test_size=0.3, random_state=0, stratify=ytrain_) # remove some data for efficiency
        elif self.dataset == 'fashion':

            test_data = datasets.FashionMNIST(
                root='data',
                train=False,
                download = True
            )
            train_data = datasets.FashionMNIST(
                root='data',
                train=True,
                # transform= #transforms.Normalize((0.1307,), (0.3081,)),
                download=True)

            Xtest, ytest = test_data.data.reshape(-1, 28 * 28).float(), test_data.targets
            Xtrain, ytrain = train_data.data.reshape(-1, 28 * 28).float(), train_data.targets

        #elif self.dataset == 'cifar10':
        #    test_data = datasets.CIFAR10(
        #        root='data',
        #        train=False,
        #        download = True,
        #        transform= Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))#, Grayscale(num_output_channels=1)])
        #    )
        #    train_data = datasets.CIFAR10(
        #        root='data',
        #        train=True,
        #        # transform= #transforms.Normalize((0.1307,), (0.3081,)),
        #        download=True,
        #    transform= torchvision.transforms.Grayscale(num_output_channels=1))

         #   Xtest, ytest = test_data.data.reshape(-1, 32 * 32).float(), test_data.targets
          #  Xtrain, ytrain = train_data.data.reshape(-1, 32 * 32).float(), train_data.targets


        return Xtrain, ytrain, Xtest, ytest



def settings(data):
    if data == 'two_moons':
        input_dim = 2
        output_dim = 2

    elif data == 'mnist' or data == 'fashion':
        input_dim = 784
        output_dim = 10

    elif data == 'abalone':
        input_dim = 10
        output_dim = 19

    elif data == 'cancer':
        input_dim = 30
        output_dim = 2

    elif data == 'emnist':
        input_dim = 784
        output_dim = 26


    return input_dim, output_dim




