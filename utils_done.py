

from copy import deepcopy
import torch.nn as nn
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from BMA import monte_carlo_bma
import pandas as pd

torch.manual_seed(1)


def save_net(path, model):
    torch.save(model.state_dict(), path)


def load_net(class_instance, path):
    """
    :param class_instance: Instance of NN class
           path: path were NN is saved
    :return:
    """
    model_ = class_instance
    model_.load_state_dict(state_dict=torch.load(path))
    model_.eval()  # set model to eval mode
    return deepcopy(model_)


def scatter(X, y, yhat, epoch):
    sns.set_theme()

    # Data
    X_ = X.detach().numpy()
    y_ = y.detach().numpy()
    yhat_ = [1 if x > 0.5 else 0 for x in yhat.detach().numpy()]

    # set axis
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle('Scatterplot, epoch %s' % epoch)
    axes[0].set_title('Labels')
    axes[1].set_title('Yhat')

    sns.scatterplot(ax=axes[0], x=X_[:, 0], y=X_[:, 1], hue=y_, style=y_)
    sns.scatterplot(ax=axes[1], x=X_[:, 0], y=X_[:, 1], hue=yhat_, style=yhat_)
    axes[2].plot(yhat.detach().numpy(), '.')
    #plt.show()
    plt.close()


def plot_acc_and_loss(testloss, trainloss, accuracy, save_path, save_image_path = True):
    sns.set_theme()
    num_epochs = len(accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Loss and Accuracy')

    dict_loss = {'Test': testloss,  'Train': trainloss, 'Epochs': [i for i in range(num_epochs)]}
    df_loss = pd.DataFrame(dict_loss)

    dict_acc = {'Accuracy':  accuracy, 'Epochs': [i for i in range(num_epochs)]}
    df_acc = pd.DataFrame(dict_acc)


    sns.lineplot(ax = axes[0], x = 'Epochs', y = 'value', hue='variable',  data=pd.melt(df_loss, 'Epochs'))
    axes[1].plot([i for i in range(num_epochs)],accuracy)
    axes[0].set_xlabel('Epochs')
    axes[1].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Accuracy')


    plt.savefig(save_path)

    #plt.show()
    plt.close()

def compare_parameter_loss_plot(trainloss, testloss, param, num_epochs, save_path = ''):
    """
    :param trainloss: dict: key = parameter value, value: loss
    :param testloss: dict: key = parameter value, value: loss
    :param: param: str: which parameter
    :return:
    """
    sns.set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Loss for different settings of' + param)

    trainloss['Epochs'] = [i for i in range(num_epochs)]
    testloss['Epochs'] = [i for i in range(num_epochs)]

    df_trainloss = pd.DataFrame(trainloss)
    df_testloss = pd.DataFrame(testloss)


    sns.lineplot(ax=axes[0], x='Epochs', y='value', hue='variable', data=pd.melt(df_trainloss, 'Epochs'),palette = "tab10" )
    sns.lineplot(ax=axes[1], x='Epochs', y='value', hue='variable', data=pd.melt(df_testloss, 'Epochs'), palette = "tab10")


    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')

    axes[1].set_title('Test Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')

    if save_path:
        plt.savefig(save_path)

    #plt.show()
    plt.close()




def plot_decision_boundary(model, dataloader, S, temp, text_string,dropout, path_to_bma='', title="", predict_func = 'predict', save_image_path = "", criterion = nn.CrossEntropyLoss(), sample_new_weights = True, show_text = True ):

    """
    :param model: instance of class build on nn modules. Must have a predict functions
    :param dataloader: datapoints to be plotted
    :param S: number of models in BMA approximations
    :param temp: parameter for temperature scaling
    :param load_net_path: path to network weights. Path should be to a folder with wanted files
    :param title: title of plot
    :param predict_func: deterministic or stochastic
    :param save_image_path: path to save image to
    :param criterion: nn loss function
    :param sample_new_weights: bool. decided whether to sample new weigths or not
    :return:
    """
    X = dataloader.dataset.X.clone().detach()
    y = dataloader.dataset.y.clone().detach()

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # plot decision boundary by sampling new models
    if predict_func == 'predict':
        probs_grid, _, _ = model.predict(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32), temp=temp)

    elif sample_new_weights and predict_func == 'stochastic':

        # Predict function value for grid
            probs_grid = monte_carlo_bma(model, Xtest=torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32), ytest=0, S=S, C=2, forplot=True, temp = temp, criterion = criterion, l2 = 0, dropout = dropout)

    # load saved models
    elif not sample_new_weights and predict_func == 'stochastic':
        probs_grid = monte_carlo_bma(model, Xtest=torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32),
                                         ytest=0, S=S, C=2, forplot=True, temp=temp, criterion=criterion, save_models = '',
                                        path_to_bma = path_to_bma, l2 = 0, dropout = dropout)



    probs0_grid = probs_grid.detach().numpy()[:, 1]
    probs0_grid = probs0_grid.reshape(xx.shape)

    # Plot the contour and test examples
    import matplotlib as mpl

    contour = plt.contourf(xx, yy, probs0_grid, cmap='RdBu', levels =100)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='RdBu',alpha = 0.2)
    ax = plt.colorbar(contour
    )

    ax.set_label("$P(y = 1)$")
    ax.set_ticks([0, .25, .5, .75, 1])
    plt.xlabel("$X_1$", fontsize='large', fontweight=300)
    plt.ylabel("$X_2$", fontsize='large', fontweight=300)
    box = dict(boxstyle='round', facecolor='black', alpha=0.5)
    
    if show_text:
        plt.text(1.2, -3,text_string, c = 'white', bbox=box, fontweight = 'medium')
        plt.title(title, fontsize=20, fontweight=300, fontfamily='sans-serif')
	


    #if save_image_path:
    plt.savefig(save_image_path)
    plt.close()





def Squared_matrix(D, is_diagonal = True):
    """
    params: D: if is_diagonal is false is needs a 2D vector, if is_diagonal is true it needs a 1D vector
    """

    if is_diagonal:
        # test if D is positive semi definite
        if np.any(D.detach().numpy() < 0):
            print("Some eigenvalues are smaller than 0")
            return False
        else:
            # principal root of D
            new = torch.sqrt(D)
            return new



def dump_to_json(PATH, dict):
    with open(PATH, 'w') as fp:
        temp_ = {'key': [dict]}
        json.dump(temp_, fp)



def dump_to_existing_json(PATH, dict):
    with open(PATH, "r+") as file:

        temp_ = json.load(file)
        temp_['key'].append(dict)
        file.seek(0)
        json.dump(temp_, file)



def l2_penalizer(model):
    s2 = 0
    for val in model.state_dict().values():
        if len(val.shape) > 1:
            s2 += sum(sum(val ** 2))
        else:
            s2 += sum(val ** 2)

    return s2

