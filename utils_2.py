

from copy import deepcopy

import torch.nn as nn
import pickle
from copy import copy
import json
import numpy as np
from data import settings, LoadDataSet,DataLoaderInput
from deterministic import Deterministic_net
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from BMA import monte_carlo_bma, monte_carlo_bma_out
from KFAC import KFAC
from SWAG import Swag
from utils import plot_decision_boundary
from sklearn.datasets import make_moons
import pandas as pd
import os
from utils import dump_to_json
torch.manual_seed(1)
def plot_ece_of_parameters(model_name):
    for which_data in ['two_moons', 'mnist', 'emnist', 'fashion']:

        path = model_name + '/' + which_data + '/gridsearch/'

        input_dim, output_dim = settings(which_data)
        criterion = torch.nn.CrossEntropyLoss()
        # load data
        D = LoadDataSet(dataset=which_data)
        Xtrain, ytrain, Xtest, ytest = D.load_data_for_CV()

        if which_data != 'two_moons':
            Xtest = (Xtest - torch.mean(Xtrain)) / torch.std(Xtrain)
            dataset = DataLoaderInput(Xtest, ytest, which_data=which_data)
            test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=1,
                                                      shuffle=False)


        elif which_data == 'two_moons':
            test_data = make_moons(n_samples=1000, noise=0.3, random_state=12)
            Xtest, ytest = test_data
            Xtest, ytest = torch.tensor(Xtest, dtype=torch.float32), torch.tensor(ytest)
            dataset = DataLoaderInput(Xtest, ytest, which_data=which_data)
            test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=1,
                                                      shuffle=False)

        if which_data == 'two_moons':
            drop_out_rate = 0.05
        elif which_data == 'mnist':
            drop_out_rate = 0.1
        elif which_data == 'fashion':
            drop_out_rate = 0.3
        elif which_data == 'emnist':
            drop_out_rate = 0.05


        # read range
        info_path = path + 'params_evals/info.json'
        predict_func = 'stochastic'

        f = open(info_path)
        data = json.load(f)
        N = len(data['key'])

        all_last_testloss = [data['key'][i][str(i)]['evals:']['test_loss'] for i in range(N)] # TODO ændre til [-1] hvis swag!!!
        argmin = np.argmin(all_last_testloss)
        opti_params = data['key'][argmin][str(argmin)]['params']
        opti_tau = opti_params['tau']
        all_tau =  [data['key'][i][str(i)]['params']['tau'] for i in range(N)]


        evaluations = {'nll':[], 'nll_se':[], 'ece':[], 'ece_se':[]}
        bma_path = path + 'bma/'
        for i in range(len(os.listdir(bma_path))):
            path_to_bma = bma_path + 'idx_models_' + str(int(i)) +'temp_1/'

            print(i, path_to_bma, all_tau[i])

            if model_name == 'KFAC':
                Net = KFAC(input_dim=input_dim, hidden_dim=int(opti_params['hidden_dim']), output_dim=output_dim,
                           momentum=opti_params['momentum'], l2_param=opti_params['l2'], L=3,
                           drop_out_rate=drop_out_rate)
            elif model_name == 'SWAG':

                Net = Swag(input_dim=input_dim, hidden_dim=int(opti_params['hidden_dim']), output_dim=output_dim, K=20, c=1,
                       S=30, criterion=criterion, l2_param=opti_params['l2'], drop_out_rate=drop_out_rate)



            p_yxw, all_probs, all_loss, accuracy, loss_l2 = monte_carlo_bma(Net, Xtest, ytest, S=30, C=output_dim,
                                                                                temp=1, criterion=criterion,
                                                                                path_to_bma=path_to_bma,
                                                                                save_probs='', l2=opti_params['l2'],
                                                                                batch=True, dropout=drop_out_rate)

            # Compute standard error of nll loss (CLT says you can do it)
            C = len(all_probs[0])
            N = len(all_loss)
            s = torch.std(torch.tensor(all_loss))
            nll_se = s / torch.sqrt(torch.tensor(N))
            nll = sum(all_loss) / N
            evaluations['nll'].append(nll.item())
            evaluations['nll_se'].append(nll_se.item())

            # Reliability diagrams
            M = 40
            I = [((m - 1) / M, m / M) for m in range(1, M + 1)]
            I[-1] = (I[-1][0], I[-1][0] + 0.1)
            Counter = {i: {'correct': 0, 'incorrect': 0, 'probs': []} for i in I}

            for probs, y in zip(all_probs, dataset.y):
                correct = probs[y]
                rest = torch.cat((probs[0:y], probs[y:-1]), 0)
                for key in Counter.keys():
                    if correct >= key[0] and correct < key[1]:
                        Counter[key]['correct'] += 1
                        Counter[key]['probs'].append(correct.item())
                    for p in rest:
                        if p >= key[0] and p < key[1]:
                            Counter[key]['incorrect'] += 1
                            Counter[key]['probs'].append(p.item())

            Counter_ = copy(Counter)
            for key in Counter_.keys():
                if Counter[key]['correct'] + Counter[key]['incorrect'] == 0:
                    Counter.pop(key)

            acc = [Counter[i]['correct'] / (Counter[i]['correct'] + Counter[i]['incorrect']) for i in Counter.keys()]
            conf = [sum(Counter[i]['probs']) / (Counter[i]['correct'] + Counter[i]['incorrect']) for i in Counter.keys()]

            # expected calibration error

            ece_all = [(Counter[key]['correct'] + Counter[key]['incorrect']) / (N * C) * abs(acc[i] - conf[i]) for i, key in
                       enumerate(Counter.keys())]
            ece = sum(ece_all)
            ece_se = torch.std(torch.tensor(ece_all)) / torch.sqrt((torch.tensor(N * C)))

            evaluations['ece'].append(ece)
            evaluations['ece_se'].append(ece_se.item())


        if model_name == 'KFAC':
            dump_to_json(PATH = 'other_results/KFAC_tau_effect_.json', dict = evaluations)

        else:
            dump_to_json(PATH='other_results/SWAG_tau_effect_.json', dict=evaluations)

    sns.set_style('darkgrid')
    # Draw plot with error band and extra formatting to match seaborn style
    fig, ax = plt.subplots(1, 2, figsize=(9,5))

    # plot nll
    # plot
    nll_lower = np.array(evaluations['nll']) - np.array(evaluations['nll_se'])
    nll_upper = np.array(evaluations['nll']) + np.array(evaluations['nll_se'])
    ax[0].plot(all_tau, evaluations['nll'], label='NLL', color = 'green')
    ax[0].plot(all_tau, nll_lower, color='tab:green', alpha=0.1)
    ax[0].plot(all_tau, nll_upper, color='tab:green', alpha=0.1)
    ax[0].fill_between(all_tau, nll_lower, nll_upper, alpha=0.2)

    ece_lower = np.array(evaluations['ece']) - np.array(evaluations['ece_se'])
    ece_upper = np.array(evaluations['ece']) + np.array(evaluations['ece_se'])
    ax[1].plot(all_tau, evaluations['ece'], label='ECE', color = 'orange')
    ax[1].plot(all_tau, ece_lower, color='tab:orange', alpha=0.1)
    ax[1].plot(all_tau, ece_upper, color='tab:orange', alpha=0.1)
    ax[1].fill_between(all_tau, ece_lower, ece_upper, alpha=0.2)

    ax[0].set_title('Varying regularizer - KFAC', fontsize=20, fontweight=200, fontfamily='sans-serif')
    ax[1].set_title('Varying regularizer - KFAC', fontsize=20, fontweight=200, fontfamily='sans-serif')
    ax[0].set_xlabel('Tau', fontsize='x-large', fontweight=300)
    ax[1].set_xlabel('Tau', fontsize='x-large', fontweight=300)
    ax[0].set_ylabel('NLL', fontsize='x-large', fontweight=300)
    ax[1].set_ylabel('ECE', fontsize='x-large', fontweight=300)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[0].plot(opti_tau, evaluations['nll'][argmin],'*', color = 'red')
    ax[1].plot(opti_tau, evaluations['ece'][argmin],'*', color = 'red', label = 'Optimal tau')

    ax[0].legend(loc="upper right")
    ax[1].legend(loc="lower right")

    plt.savefig('other_results/KFAC_tau_effect_' + which_data, dpi = 500 )



def density_log_scale(data_out, data_in,i, ax, title):

    """
    Method plots density log histogram

    """

    sns.set_theme()

    data_out = data_out[data_out>0]
    data_in = data_in[data_in > 0]
    _, bins = np.histogram(data_out, bins=50, density=True)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax[i].hist(data_out, bins=logbins, label = 'Out of distribution data')

    _, bins = np.histogram(data_in, bins=50, density=True)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax[i].hist(data_in, bins=logbins, label = 'In distribution data')




    ax[i].set_xscale('log')
    ax[i].set_xlabel('Entropy')
    if i == 0:
        ax[i].set_ylabel('Count')
    ax[i].set_title(title)
    ax[i].legend(loc="upper left")
    plt.show()



def read_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data.numpy()

def load_in_and_out_data(path):
    """
    Method reads specific pickles and load to dict

    """
    Dict =  {'in': None, 'out':None}
    Dict['in'] = read_pickle('InDist/' + path +'.pickle')
    Dict['out'] = read_pickle('OutOfDist/' + path + '.pickle')

    return Dict

def load_H_ale(which_data):
    """
    Method reads specific pickles and load to dict

    """
    Dict_ale =  {'Ensemble': None, 'Ensemble w. T':None, 'KFAC': None, 'KFAC w. T':None, 'SWAG': None, 'SWAG w. T':None}
    Dict_epi = {'Ensemble': None, 'Ensemble w. T':None, 'KFAC': None, 'KFAC w. T':None, 'SWAG': None, 'SWAG w. T':None}
    Dict_total =  {'Ensemble': None, 'Ensemble w. T':None, 'KFAC': None, 'KFAC w. T':None, 'SWAG': None, 'SWAG w. T':None}

    paths = ['ensemble', 'ensemble_temp', 'kfac','kfac_temp', 'swag', 'swag_temp']
    models = ['Ensemble', 'Ensemble w. T', 'KFAC', 'KFAC w. T', 'SWAG', 'SWAG w. T']

    for m, path in zip(models, paths):
        total = read_pickle('total_uncertainty/' + which_data + '_' +path + '.pickle')
        ale = read_pickle('Aleatoric/' + which_data + '_' +path + '.pickle')
        Dict_total[m] = total
        Dict_ale[m] = ale
        Dict_epi[m] = total - ale

    return Dict_total, Dict_ale, Dict_epi

def KL(P, Q):
    """
    Computes KL divergence of entropy distributions
    :param P:
    :param Q:
    :return:
    """

    # Compute density
    density_Q, _ = np.histogram(Q, bins=50, density=True)
    density_P, _ = np.histogram(P, bins=50, density=True)

    # Normalise
    density_Q = density_Q/sum(density_Q)
    density_P = density_P/sum(density_P)

    # compute KL divegence
    KL = - sum(density_P *(np.log2((density_Q + 1e-7) / (density_P + 1e-7))))

    return KL

def symmetrised_KL(P, Q):
    """
    Computes symetrized KL divergence
    :param P:
    :param Q:
    :return:
    """
    return KL(P, Q) + KL(Q, P)



def plot_ecdf(DICT, uncertainty, which_data):
    """
    CPlots empirical cummulative distribution function of given uncertainty
    :param DICT:
    :param uncertainty:
    :return:
    """
    if uncertainty == 'epistemic':
        title = 'Epistemic Uncertainty ECDF'
        xlab = 'Epistemic Uncertainty'
    elif uncertainty == 'aleatoric':
        title ='Aleatoric Uncertainty ECDF'
        xlab = 'Aleatoric Uncertainty'
    elif uncertainty == 'total':
        title = 'Entropy ECDF'
        xlab = 'Entropy'

    sns.set_style('darkgrid')
    fig, ax = plt.subplots(1,1, figsize = (10,6))
    for i, key in enumerate(DICT.keys()):
        ax.plot(DICT[key].x, DICT[key].y, label = key)
        ax.legend(loc="upper left", fontsize = 15)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=20, fontweight=150, fontfamily='sans-serif')
        ax.set_xlabel(xlab, fontsize=15 )
        ax.set_ylabel('ecdf', fontsize=15)
    plt.savefig('uncertainty_plots/' + which_data + '_' + uncertainty)


def median(DICT):

    print('Temperature Scaling & SWAG & KFAC & Ensemble  \ \ \hline')
    print('No & {:.4} & {:.4} & {:.4} \ \ '.format(np.median(DICT['SWAG']), np.median(DICT['KFAC']), np.median(DICT['Ensemble'])))

    print('Yes & {:.4} & {:.4} & {:.4}'.format(np.median(DICT['SWAG w. T']), np.median(DICT['KFAC w. T']), np.median(DICT['Ensemble w. T'])))



def loss_plot(test_loss_swag, test_loss, train_loss_swag, train_loss, which_data):

    """
    Plot test and train loss for SWAG and ANN
    :param test_loss_swag:
    :param test_loss:
    :param train_loss_swag:
    :param train_loss:
    :return:
    """
    sns.set_theme()
    num_epochs = len(test_loss_swag)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    dict_loss = {'SWAG': test_loss_swag, 'ANN': test_loss, 'Epochs': [i for i in range(num_epochs)]}
    df_loss = pd.DataFrame(dict_loss)

    dict_loss_train = {'SWAG': train_loss_swag, 'ANN': train_loss, 'Epochs': [i for i in range(num_epochs)]}
    df_loss_train = pd.DataFrame(dict_loss_train)


    sns.lineplot(ax=axes[0], x='Epochs', y='value', hue='variable', data=pd.melt(df_loss, 'Epochs'))
    sns.lineplot(ax=axes[1], x='Epochs', y='value', hue='variable', data=pd.melt(df_loss_train, 'Epochs'))

    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Test loss')
    axes[0].set_title('SWAG and ANN Test Loss', fontsize=20, fontweight=300, fontfamily='sans-serif')

    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Train loss')
    axes[1].set_title('SWAG and ANN Train Loss', fontsize=20, fontweight=300, fontfamily='sans-serif')

    plt.savefig('other_results/loss_'+which_data+'.png', dpi = 500)

    # plt.show()
    plt.close()


def calibration_and_decision_boundary():


    which_model = input()
    decision_title = input()
    for what_to_do in ['gridsearch', 'calibrating']:
        print(what_to_do)

        for which_data, title_cal in zip(['two_moons'], ['Two Moons']):
            print(which_data, title_cal)
            # lr, l2, momentum, batch_size, hidden_dim, dropout

            in_dist = True
            predict_func_boundary = 'stochastic'

            info_path = 'Deterministic/' + which_data + '/BO/params_evals/model_info_BO.json'
            temp_info_path = 'Deterministic/' + which_data + '/ensample_temp/info.json'

            path_to_ensample = 'Deterministic/' + which_data + '/ensample/models/'


            f = open(info_path)
            data = json.load(f)
            N = len(data['key'])
            if which_model.lower() == 'swag':
                all_last_testloss = [data['key'][i][str(i)]['evals:']['test_loss'][-1] for i in range(N)]
                argmin = np.argmin(all_last_testloss)
                opti_params = data['key'][argmin][str(argmin)]['params']
                lr = opti_params['lr']
                print('lr: {:.4}'.format(lr))

            elif which_model.lower() == 'kfac':
                all_last_testloss = [data['key'][i][str(i)]['evals:']['test_loss'] for i in range(N)]
                argmin = np.argmin(all_last_testloss)
                opti_params = data['key'][argmin][str(argmin)]['params']
                tau = opti_params['tau']
                print('tau: {:.4}'.format(tau))

            else:
                all_last_testloss = [data['key'][i][str(i)]['evals']['test_loss'][-1] for i in range(N)]
                argmin = np.argmin(all_last_testloss)
                opti_params = data['key'][argmin][str(argmin)]['params']

            # find optimal temp
            if what_to_do == 'calibrating':
                f = open(temp_info_path)
                data = json.load(f)
                all_test_loss = data['key'][0]['test_loss']
                temp_argmin = np.argmin(all_test_loss)
                temp = data['key'][0]['temp'][temp_argmin]
                print('temp: ${:.4}$'.format(temp))

            else:
                temp = 1
                print('temp: ${}$'.format(temp))

            if what_to_do == 'calibrating':
                if which_model == 'ensample':
                    decision_text = 'Temperature: {:.4}'.format(temp)
                elif which_model == 'kfac':
                    decision_text = 'Temperature: {:.4}  \n  tau: ${:.4}'.format(temp, tau)

                elif which_model == 'swag':
                    decision_text = 'Temperature: {:.4}  \n  lr: ${:.4}'.format(temp, lr)

                else:
                    decision_text = ''

            elif what_to_do == 'gridsearch':

                if which_model == 'kfac':
                    decision_text = 'tau: {:.4}'.format(tau)

                elif which_model == 'swag':
                    decision_text = 'Learning rate: {:.4}'.format(lr)

                else:
                    decision_text = ''

            input_dim, output_dim = settings(which_data)
            opti_val_loss = min(all_last_testloss)

            if which_model.lower() == 'kfac':
                path_to_bma = "KFAC/" + which_data + "/gridsearch/bma/idx_models_" + str(argmin) + 'temp_1/'
            elif which_model.lower() == 'swag':
                path_to_bma = "SWAG/" + which_data + "/gridsearch/bma/idx_models_" + str(argmin) + 'temp_1/'
            else:
                path_to_bma = ''

            if which_data == 'two_moons':
                drop_out_rate = 0.05
            elif which_data == 'mnist':
                drop_out_rate = 0.1
            elif which_data == 'fashion':
                drop_out_rate = 0.3
            elif which_data == 'emnist':
                drop_out_rate = 0.05

            # Load data
            D = LoadDataSet(dataset=which_data)
            Xtrain, ytrain, Xtest, ytest = D.load_data_for_CV()

            if which_data != 'two_moons':
                Xtest = (Xtest - torch.mean(Xtrain)) / torch.std(Xtrain)
                dataset = DataLoaderInput(Xtest, ytest, which_data=which_data)
                test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                          batch_size=1,
                                                          shuffle=False)


            elif which_data == 'two_moons':
                test_data = make_moons(n_samples=1000, noise=0.3, random_state=12)
                Xtest, ytest = test_data
                Xtest, ytest = torch.tensor(Xtest, dtype=torch.float32), torch.tensor(ytest)
                dataset = DataLoaderInput(Xtest, ytest, which_data=which_data)
                test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                          batch_size=1,
                                                          shuffle=False)

            criterion = torch.nn.CrossEntropyLoss()
            # Load model etc.
            if which_model == 'deterministic':
                Net = Deterministic_net(input_dim, int(opti_params['hidden_dim']), output_dim,
                                        drop_out_rate=drop_out_rate)
                model_path = 'Deterministic/' + which_data + '/bo/models/m_' + str(int(argmin))
                Net.load_state_dict(torch.load(model_path))

                # compute accuracy, average loss, all probs, average loss with reg, all loss
                if in_dist:
                    accuracy, loss_ave, all_probs, loss_l2_ave, all_loss = Net.test_net(test_loader=test_loader,
                                                                                        criterion=criterion,
                                                                                        temp=temp, l2=opti_params['l2'],
                                                                                        test=True)
                elif not in_dist:
                    all_probs = Net.test_net_out_dist(test_loader, temp)

            elif which_model.lower() == 'kfac':
                Net = KFAC(input_dim=input_dim, hidden_dim=int(opti_params['hidden_dim']), output_dim=output_dim,
                           momentum=opti_params['momentum'], l2_param=opti_params['l2'], L=3,
                           drop_out_rate=drop_out_rate)

                if in_dist:
                    p_yxw, all_probs, all_loss, accuracy, loss_l2 = monte_carlo_bma(Net, Xtest, ytest, S=30,
                                                                                    C=output_dim, temp=temp,
                                                                                    criterion=criterion,
                                                                                    path_to_bma=path_to_bma,
                                                                                    save_probs='', batch=True,
                                                                                    dropout=drop_out_rate,
                                                                                    l2=opti_params['l2'])
                else:
                    all_probs, p_yxw = monte_carlo_bma_out(Net, Xtest, S=30, C=output_dim, temp=temp,
                                                           path_to_bma=path_to_bma)


            elif which_model.lower() == 'swag':
                Net = Swag(input_dim=input_dim, hidden_dim=int(opti_params['hidden_dim']), output_dim=output_dim, K=20,
                           c=1, S=30, criterion=criterion, l2_param=opti_params['l2'], drop_out_rate=drop_out_rate)

                if in_dist:
                    p_yxw, all_probs, all_loss, accuracy, loss_l2 = monte_carlo_bma(Net, Xtest, ytest, S=30,
                                                                                    C=output_dim, temp=temp,
                                                                                    criterion=criterion,
                                                                                    path_to_bma=path_to_bma,
                                                                                    save_probs='', l2=opti_params['l2'],
                                                                                    batch=True, dropout=drop_out_rate)

                else:
                    all_probs, p_yxw = monte_carlo_bma_out(Net, Xtest, S=30, C=output_dim, temp=temp,
                                                           path_to_bma=path_to_bma)


            elif which_model.lower() == 'ensample':
                Net = Deterministic_net(input_dim=input_dim, hidden_dim=int(opti_params['hidden_dim']),
                                        output_dim=output_dim, drop_out_rate=drop_out_rate)

                if in_dist:
                    p_yxw, all_probs, all_loss, accuracy, loss_l2 = monte_carlo_bma(model=Net, Xtest=Xtest, ytest=ytest,
                                                                                    S=5, C=output_dim, temp=temp,
                                                                                    criterion=criterion,
                                                                                    l2=opti_params['l2'], forplot=False,
                                                                                    path_to_bma=path_to_ensample,
                                                                                    batch=True, which_data=which_data,
                                                                                    dropout=drop_out_rate)
                else:
                    all_probs, p_yxw = monte_carlo_bma_out(Net, Xtest, S=5, C=output_dim, temp=temp,
                                                           path_to_bma=path_to_ensample)



            # Compute standard error of nll loss (CLT says you can do it)
            C = len(all_probs[0])
            N = len(all_loss)
            s = torch.std(torch.tensor(all_loss))
            SE = s / torch.sqrt(torch.tensor(N))
            nll = sum(all_loss) / N

            # Reliability diagrams
            M = 40
            I = [((m - 1) / M, m / M) for m in range(1, M + 1)]
            I[-1] = (I[-1][0], I[-1][0] + 0.1)
            Counter = {i: {'correct': 0, 'incorrect': 0, 'probs': []} for i in I}

            for probs, y in zip(all_probs, dataset.y):
                correct = probs[y]
                rest = torch.cat((probs[0:y], probs[y:-1]), 0)
                for key in Counter.keys():
                    if correct >= key[0] and correct < key[1]:
                        Counter[key]['correct'] += 1
                        Counter[key]['probs'].append(correct.item())
                    for p in rest:
                        if p >= key[0] and p < key[1]:
                            Counter[key]['incorrect'] += 1
                            Counter[key]['probs'].append(p.item())

            acc = [Counter[i]['correct'] / (Counter[i]['correct'] + Counter[i]['incorrect']) for i in Counter.keys()]
            conf = [sum(Counter[i]['probs']) / (Counter[i]['correct'] + Counter[i]['incorrect']) for i in
                    Counter.keys()]

            # expected calibration error

            ece_all = [(Counter[key]['correct'] + Counter[key]['incorrect']) / (N * C) * abs(acc[i] - conf[i]) for
                       i, key in enumerate(Counter.keys())]
            ece = sum(ece_all)
            ece_se = torch.std(torch.tensor(ece_all)) / torch.sqrt((torch.tensor(N * C)))

            # calibration hist
            fig, ax = plt.subplots()
            sns.set_style('darkgrid')
            bins = [key[0] for key in Counter.keys()] + [1]
            ax.hist(bins[:-1], bins, weights=acc)
            ax.plot([0, 1], [0, 1], transform=ax.transAxes)
            ax.set_xlabel('Confidence', fontsize='large', fontweight=300, )
            ax.set_ylabel('Accuracy', fontsize='large', fontweight=300)
            props = dict(facecolor='gray', alpha=0.6)
            txt_string = 'ECE: {:.4}'.format(ece)
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, txt_string, transform=ax.transAxes, fontsize=17,
                    verticalalignment='top', bbox=props)
            ax.set_title(title_cal, fontsize=20, fontweight=300, fontfamily='sans-serif')
            plt.savefig('moons/cal_plots/cal_' + which_data + '_' + which_model + '_' + what_to_do, dpi=500)

            plt.close()

            print('Accuracy ${:.4}$'.format(accuracy))
            print('NLL ${:.4} \pm {:.4}$'.format(nll, SE))
            print('ECE ${:.4} \pm {:.4}$'.format(ece, ece_se))
            print('---------------------------------------------' * 5)

            # decision boundary
            if which_data == 'two_moons':
                if which_model == 'ensample':
                    plot_decision_boundary(Net, test_loader, S=5, temp=temp, path_to_bma=path_to_ensample,
                                           title=decision_title, predict_func=predict_func_boundary,
                                           save_image_path="moons/cal_plots/" + which_model + '_' + what_to_do,
                                           sample_new_weights=False,
                                           text_string=decision_text, dropout=drop_out_rate)
                else:
                    plot_decision_boundary(Net, test_loader, S=30, temp=temp, path_to_bma=path_to_bma,
                                           title=decision_title, predict_func=predict_func_boundary,
                                           save_image_path="moons/cal_plots/test_" + which_model + '_' + what_to_do,
                                           sample_new_weights=False,
                                           text_string=decision_text, dropout=drop_out_rate)



