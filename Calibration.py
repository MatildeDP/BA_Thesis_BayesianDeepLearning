from data import LoadDataSet, settings, DataLoaderInput
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from run import run_swag, run_kfac, train_deterministic
from copy import deepcopy
from deterministic import Deterministic_net
from utils import dump_to_json, plot_decision_boundary
from SWAG import Swag
from BMA import monte_carlo_bma
from KFAC import KFAC
import json
import numpy as np



# load data
def optimise_with_temp(temp_range, model_type, which_data, X_test, y_test, optimal_model_name, hidden_dim, l2, dropout):
    """
    Optimise temperature scale
    :param temp_range: temperature scala range
    :param model_type: which model to optimise
    :param which_data: what data to optimise
    :param X_test: test data
    :param y_test: test labels
    :param optimal_model_name:  path to optimal model (to load it for temp scaling)
    :param hidden_dim: optimal hidden dims
    :param l2: optimal l2
    :param dropout: optimal dropout
    :return:
    """
    criterion = torch.nn.CrossEntropyLoss()

    if which_data == 'mnist' or which_data == 'emnist' or which_data == 'fashion':
        X_test = (X_test - torch.mean(X_test)) / torch.std(X_test)

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test, which_data = which_data),
                                              batch_size=1000,
                                              shuffle=False)

    input_dim, output_dim = settings(which_data)

    evals = {'temp':[], 'acc': [],  'test_loss': [], 'test_loss_l2': []}
    if model_type == 'deterministic':

        for idx_models, temp in enumerate(temp_range):
            temp = temp.item()
            print('temperature: {}'.format(temp))

            info_PATH = 'Deterministic/' + which_data + '/calibrating/params_evals/info.json'
            decision_path = 'Deterministic/' + which_data + '/calibrating/plots/' + str(
                idx_models) + '_decision_boundary.jpg'

            # path to optimal model params
            load_model_path = 'Deterministic/' + which_data + '/bo/models/' + str(optimal_model_name)
            load_opti_path = 'Deterministic/' + which_data + '/bo/opti/' + str(optimal_model_name)

            Net = Deterministic_net(input_dim, hidden_dim, output_dim, dropout)
            Net.load_state_dict(torch.load(load_model_path))  # load pretrained net
            optimizer = torch.optim.SGD(Net.parameters(), lr=0, momentum = 0)
            optimizer.load_state_dict(torch.load(load_opti_path))

            accuracy, loss_ave, all_probs, loss_l2_ave = Net.test_net(test_loader=test_loader, criterion=criterion, temp=temp, l2 = l2, freq=0)

            evals['acc'].append(accuracy.item())
            evals['test_loss'].append(loss_ave.item())
            evals['test_loss_l2'].append(loss_l2_ave)
            evals['temp'].append(temp)

            if which_data == 'two_moons':
                plot_decision_boundary(Net, dataloader=test_loader, S=20,
                                       title="Single model decision boundary",
                                       save_image_path=decision_path, temp=temp, text_string = 'Temperature: {:.3f}'.format(temp))

    if model_type == 'ensample':
        for idx_models, temp in enumerate(temp_range):
            temp = temp.item()
            print('temperature: {}'.format(temp))

            path_to_bma = "Deterministic/" + which_data + '/ensample/models/'
            info_PATH = 'Deterministic/' + which_data + '/ensample_temp/info.json'
            decision_path = 'Deterministic/' + which_data + '/ensample_temp/' + str(
                idx_models) + '_decision_boundary.jpg'


            Net = Deterministic_net(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout = dropout)
            p_yxw, all_probs, loss, accuracy, loss_l2 = monte_carlo_bma(model=Net, Xtest=X_test, ytest=y_test, S=5,
                                                                            C=output_dim, temp=temp,
                                                                            criterion=criterion, l2=l2,
                                                                             path_to_bma=path_to_bma,
                                                                            which_data=which_data, dropout = dropout)


            evals['acc'].append(accuracy.item())
            evals['test_loss'].append(loss)
            evals['test_loss_l2'].append(loss_l2)
            evals['temp'].append(temp)

            if which_data == 'two_moons':
                plot_decision_boundary(Net, dataloader=test_loader, S=5,
                                       title="Ensample decision boundary",
                                       save_image_path=decision_path, temp=temp, sample_new_weights = False, predict_func= 'stochastic', path_to_bma = path_to_bma,
                                       text_string = 'Temperature: {:.3f}'.format(temp), dropout = dropout)


    if model_type == 'swag':
        info_PATH = 'SWAG/' + which_data + '/calibrating/params_evals/info.json'

        opti_info_path = 'SWAG/'+which_data+'/gridsearch/params_evals/info.json'
        f = open(opti_info_path)
        data = json.load(f)
        loss = [data['key'][i][str(i)]['evals:']['test_loss'][-1] for i in range(len(data['key']))]
        opti_model_idx = np.argmin(loss)

        path_to_bma = 'SWAG/' + which_data + "/gridsearch/bma/idx_models_" + str(opti_model_idx) +"temp_1/"
        for idx_models, temp in enumerate(temp_range):
            decision_path = 'SWAG/' + which_data + '/calibrating/plots/' + str(
                idx_models) + '_decision_boundary.jpg'
            temp = temp.item()
            print('temperature: {}'.format(temp))


            model = Swag(input_dim = input_dim, hidden_dim= hidden_dim, output_dim = output_dim, K = 20, c = None, S = 30, criterion = criterion, l2_param = l2, dropout = dropout)
            p_yxw, p_yx, loss, acc, loss_l2 = monte_carlo_bma(model = model, Xtest = X_test, ytest = y_test, S = 30, C = output_dim, temp = temp, criterion = criterion, l2 = l2, save_models='',
                            path_to_bma=path_to_bma, dropout = dropout)

            evals['acc'].append(acc.item())
            evals['test_loss'].append(loss)
            evals['test_loss_l2'].append(loss_l2)
            evals['temp'].append(temp)

            if which_data == 'two_moons':
                plot_decision_boundary(model, dataloader=test_loader, S=30,
                                       title="SWAG decision boundary",
                                       save_image_path=decision_path, temp=temp, sample_new_weights = False, predict_func= 'stochastic', path_to_bma = path_to_bma,
                                       text_string = 'Temperature: {:.3f}'.format(temp), dropout = dropout)


    if model_type == 'kfac':
        info_PATH = 'KFAC/' + which_data + '/calibrating/params_evals/info.json'

        opti_info_path = 'KFAC/'+which_data+'/gridsearch/params_evals/info.json'
        f = open(opti_info_path)
        data = json.load(f)
        loss = [data['key'][i][str(i)]['evals:']['test_loss']for i in range(len(data['key']))]
        opti_model_idx = np.argmin(loss)

        path_to_bma = 'KFAC/' + which_data + "/gridsearch/bma/idx_models_" + str(opti_model_idx) +"temp_1/"
        for idx_models, temp in enumerate(temp_range):
            decision_path = 'KFAC/' + which_data + '/calibrating/plots/' + str(
                idx_models) + '_decision_boundary.jpg'
            temp = temp.item()
            print('temperature: {}'.format(temp))


            model = KFAC(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, momentum = '',l2_param = l2, L = 3, dropout = dropout)
            p_yxw, p_yx, loss, acc, loss_l2 = monte_carlo_bma(model = model, Xtest = X_test, ytest = y_test, S = 30, C = output_dim, temp = temp, criterion = criterion, l2 = l2, save_models='',
                            path_to_bma=path_to_bma, dropout = dropout)

            evals['temp'].append(temp)
            evals['acc'].append(acc.item())
            evals['test_loss'].append(loss)
            evals['test_loss_l2'].append(loss_l2)

            if which_data == 'two_moons':
                plot_decision_boundary(model, dataloader=test_loader, S=30,
                                       title="KFAC decision boundary",
                                       save_image_path=decision_path, temp=temp, sample_new_weights = False, predict_func='stochastic', path_to_bma = path_to_bma,
                                       text_string = 'Temperature: {:.3f}'.format(temp), dropout = dropout)





    dump_to_json(info_PATH, evals)







def optimise(alpha_range, params: dict, load_net_path: str, load_opti_path: str, model_type: str, X_train, y_train, X_test_, y_test_, criterion, which_data, test_each_epoch, dropout):
    """
    Optimise tau or learning rate (KFAC, SWAG)
    :param alpha_range: range of parameter
    :param params: other nessesary parameters
    :param load_net_path: path to net
    :param load_opti_path: path to net
    :param model_type: which model to optimise
    :param X_train: Train data
    :param y_train: Train labels
    :param X_test: validation data
    :param y_test: validation labels
    :param criterion: loss
    :param which_data: what data to use
    :return:
    """


    if model_type == 'swag':

        # Grid search loop
        for idx_models, alpha in enumerate(alpha_range):
            alpha = alpha.item()
            path_to_bma = 'SWAG/'+which_data+'/gridsearch/bma/'
            info_PATH = 'SWAG/'+which_data+'/gridsearch/params_evals/info.json'
            #path_to_bma_probs = 'SWAG/' + which_data + '/gridsearch/probs/lr_' + str(alpha)  + 'json'
            path_to_bma_probs = ''
            X_test = deepcopy(X_test_)
            y_test = deepcopy(y_test_)

            # paths to plots
            decision_path = 'SWAG/' + which_data + '/gridsearch/plots/' + str(idx_models) + '_decision_boundary.jpg'
            loss_path = 'SWAG/' + which_data + '/gridsearch/plots/' + str(idx_models) + "_loss_acc.jpg"  # loss plots is created if we test every epoch

            run_swag(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                      criterion = criterion, lr = alpha, idx_models = idx_models,
                      load_net_path = load_net_path, load_opti_path = load_opti_path, info_PATH = info_PATH,
                      params = params, which_data = which_data, temp = 1, path_to_bma = path_to_bma,
                     decision_path = decision_path, loss_path = loss_path, test_each_epoch = test_each_epoch, save_probs_path=path_to_bma_probs, count=0, dropout = dropout)

    if model_type == 'kfac':

        for idx_models, alpha in enumerate(alpha_range):
            alpha = alpha.item()
            path_to_bma = 'KFAC/'+which_data+'/gridsearch/bma/'
            info_PATH = 'KFAC/'+which_data+'/gridsearch/params_evals/info.json'
            #path_to_bma_probs = 'KFAC/' + which_data + '/gridsearch/probs/lr_' + str(alpha) + 'json'
            path_to_bma_probs = ''

            # paths to plots
            decision_path = 'KFAC/' + which_data + '/gridsearch/plots/' + str(idx_models) + '_decision_boundary.jpg'
            loss_path = 'KFAC/' + which_data + '/gridsearch/plots/' + str(idx_models) + "_loss_acc.jpg"  # loss plots is created if we test every epoch
            N = X_test_.shape[0]

            X_test = deepcopy(X_test_)
            y_test = deepcopy(y_test_)

            run_kfac(tau = alpha, params = params, X_train = 0, y_train = 0,
                     X_test = X_test, y_test = y_test, criterion = criterion, which_data = which_data,
                    load_net_path = load_net_path, load_opti_path = load_opti_path,
                    loss_path = loss_path, decision_path = decision_path, path_to_bma = path_to_bma, idx_model = idx_models,
                    test_each_epoch = test_each_epoch, temp = 1, N = N, info_PATH = info_PATH, save_probs_path=path_to_bma_probs, count = 0, dropout = dropout)






