from run import run_kfac
import torch
import torch.nn as nn
import json
import numpy as np
from data import  LoadDataSet, settings
from sklearn.model_selection import train_test_split
from Calibration import optimise, optimise_with_temp

if __name__ == '__main__':

    test_each_epoch = True
    which_data = 'two_moons'
    action = 'optimise'  # run_once, optimise, calibrate

    tau_range = torch.logspace(1.1,1.7,30)  # range to optimise swag over
    tau = 20  # value when training once
    temp_range = torch.logspace(-2, 1, 100)
    temp = 1


    param_path = 'Deterministic/' + which_data + '/bo/params_evals/model_info_BO.json'
    f = open(param_path)
    data = json.load(f)
    N = len(data['key'])
    all_last_testloss = [data['key'][i][str(i)]['evals']['test_loss'][-1] for i in range(N)]
    model_idx = np.argmin(all_last_testloss)
    opti_params = data['key'][model_idx][str(model_idx)]['params']
        
    if which_data == 'two_moons':
        drop_out_rate = 0.05
    elif which_data == 'mnist':
        drop_out_rate = 0.1
    elif which_data == 'fashion':
        drop_out_rate =  0.3
    elif which_data == 'emnist':
        drop_out_rate = 0.05

    params = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                   'batch_size': 1,
                   'n_epochs': 1,
                   'seed': 0,
                   'hidden_dim': int(opti_params['hidden_dim']),
                   'lr': opti_params['lr'],
                   'l2': opti_params['l2'],
                   'momentum': opti_params['momentum'],
                   'S': 30,
                   'dropout': drop_out_rate}


    # Loads model trained with optimal parameters
    pretrained_net_path = 'Deterministic/' + which_data + '/bo/models/m_'+ str(model_idx) #/optimal/model'
    pretrained_opti_path = 'Deterministic/' + which_data + '/bo/opti/m_' +  str(model_idx) #/optimal/optimizer'

    # Load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()
    input_dim, output_dim = settings(which_data)
    if which_data != 'two_moons':
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    else:
        _, _, X_test, y_test = X, y, Xval, yval

   # _, X_test, _, y_test = train_test_split(X_, y_, test_size=0.2, random_state=0, stratify=y_)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # optimise chosen method
    if action == 'optimise':
        optimise(alpha_range=tau_range, params=params, load_net_path=pretrained_net_path,
                 load_opti_path=pretrained_opti_path, model_type='kfac', criterion=criterion,
                 X_train=0, y_train=0, X_test_=X_test, y_test_=y_test, which_data=which_data,
                 test_each_epoch = test_each_epoch, dropout = drop_out_rate)

    elif action == 'calibrate':
        optimise_with_temp(temp_range=temp_range, model_type='kfac', which_data=which_data, X_test=X_test,
                           y_test=y_test,
                           optimal_model_name='', hidden_dim=int(opti_params['hidden_dim']), l2=opti_params['l2'], dropout = drop_out_rate)

 
