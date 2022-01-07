import torch
import torch.nn as nn
from deterministic import Deterministic_net
from data import DataLoaderInput, settings
from utils import plot_decision_boundary, plot_acc_and_loss, dump_to_json, dump_to_existing_json
from SWAG import Swag
from BMA import monte_carlo_bma
import json
import os

from KFAC import KFAC


def run_swag(X_train, y_train, X_test, y_test, criterion, lr, idx_models, load_net_path, load_opti_path, info_PATH,
              params, which_data, temp, path_to_bma, decision_path, loss_path, save_probs_path, count,
              dropout, test_each_epoch=False):
    """
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data (for holdout)
    :param y_test: Test labels (for holdout)
    :param criterion: nn Cross entropy
    :param lr: learning rate value
    :param idx_models: which models i tested
    :param load_net_path: Path to pretrained nework
    :param load_opti_path: Path to optimizer belonging to pretrained network
    :param info_PATH: path where parameter values + evaluations are saved in json
    :param params: parameters to test (except lr)
    :param which_data: what data to run in
    :param temp: temperature scale, if any
    :param do_return: if ..... should be returned (for plots etc)
    :param test_each_epoch: if test with BMA should be performed each epoch or not
                            if True, BMA models will not be saved. If False, BMA models will be saved
    :return:
    """

    # set seed
    torch.manual_seed(params['seed'])

    # Define model and optimizer
    input_dim, output_dim = settings(which_data)
    model = Swag(input_dim=input_dim, hidden_dim=params['hidden_dim'], output_dim=output_dim,
                 K=params['K'], c=params['c'], S=params['S'],
                 criterion=criterion, l2_param=params['l2'], drop_out_rate = dropout)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Re define model and optimizer to ensure same start conditions
    model.load_state_dict(torch.load(load_net_path))  # load pretrained net
    optimizer.load_state_dict(torch.load(load_opti_path))  # load optimizer

    for g in optimizer.param_groups:
        g['lr'] = lr
        g['weight_decay'] = params['l2']
        g['momentum'] = params['momentum']

    print('Training model {} with parameters: \n Hidden dimension: {} \n l2: {} '
          '\n learning rate: {} \n momentum: {} \n batch size: {} \n c(update freq): {}. \n K(Dhat rank): {} \n S(# samples in BMA): {}'.format(
        idx_models, params['hidden_dim'],
        params['l2'], lr,
        params['momentum'],
        params['batch_size'], params['c'], params['K'], params['S']))


    # Get train and test data
    if which_data == 'mnist' or which_data == 'emnist' or which_data == 'fashion':
        X_test = (X_test - torch.mean(X_test)) / torch.std(X_test)
        X_train = (X_train - torch.mean(X_train)) / torch.std(X_train)

    elif which_data == 'cancer':
        X_test = (X_test - torch.mean(X_test, dim = 0)) / torch.std(X_test, dim = 0)
        X_train = (X_train - torch.mean(X_train, dim = 0)) / torch.std(X_train, dim = 0)

        # Torch dataloader instance
    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_train, y_train, which_data = which_data),
                                               batch_size=params['batch_size'],
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test, which_data = which_data),
                                              batch_size=1000,
                                              shuffle=False)

    # Create new dir to save models to
    new_path_to_bma = path_to_bma + "idx_models_" + str(idx_models) + 'temp_' + str(temp)
    try:
        os.mkdir(new_path_to_bma)
    except FileExistsError:
        pass

    # Train and test
    # Extract initial weights
    theta1, theta2 = model.get_ith_moments()

    # Initialize error and accuracy container
    n = 0
    print("Swag is training..")
    # Run T epoch of training and testing

    if test_each_epoch:
        # Initialise data container
        evals = {'acc': [], 'train_loss': [], 'test_loss': [], 'test_loss_l2':[],'train_loss_l2':[] }
        for i in range(params['n_epochs']):

            # Train swag
            n, train_Loss, optimizer, theta1, theta2, train_loss_l2_ave = model.train_swag(epoch=i, n=n, theta1=theta1, theta2=theta2,
                                                                        train_loader=train_loader,
                                                                        test_loader=test_loader, optimizer=optimizer)


            # Collect average acc and loss per fold (overwrites previous model info)
            evals['train_loss'].append(train_Loss)
            evals['train_loss_l2'].append(train_loss_l2_ave.item())

            print('Epoch: {}.......Train loss: {}.'.format(i, evals['train_loss'][-1]))
            # Test on trained model
            if i >= params['K']+1:

                # do not save BMA models (save space)
                if i < params['n_epochs']-1:
                    p_yxw, p_yx, test_loss, acc, test_loss_l2 = monte_carlo_bma(model=model, S=params['S'], Xtest=X_test, ytest=y_test,
                                                          C=output_dim, temp=temp, criterion= criterion, l2 = params['l2'], dropout = dropout)


                # save BMA models if epoch is the last
                else:
                    p_yxw, p_yx, test_loss, acc, test_loss_l2 = monte_carlo_bma(model=model, S=params['S'], Xtest=X_test, ytest=y_test,
                                                                  C=output_dim, temp=temp, criterion=criterion,
                                                                  save_models = new_path_to_bma + '/', save_probs='',
                                                                                l2 = params['l2'], dropout = dropout)

                # collect
                evals['acc'].append(acc.item())
                evals['test_loss'].append(test_loss)
                evals['test_loss_l2'].append(test_loss)



                print('Epoch: {} .......Test loss: {}.........Accuracy: {}.'.format(i, evals['test_loss'][-1],
                                                                                             evals['acc'][-1]))
            else:
                # collect
                evals['acc'].append(0)
                evals['test_loss'].append(0)
                evals['test_loss_l2'].append(0)


        plot_acc_and_loss(testloss=evals['test_loss'], trainloss=evals['train_loss'], accuracy=evals['acc'],
                          save_path=loss_path)

    else:
        # Initialise data container
        evals = {'acc': None, 'train_loss': [], 'test_loss': None, 'test_loss_l2': None, 'train_loss_l2': None}

        for i in range(params['n_epochs']):
            if i % 100 == 0:
                print("Epoch {}".format(i))

            # Train swag
            n, train_Loss, optimizer, theta1, theta2,  train_loss_l2= model.train_swag(epoch=i, n=n, theta1=theta1, theta2=theta2,
                                                                        train_loader=train_loader,
                                                                        test_loader=test_loader, optimizer=optimizer)
            evals['train_loss'].append(train_Loss)
            evals['train_loss_l2'].append(train_loss_l2)


        # Test on trained model
        p_yxw, p_yx, test_loss, acc, test_loss_l2 = monte_carlo_bma(model=model, S=model.S, Xtest=X_test, ytest=y_test,
                                                      C=output_dim, temp=temp,criterion = criterion,
                                                      save_models=new_path_to_bma+ '/', save_probs = save_probs_path, l2 = params['l2'], dropout = dropout)

        # Collect average acc and loss per fold (overwrites previous model info)
        evals['test_loss'] = test_loss
        evals['acc'] = acc.item()
        evals['test_loss_l2']= test_loss_l2

        print('.......Train loss: {}.      Test loss: {}       Accuracy: {}.'.format(evals['train_loss'][-1],
                                                                                     evals['test_loss'], evals['acc']))

    if which_data == 'two_moons':
        # plot decision boundary with saved models on latest created directory
        plot_decision_boundary(model, test_loader, S=params['S'],
                               title="SWAG decision boundary", predict_func='stochastic',
                               save_image_path=decision_path, temp = temp, sample_new_weights = False,
                               path_to_bma = new_path_to_bma  + '/', text_string = 'Learning rate: {:.3f}'.format(lr), dropout = dropout)


    print('-' * 100)
    print('-' * 100)

    params_temp = params.copy()
    params_temp['batch_size'] = str(params['batch_size'])  # stringify
    params_temp['hidden_dim'] = str(params['hidden_dim'])  # stringify
    params_temp['lr'] = lr
    params_temp['temp'] = temp
    if idx_models == 0:# and count ==0:  # create json files if model is the first
        model_ = {idx_models: {'params': params_temp, 'evals:': evals}}
        dump_to_json(info_PATH, model_)

    else: # dump to existing
        model_ = {idx_models: {'params': params_temp, 'evals:': evals}}
        dump_to_existing_json(info_PATH, model_)
    #elif count == 0:
    #    model_ = {count: {'params': params_temp, 'evals:': evals}}
    #    dump_to_json(info_PATH, model_)

    #elif idx_models != 0 and count == 0:  # dump to existing
       # model_ = {idx_models: {'params': params_temp, 'evals:': evals}}
      #  dump_to_existing_json(info_PATH, model_)

    #elif count != 0:  # dump to existing
    #    model_ = {count: {'params': params_temp, 'evals:': evals}}
    #    dump_to_existing_json(info_PATH, model_)



def train_deterministic(which_data: str, net_path: str, opti_path: str, infoPATH: str, param_space: dict,
                        temp, X_train, y_train, X_test, y_test, decision_path,loss_path, idx_model,dropout,
                        test_each_epoch = False):


    # set seed
    torch.manual_seed(param_space['seed'])

    # get dims
    input_dim, output_dim = settings(which_data)

    if which_data == 'mnist' or which_data == 'emnist' or which_data == 'fashion':
        X_test = (X_test - torch.mean(X_test)) / torch.std(X_test)
        X_train = (X_train - torch.mean(X_train)) / torch.std(X_train)

    elif which_data == 'cancer':
        X_test = (X_test - torch.mean(X_test, dim = 0)) / torch.std(X_test, dim = 0)
        X_train = (X_train - torch.mean(X_train, dim = 0)) / torch.std(X_train, dim = 0)


    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_train, y_train, which_data = which_data),
                                               batch_size=param_space['batch_size'],
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test, which_data = which_data),
                                              batch_size=1000,
                                              shuffle=False)


    # Define model, optimizer, criterion
    model = Deterministic_net(input_dim, param_space['hidden_dim'], output_dim, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=param_space['lr'],
                                weight_decay=param_space['l2'],
                                momentum=param_space['momentum'])



    if test_each_epoch:
        evals = {'acc': [], 'train_loss': [], 'test_loss': [], 'test_loss_l2':[],'train_loss_l2':[] }
        for epoch in range(param_space['n_epochs']):
            print("Epoch {}".format(epoch))

            # One epoch of training
            optimizer, train_loss_, train_loss_l2 = model.train_net(train_loader, optimizer, criterion,l2 =param_space['l2'], save_net=False)

            # Test
            accuracy, tst_loss, _, test_loss_l2= model.test_net(test_loader = test_loader, criterion = criterion,l2 =param_space['l2'], temp = temp, freq=0)

            # Collect loss
            evals['train_loss'].append(float(train_loss_))
            evals['test_loss'].append(float(tst_loss.detach().numpy()))
            evals['acc'].append(accuracy.item())
            evals['test_loss_l2'].append(test_loss_l2)
            evals['train_loss_l2'].append(train_loss_l2)

        # save models and optimizers
        torch.save(model.state_dict(), net_path)
        torch.save(optimizer.state_dict(), opti_path)

        print('After {} epochs......Test Loss: {}. Train Loss: {}. Accuracy: {}'.format(param_space['n_epochs'], evals['test_loss'][-1],
                                                                                          evals['train_loss'][-1], evals['acc'][-1]))

    else:
        evals = {'acc': None, 'train_loss': None, 'test_loss': None, 'train_loss_l2':None, 'test_loss_l2':None}
        for epoch in range(param_space['n_epochs']):

            # One epoch of training
            optimizer, train_loss_, train_loss_l2 = model.train_net(train_loader = train_loader, optimizer = optimizer,
                                                     criterion = criterion, save_net=False,
                                                     net_path=net_path, opti_path=opti_path, l2 =param_space['l2'])

            evals['train_loss'] = float(train_loss_)
            evals['train_loss_l2'] = train_loss_l2

        # save models and optimizers
        torch.save(model.state_dict(), net_path)
        torch.save(optimizer.state_dict(), opti_path)

        # Test
        accuracy, tst_loss, _, test_loss_l2 = model.test_net(test_loader=test_loader, criterion=criterion,
                                               freq=30, temp = temp,l2 =param_space['l2'])

        evals['test_loss'] = float(tst_loss.detach().numpy())
        evals['acc'] = accuracy.item()
        evals['test_loss_l2'] = test_loss_l2.item()

        print('After {} epochs......Test Loss: {}. Train Loss: {}. Accuracy: {}'.format(param_space['n_epochs'], evals['test_loss'],
                                                                                          evals['train_loss'], evals['acc']))


    # Final plots
    if which_data == 'two_moons':
        plot_decision_boundary(model, dataloader=test_loader, S=20,
                               title="Single model decision boundary".format(temp),
                               save_image_path=decision_path, temp = temp, show_text = False, text_string = '', dropout = dropout)


    plot_acc_and_loss(testloss= evals['test_loss'], trainloss=evals['train_loss'], accuracy=evals['acc'], save_path=loss_path)


    params_temp = param_space.copy()
    params_temp['batch_size'] = str(param_space['batch_size'])  # stringify
    params_temp['hidden_dim'] = str(param_space['hidden_dim'])  # stringify
    params_temp['temp'] = temp
    if idx_model == 0:  # create json files if model is the first
        model_ = {idx_model: {'params': params_temp, 'evals:': evals}}
        dump_to_json(infoPATH, model_)
    else:  # dump to existing
        model_ = {idx_model: {'params': params_temp, 'evals:': evals}}
        dump_to_existing_json(infoPATH, model_)





def run_kfac(tau, params, load_net_path, load_opti_path, X_train, y_train, X_test, y_test, criterion, which_data,
            loss_path, decision_path, path_to_bma, idx_model, test_each_epoch, N, temp, info_PATH, save_probs_path, count, dropout):

    """
    :param criterion: cross entropy loss
    :param tau: "hacked" prior param
    :param idx_models: model counter
    :param kfold: sklean stratifiedFold
    :param X: independent data
    :param y: dependent data
    :param load_net_path: NN.Net class network, trained
    :param load_opti_path: optimizer belonging to trained network
    :param info_PATH: information is dumped here
    :param params: dict with parameters (except tau and temp)
    :param which_data: dataset name
    :param temp: calibration parameter
    :return:
    """

    # set seed
    torch.manual_seed(params['seed'])

    # Create new dir to save models to
    new_path_to_bma = path_to_bma + "idx_models_" + str(idx_model) + 'temp_' + str(temp)
    try:
        os.mkdir(new_path_to_bma)
    except FileExistsError:
        pass

    # Define model and optimizer
    input_dim, output_dim = settings(which_data)

    model = KFAC(input_dim= input_dim, hidden_dim=params['hidden_dim'], output_dim = output_dim, drop_out_rate = dropout,
                      momentum = params['momentum'],l2_param = params['l2'], L = 3)


    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'] )

    # Re define model and optimizer to ensure same start conditions
    model.load_state_dict(torch.load(load_net_path))  # load pretrained net
    optimizer.load_state_dict(torch.load(load_opti_path))  # load optimizer

    for g in optimizer.param_groups:
        g['lr'] = params['lr']
        g['weight_decay'] = params['l2']
        g['momentum'] = params['momentum']

    print('Training model {} with parameters: \n Hidden dimension: {} \n l2: {} '
          '\n learning rate: {} \n momentum: {} \n batch size: {} \n tau {}. \n \n S(# samples in BMA): {}'.format(
        idx_model, params['hidden_dim'],
        params['l2'],params['lr'],
        params['momentum'],
        params['batch_size'],tau, params['S']))

    # Get train and test data
    if which_data == 'mnist' or which_data == 'emnist' or which_data == 'fashion':
        X_test = (X_test - torch.mean(X_test)) / torch.std(X_test)
        #X_train = (X_train - torch.mean(X_train)) / torch.std(X_train)

    elif which_data == 'cancer':
        X_test = (X_test - torch.mean(X_test, dim = 0)) / torch.std(X_test, dim = 0)
       # X_train = (X_train - torch.mean(X_train, dim = 0)) / torch.std(X_train, dim = 0)



    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test, which_data = which_data),
                                              batch_size=1,
                                              shuffle=False)



    # Initialise data container
    evals = {'acc': [], 'train_loss': None, 'test_loss': [], 'test_loss_l2':[]}

    # Create distribution
    train_loss = model.collect_values(test_loader, optimizer, criterion, temp)
    model.regularize_and_add_prior(tau=tau, N=N)

    # Test KFAC and save models
    p_yxw, p_yx, test_loss, acc, test_loss_l2 = monte_carlo_bma(model=model, S=params['S'], Xtest=X_test, ytest=y_test,
                                                      C=output_dim, temp=temp, criterion=criterion,
                                                      save_models = new_path_to_bma + '/', save_probs = save_probs_path, l2  = params['l2'], dropout = dropout)

    if p_yxw == None:
        return


    # Collect average acc and loss per fold (overwrites previous model info)
    evals['test_loss'] = test_loss
    evals['acc'] = acc.item()
    evals['train_loss'] = train_loss
    evals['test_loss_l2']= test_loss_l2


    print('.......Test loss: {}       Accuracy: {}.'.format(evals['test_loss'],evals['acc']))

    #plot_acc_and_loss(testloss=evals['test_loss'], trainloss=evals['train_loss'], accuracy=evals['acc'],
                     # save_path=loss_path)



    if which_data == 'two_moons':
        plot_decision_boundary(model, test_loader, S=params['S'],
                               title="KFAC decision boundary", predict_func='stochastic',
                               save_image_path=decision_path, temp = temp, text_string = "tau: {:.3f}".format(tau), dropout = dropout)



    print('-' * 100)
    print('-' * 100)

    params_temp = params.copy()
    params_temp['batch_size'] = str(params['batch_size'])  # stringify
    params_temp['hidden_dim'] = str(params['hidden_dim'])  # stringify
    params_temp['tau'] = tau
    params_temp['temp'] = temp


    if idx_model == 0:  # and count ==0:  # create json files if model is the first
        model_ = {idx_model: {'params': params_temp, 'evals:': evals}}
        dump_to_json(info_PATH, model_)

    else:  # dump to existing
        model_ = {idx_model: {'params': params_temp, 'evals:': evals}}
        dump_to_existing_json(info_PATH, model_)

