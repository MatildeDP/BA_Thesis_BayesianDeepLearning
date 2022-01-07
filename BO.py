from deterministic import Deterministic_net
import torch
from sklearn.model_selection import train_test_split
from data import DataLoaderInput, LoadDataSet, settings
import torch.nn as nn
from utils import dump_to_json, dump_to_existing_json, plot_decision_boundary, plot_acc_and_loss
from os.path import exists
import GPyOpt
import numpy as np



def Loss(X_train, y_train, X_test, y_test, which_data, params, model, optimizer, test_each_epoch = False):
    torch.manual_seed(0)

    criterion = nn.CrossEntropyLoss()

    # Standardise
    if which_data == 'mnist' or which_data == 'emnist' or which_data == 'fashion':
        X_test = (X_test - torch.mean(X_train)) / torch.std(X_train)
        X_train = (X_train - torch.mean(X_train)) / torch.std(X_train)



    # Torch dataloader instance
    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_train, y_train, which_data = which_data),
                                               batch_size=params['batch_size'],
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test, which_data = which_data),
                                              batch_size=1000,
                                              shuffle=False)

    # Epoch loop
    if test_each_epoch:
        # Initialise data container
        evals = {'acc': [], 'train_loss': [], 'test_loss': [], 'test_loss_l2':[],'train_loss_l2':[] }
        for epoch in range(n_epochs):
            if epoch % 50 == 0:
                print("Epoch {}".format(epoch))

            # One epoch of training
            optimizer, ave_train_loss,  ave_train_loss_l2 = model.train_net(train_loader, optimizer, criterion,
                                                        save_net=False, l2 = params['l2'])


            # Collect train loss for each epoch
            evals['train_loss'].append(ave_train_loss)


            # Test on trained model

            accuracy, ave_test_loss, _, ave_test_loss_l2= model.test_net(test_loader=test_loader, criterion=criterion,
                                                        freq=30, temp = 1, l2 = params['l2'])
            evals['test_loss'].append(ave_test_loss.item())
            evals['acc'].append(accuracy.item())
            evals['test_loss_l2'].append(ave_test_loss_l2)
            evals['train_loss_l2'].append(ave_train_loss_l2)

        print('.......Train loss: {}.      Test loss: {}       Accuracy: {}.'.format(evals['train_loss'][-1],
                                                                                     evals['test_loss'][-1],
                                                                                     evals['acc'][-1]))

        # plots loss/acc
        loss_path = 'Deterministic/'+which_data+'/bo_with_dropout/plots/loss_acc_' + counter.__getitem__('string') + '.jpg'
        plot_acc_and_loss(testloss=evals['test_loss'], trainloss=evals['train_loss'], accuracy=evals['acc'],
                          save_path=loss_path)

        # save model in each epoch (makes it possible to recreate evaluations)
        net_path = 'Deterministic/' + which_data.lower() + '/bo_with_dropout/models/m_' + counter.__getitem__('string')# + '_e' + str(
            #epoch)
        opti_path = 'Deterministic/' + which_data.lower() + '/bo_with_dropout/opti/m_' + counter.__getitem__('string')# + '_e' + str(
            #epoch)
        torch.save(model.state_dict(), net_path)
        torch.save(optimizer.state_dict(), opti_path)


    else:
        # Initialise data container
        evals = {'acc': None, 'train_loss': None, 'test_loss': None, 'train_loss_l2':None, 'test_loss_l2':None}

        for epoch in range(n_epochs):
            if epoch % 50 == 0:
                print("Epoch {}".format(epoch))

            # One epoch of training
            optimizer, ave_train_loss, ave_train_loss_l2 = model.train_net(train_loader, optimizer, criterion,
                                                        save_net=False, l2 =params['l2'])

            # Collect train loss for each epoch
            evals['train_loss'].append(ave_train_loss)
            evals['train_loss_l2'] = (ave_train_loss_l2)

        # save model in each epoch (makes it possible to recreate evaluations)
        net_path = 'Deterministic/' + which_data.lower() + '/bo_with_dropout/models/m_' + counter.__getitem__(
            'string') #+ '_e' + str(epoch)
        opti_path = 'Deterministic/' + which_data.lower() + '/bo_with_dropout/opti/m_' + counter.__getitem__(
            'string') #+ '_e' + str(epoch)
        torch.save(model.state_dict(), net_path)
        torch.save(optimizer.state_dict(), opti_path)


        # Test on trained model
        accuracy, ave_test_loss, _, ave_test_loss_l2 = model.test_net(test_loader=test_loader, criterion=criterion,
                                                    freq=30, l2=params['l2'])
        evals['test_loss'] = ave_test_loss.item()
        evals['acc'] = accuracy.item()
        evals['test_loss_l2'] =(ave_test_loss_l2)



        print('.......Train loss: {}.      Test loss: {}       Accuracy: {}.'.format(evals['train_loss'][-1],
                                                                                     evals['test_loss'],
                                                                                     evals['acc']))

    print('-' * 100)
    print('-' * 100)

    # dump parameters, test-/train loss, accuracy
    params['batch_size'] = str(params['batch_size'])  # stringify
    params['hidden_dim'] = str(params['hidden_dim'])  # stringify
    model_ = {counter.__getitem__('int'): {'params': params, 'evals': evals}}

    if exists('Deterministic/'+which_data+'/bo_with_dropout/params_evals/model_info_BO.json'):  # dump to existing
        dump_to_existing_json('Deterministic/'+which_data+'/bo_with_dropout/params_evals/model_info_BO.json', model_)

    else:  # create json files if it is the first parameters check
       dump_to_json('Deterministic/'+which_data+'/bo_with_dropout/params_evals/model_info_BO.json', model_)


    if which_data == 'two_moons':

        decision_path = 'Deterministic/two_moons/bo_with_dropout/plots/decision_boundary' + counter.__getitem__('string') +'.jpg'
        plot_decision_boundary(model, dataloader=test_loader, S=20,
                               title="Decision boundary with test points on trained model",
                               save_image_path=decision_path, temp = 1)


    if test_each_epoch:
        return evals['test_loss'][-1]
    else:
        return evals['test_loss']

class iter:
    def __init__(self):
        self.n = 0
    def increase(self):
        self.n+=1

        return self.n
    def __getitem__(self, type):
        if type == 'string':
            return str(self.n)
        elif type == 'int':
            return self.n



def objective_function(x):

    # increase iterator count
    param = x[0]

    # get correct input/output dim dependent on *which_data*
    input_dim, output_dim = settings(which_data)

    print('Parameters: \n Hidden dimension: {} \n l2: {} '
          '\n learning rate: {} \n momentum: {} \n batch size: {} \n  drop out: {}'.format(int(param[4]),
                                                                              param[1], param[0],
                                                                              param[2],
                                                                              int(param[3]), param[5]))

    param_dict = {'batch_size': int(param[3]), 'hidden_dim':int(param[4]),
              'lr': param[0], 'l2': param[1],
              'momentum': param[2]}

    model = Deterministic_net(input_dim=input_dim, hidden_dim=param_dict['hidden_dim'], output_dim=output_dim, dropout_rate = param[5])

    optimizer = torch.optim.SGD(model.parameters(), lr=param_dict['lr'],
                                weight_decay=param_dict['l2'],
                                momentum=param_dict['momentum'])


    # load train data
    loss = Loss(X_train, y_train,X_val, y_val, which_data, param_dict, model, optimizer, test_each_epoch = True)

    # increase counter
    counter.increase()
    return loss


if __name__ == '__main__':

    global counter
    counter = iter()

    global which_data
    which_data = 'fashion'

    global n_epochs
    n_epochs = 300


    Data = LoadDataSet(which_data)
    X, y, X_val, y_val = Data.load_data_for_CV(n_samples = 2000, noise = 0.3)
    if which_data != 'two_moons':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    else:
        X_train, y_train, X_test, y_test = X, y, X_val, y_val

    if which_data == 'two_moons' or which_data == 'cancer':
        batch_size = (16, 32, 64)
        hidden_dim = range(50, 200)
        max_ite = 100 
    elif which_data == 'mnist' or which_data == 'emnist' or which_data == 'fashion':
        batch_size = (128, 256)
        hidden_dim = range(150, 300)
        max_ite = 30


    lr_range = tuple([i.item() for i in torch.logspace(-4, -2, 30)])
    l2_range = tuple([i.item() for i in torch.logspace(-4, -2, 30)])
    momentum_range = tuple([i.item() for i in torch.logspace(-3, -0.1, 30)])


    domain = [{'name': 'lr', 'type': 'discrete', 'domain': lr_range}, #torch.logspace(-4, -1, 200)
              {'name': 'l2', 'type': 'discrete', 'domain': l2_range}, #torch.logspace(-3, -0.1, 200)
              {'name': 'momentum', 'type': 'discrete', 'domain': momentum_range}, #torch.logspace(-2, -0.1, 200)
              {'name': 'batch_size', 'type': 'discrete', 'domain': batch_size},
              {'name': 'hidden_dim', 'type': 'discrete', 'domain': hidden_dim},
              {'name': 'dropout', 'type': 'discrete', 'domain': (0.05,0.1,0.2,0.3,0.4,0.5)}]


    opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                              domain=domain,  # box-constrains of the problem
                                              acquisition_type='EI',  # Select acquisition function MPI, EI, LCB
                                              )



    opt.run_optimization(max_iter=max_ite, report_file = 'Deterministic/'+which_data+'/bo_with_dropout/txt/report_file.txt', evaluations_file = 'Deterministic/'+which_data+'/bo_with_dropout/txt/eval_file.txt', models_file='Deterministic/'+which_data+'/bo_with_dropout/txt/model_file.txt')

    from importlib import reload
    import matplotlib.pyplot as plt
    reload(plt)
    opt.plot_convergence("Deterministic/" + which_data + "/bo_with_dropout/bo_convergence_test.png")



    x_best = opt.X[np.argmin(opt.Y)]
    print('Optimal parameters {}:'.format(x_best))
    print('Optimal test nll {}'.format(opt.Y))

    # GP parameter name and values. This is also what we save in model_file
    print(opt.model.get_model_parameters_names())
    print(opt.model.get_model_parameters())


