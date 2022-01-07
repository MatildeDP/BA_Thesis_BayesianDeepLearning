import torch
import os
from copy import deepcopy
import json
from data import DataLoaderInput

def l2_penalizer(model):
    s2 = 0
    for val in model.state_dict().values():
        if len(val.shape) > 1:
            s2 += sum(sum(val ** 2))
        else:
            s2 += sum(val ** 2)

    return s2

def dump_to_json(PATH, dict):
    with open(PATH, 'w') as fp:
        temp_ = {'key': [dict]}
        json.dump(temp_, fp)

def monte_carlo_bma(model, Xtest, ytest, S, C, temp, criterion,l2,dropout, forplot=False, save_models = '', path_to_bma = '', save_probs = '', batch = False,  which_data = None):

    """
    Monte carlo approximation of BMA.

    Notice, that the model class MUST contain following functions:
       1. sample_from_posterior
       2. replace_network_weights
       3. predict
       4. nn Criterion

    :param test_loader: test data
    :param S: number of models to sum over
    :param C: number of classes
    :param model: instance of KFAC or SWAG class
    :param Xtest: test data
    :param ytest: test labels
    :param temp: temperature scale parameter
    :param criterion: nn Cross entropy loss
    :param save_models: Path to models
    :param forplot: If True, method does not compute accuracy and only returns p_yx
    :param load_models: If not empty string, method loads models from path to compute BMA. If empty string, new models will be sampled
    :param param_idx: parameter count. Used to load correct models
    :param: return_probs: if true: returns p_yx and P_yxw


    :return: p_yx: torch.tensor (nxc), contains p(y = c|x) for all classes and for all x in test_loader
    :return: p_yxw: dict. one key-value pair per model. Value = torch tensor (nxc), with (y=c|x,w), c in classes, x in test_loader
    :return: loss for each model
    :return: acc: accuracy of BMA prediction
    """

    # Deepcopy model
    model_ = deepcopy(model)

    n = len(Xtest)  # number of test points
    p_yx = torch.zeros(n, C)
    p_yxw = {i: [] for i in range(S)}
    accuracy, all_loss, all_loss_l2 = [], [],[]
    ave_score, ave_loss= 0, 0

    # compute bma with saved models
    if path_to_bma:
        ytest = torch.tensor(ytest)
        # iterate through path
        for i, filename in enumerate(os.listdir(path_to_bma)):
            if '.DS' not in filename:
                model_.load_state_dict(torch.load(path_to_bma + filename))

                with torch.no_grad():
                    # Monte Carlo
                    p_yxw[i], score, _ = model_.predict(Xtest, temp=temp)
                    p_yx += 1 / S * p_yxw[i]
                    ave_score +=1/S *score


    # sample weights to compute bma
    else:

        for i in range(S):

            with torch.no_grad():

                # Sample weights from posterior
                sampled_weights = model_.sample_from_posterior()

                if sampled_weights == None:
                    return None, None, None, None, None

                # Replace network weights with sampled network weights
                test_model = model_.replace_network_weights(sampled_weights, dropout)

                # dump sampled_weights to json at path save_models
                if save_models and test_model is None:
                    torch.save(model_.state_dict(), save_models + 'bma_model_' + str(i))
                elif save_models and test_model is not None:
                    torch.save(test_model.state_dict(), save_models + 'bma_model_' + str(i))

                # Monte Carlo
                # if statement nessesary because SWAG return new model, KFAC does not
                if test_model is None:
                    p_yxw[i], score, _ = model_.predict(Xtest, temp = temp)
                    ave_score += 1 / S * score


                else:
                    p_yxw[i], score, _ = test_model.predict(Xtest, temp = temp)
                    ave_score += 1 / S * score

                    if not forplot:
                        loss = criterion(score, ytest)
                        ave_loss  +=1/S *loss.item()

                p_yx += 1 / S * p_yxw[i]


    if save_probs:

        p_yxw_ = {key: val.numpy().tolist() for key, val in p_yxw.items()}
        dict_ = {'p_yxw':p_yxw_, 'p_yx': p_yx.numpy().tolist()}
        dump_to_json(save_probs, dict_)

    if forplot:
        return p_yx

    else:
        with torch.no_grad():
        # compute overall accuracy
            yhat = torch.max(p_yx, 1).indices
            acc = (yhat == ytest).sum() / len(ytest)

            # compute loss of prediction
            if batch:
                data = DataLoaderInput(ave_score, ytest, which_data=which_data)
                dataloader =  torch.utils.data.DataLoader(dataset=data,
                                          batch_size=1,
                                          shuffle=False)

                batch_loss, batch_loss_l2 =[], []
                for score, y in dataloader:
                    loss_ = criterion(score, y)
                    loss_l2_ = loss_+ l2_penalizer(model_) * l2
                    batch_loss.append(loss_)
                    batch_loss_l2.append(loss_l2_)

                return p_yxw, p_yx, batch_loss, acc, batch_loss_l2

            else:
                loss = criterion(ave_score, ytest)
                loss_l2 = loss.item() + l2_penalizer(model_) * l2





        return p_yxw, p_yx, ave_loss, acc, loss_l2.item()

def monte_carlo_bma_out(model, Xtest, S, C, temp, path_to_bma):

    """
    Use this version if only p_yx and p_yxw is required
    :param model:
    :param Xtest:
    :param S:
    :param C:
    :param temp:
    :param path_to_bma:
    :return:
    """



    # Deepcopy model, just in case
    model_ = deepcopy(model)

    n = len(Xtest)  # number of test points
    p_yx = torch.zeros(n, C)
    p_yxw = {i: [] for i in range(S)}

    # compute bma with saved models
    if path_to_bma:
        # iterate through path
        for i, filename in enumerate(os.listdir(path_to_bma)):
            if '.DS' not in filename:
                model_.load_state_dict(torch.load(path_to_bma + filename))

                with torch.no_grad():
                    # Monte Carlo
                    p_yxw[i], score, _ = model_.predict(Xtest, temp=temp)
                    p_yx += 1 / S * p_yxw[i]

        return p_yx, p_yxw

