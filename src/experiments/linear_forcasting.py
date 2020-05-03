import numpy as np
import torch
from torch import nn, autograd, optim
import pickle
import matplotlib.pyplot as plt

from src.model.mlp import mlp_linear
import src.model.autoencoder as autoencoder

def compute_MAE(y_pred, y_true):
    diff = np.abs(np.array(y_pred) - np.array(y_true))
    return np.sum(diff) / len(diff)

def model_eval(model, feature_generators, test_samples, true_labels, check, regions, last_out, labels, train_len, covars_train, covars_test):
    '''
    @param model: prediction model
    @param feature_generators
    @param test_samples: test_x
    @param true_labels: test_y
    @param check: list need to visualize
    @param covariates: for feature concatenation
    @param regions: list of region
    @param last_out, labels, train_len: for visualize training result
    '''
    confirmed_err = 0
    death_err = 0
    recovered_err = 0

    y_trues = list()
    y_preds = list()
    future_preds = list()
    x_trues = list()
    x_preds = list()

    for i in range(len(test_samples['confirmed'])):
        y_true = true_labels[i]
        y_pred = [list() for i in range(3)]
        future_pred = [list() for i in range(3)]
        pred_len = 7

        for j in range(len(test_labels['confirmed'][i]) + pred_len):
            # generate features by feature generators
            confirmed_input = torch.Tensor(test_samples['confirmed'][i]).float()
            confirmed = [i for i in feature_generators['confirmed'](confirmed_input).detach().numpy()]

            death_input = torch.Tensor(test_samples['death'][i]).float()
            death = [i for i in feature_generators['death'](death_input).detach().numpy()]

            recovered_input = torch.Tensor(test_samples['recovered'][i]).float()
            recovered = [i for i in feature_generators['recovered'](recovered_input).detach().numpy()]

            input_ = confirmed + death + recovered + covars_test[i] # concatenate covariates
            input_ = torch.Tensor(input_).float()

            # make prediction
            pred = model(input_).detach().numpy()
            if j >= len(test_labels['confirmed'][i]):
                for k in range(3):
                    future_pred[k].append(pred[k])
            else:
                for k in range(3):
                    y_pred[k].append(pred[k])

            test_samples['confirmed'][i].append(pred[0])
            test_samples['confirmed'][i] = test_samples['confirmed'][i][1:]

            test_samples['death'][i].append(pred[1])
            test_samples['death'][i] = test_samples['death'][i][1:]

            test_samples['recovered'][i].append(pred[2])
            test_samples['recovered'][i] = test_samples['recovered'][i][1:]
        
        # compute error
        confirmed_err += compute_MAE(y_pred[0], y_true[0])
        death_err += compute_MAE(y_pred[1], y_true[1])
        recovered_err += compute_MAE(y_pred[2], y_true[2])
        
        # store result
        y_trues.append(y_true)
        y_preds.append(y_pred)
        future_preds.append(future_pred)

        # visualize
        if regions[i] in check:
            train_res = last_out[i*train_len:i*train_len + train_len]
            train_label = labels[i*train_len:i*train_len + train_len]
            x_pred = list()
            x_true = list()

            for k in range(3):
                x_pred.append([i[k] for i in train_res])
                x_true.append([i[k] for i in train_label])
            
            # store result
            x_preds.append(x_pred)
            x_trues.append(x_true)

            # # plot
            # time = [i for i in range(len(x_pred[0]) + len(y_true[0]) + pred_len)]
            # fig, axs = plt.subplots(1, 3)
            # titles = ['confirmed', 'death', 'recovered']
            # for j in range(3):
            #     axs[j].plot(time[len(time) - pred_len - len(y_true[j]):len(time) - pred_len], y_true[j], label='True_new_cases', c='b')
            #     axs[j].plot(time[:len(time) - pred_len - len(y_true[j])], x_true[j], c='b')
            #     axs[j].plot(time[len(time) - pred_len - len(y_pred[j]):len(time) - pred_len], y_pred[j], label='test_pred', c='r')
            #     axs[j].plot(time[:len(time) - pred_len - len(y_true[j])], x_pred[j], label='train_pred', c='g')
            #     axs[j].plot(time[len(time) - pred_len:], future_pred[j], label='future_pred', c='y')
            #     axs[j].legend()
            #     axs[j].set_title('New '+ titles[j] + ' cases, in ' + regions[i])
            # for ax in axs.flat:
            #     ax.set(xlabel='Days->', ylabel='Number of cases')
            # plt.show()
    
    with open('src/experiments/pred_check.pkl', 'wb') as f:
        pickle.dump((y_preds, y_trues, future_preds, x_preds, x_trues, pred_len), f)
    
    return confirmed_err / len(test_samples['confirmed']), death_err / len(test_samples['confirmed']), recovered_err / len(test_samples['confirmed'])
        

def train_feature_generator(train_samples, train_labels, ts_name):
    print('######### Training feature generators... #################')
    feature_generators = dict()
    for ts in ts_name:
        print('generator for', ts)
        feature_generator = autoencoder.train_dense_mlp_autoencoder(train_samples[ts])
        feature_generators[ts] = feature_generator

    with open('src/model/feature_generators.pkl', 'wb') as f:
        pickle.dump(feature_generators, f)
    
    print('###### feature generator training complete ###########')
    exit()
    #return feature_generator

if __name__ == '__main__':    
    ## read and split data #######################################################################################
    data = None
    with open('src/feature_engineering/time_series.pkl', 'rb') as f:
        data = pickle.load(f)

    # wash data
    covariates = None
    with open('src/feature_engineering/features.pkl', 'rb') as f:
        covariates = pickle.load(f)
    countries = [i for i in covariates]
    droped = list()
    for region in data:
        country = region.split('_')[0]
        if country not in countries:
            droped.append(region)
    for region in droped:
        del data[region]

    # split data    
    look_back = 14
    test_len = 7
    train_len = -1
    
    train_samples = {'confirmed':list(), 'death':list(), 'recovered':list()}
    train_labels = {'confirmed':list(), 'death':list(), 'recovered':list()}
    test_samples = {'confirmed':list(), 'death':list(), 'recovered':list()}
    test_labels = {'confirmed':list(), 'death':list(), 'recovered':list()}
    ts_name = ['confirmed', 'death', 'recovered']

    regions = list()

    covars_train = list()
    covars_test = list()

    for i in range(3):
        for region in data:
            # if 'China' not in region:
            #     continue
            x = data[region][i]
            if i == 0:
                regions.append(region)

            # split data
            train_x = x[:len(x)-test_len]
            test_x = test_x = x[len(x)-test_len:]
            
            # append train data
            train_len = len(train_x) - 1 - look_back
            for j in range(len(train_x) - 1 - look_back):
                train_samples[ts_name[i]].append(train_x[j:j+look_back])
                train_labels[ts_name[i]].append(train_x[j+look_back])
                if i == 0:
                    covars_train.append(covariates[region.split('_')[0]])
            
            # append test data
            test_samples[ts_name[i]].append(train_x[len(train_x) - look_back:])
            test_labels[ts_name[i]].append([i for i in test_x])
            if i == 0:
                covars_test.append(covariates[region.split('_')[0]])
    
    #####################################################################################################################################

    ## train feature generator ##########################################################################################################
    #train_feature_generator(train_samples, train_labels, ts_name)
    feature_generators = None
    with open('src/model/feature_generators.pkl', 'rb') as f:
        feature_generators = pickle.load(f)

    #####################################################################################################################################

    ## concatenate covariates ###########################################################################################################
    print('######## concatenate features... ##############')

    samples = [list() for i in range(len(train_samples['confirmed']))]
    labels = [list() for i in range(len(train_samples['confirmed']))]

    # extract features by feature generators
    for ts in train_samples:
        for i in range(len(train_samples[ts])):
            input_seq = torch.Tensor(train_samples[ts][i]).float()
            new_feature = [i for i in feature_generators[ts](input_seq).detach().numpy()]
            samples[i] += new_feature
            labels[i].append(train_labels[ts][i])
    for i in range(len(samples)):
        samples[i] += covars_train[i]
    
    # for test
    true_labels = [list() for i in range(len(test_labels['confirmed']))] # [[confirmed, death, recovered]]

    # extract features by feature generators
    for ts in test_labels:
        for i in range(len(test_labels[ts])):
            true_labels[i].append(test_labels[ts][i])
    #####################################################################################################################################

    ## Train linear models ##############################################################################################################
    print('### Training Linear Models... #############')
    samples = torch.Tensor(samples).float()
    samples = autograd.Variable(samples).float()
    labels = torch.Tensor(labels).float()

    input_size = len(samples[0])
    # hyper-param
    hidden_size = 64
    output_size = 3
    step_size = 10 ** -3
    regu_lam = 10 ** -4
    epochs = 700

    # build model
    model = mlp_linear(input_size, hidden_size, output_size)
    loss_func = torch.nn.MSELoss()
    opt = optim.Adam(params=model.parameters(), lr=step_size, weight_decay=regu_lam)

    last_out = None
    min_loss = np.inf
    confirmed_errs, death_errs, recovered_errs = list(), list(), list()
    # training
    for epoch in range(epochs):
        model.zero_grad()
        out = model(samples)
        loss = loss_func(out, labels)
        loss.backward()
        opt.step()

        if loss.item() < min_loss:
            last_out = out.detach().numpy()
        
        if epoch % 10 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
        
        # check error on test each epoch
        # confirmed_err, death_err, recovered_err = model_eval(model, feature_generators, test_samples, true_labels, [], regions, last_out, labels, train_len, covars_train, covars_test)
        # confirmed_errs.append(confirmed_err)
        # death_errs.append(death_err)
        # recovered_errs.append(recovered_err)
    
    # plt.plot([i for i in range(epochs)], confirmed_errs)
    # plt.show()
    # plt.plot([i for i in range(epochs)], death_errs)
    # plt.show()
    # plt.plot([i for i in range(epochs)], recovered_errs)
    # plt.show()

    last_out = [i for i in last_out]
    labels = [i for i in labels.detach().numpy()]
    #####################################################################################################################################

    #### model eval #####################################################################################################################
    check = regions
    #check = ['China_Beijing', 'China_Chongqing', 'China_Sichuan', 'China_Hainan', 'US_nan', 'Russia_nan', 'Japan_nan', 'Korea, South_nan']
    #check = []    
    confirmed_err, death_err, recovered_err = model_eval(model, feature_generators, test_samples, true_labels, check, regions, last_out, labels, train_len, covars_train, covars_test)

    print('confirmed err', confirmed_err)
    print('death err', death_err)
    print('recovered err', recovered_err)
    #####################################################################################################################################
