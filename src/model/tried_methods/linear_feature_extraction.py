import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn, autograd, optim

from src.model.mlp import mlp_linear

def mean_square_error(y_pred, y_true):
    # mean absolute error
    # ||y_pred-y_true||^2
    diff = y_pred - y_true
    return np.sum(np.abs(diff)) / len(diff)
    #return np.dot(diff, diff) / len(diff)

def train_mlp_linear(train_samples, train_labels):
    train_samples = torch.Tensor(train_samples).float()
    train_samples = autograd.Variable(train_samples).float()

    train_labels = torch.Tensor(train_labels).float()
    train_labels = train_labels.view(-1, 1)

    input_size = len(train_samples[0])
    # hyper-param
    hidden_size = 128
    output_size = 1
    step_size = 10 ** -3
    regu_lam = 10 ** -4
    epochs = 100

    # build model
    model = mlp_linear(input_size, hidden_size, output_size)
    loss_func = torch.nn.MSELoss()
    opt = optim.Adam(params=model.parameters(), lr=step_size, weight_decay=regu_lam)

    # training
    for epoch in range(epochs):
        model.zero_grad()
        out = model(train_samples)
        loss = loss_func(out, train_labels)
        loss.backward()
        opt.step()
        
        if epoch % 10 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
    return model

def model_eval(model, test_samples, test_labels, test_len, compute_err, train_samples, train_labels, train_len, regions, check):
    '''
    @param test_samples: list of last look_back window # 2d array
    @param test_labels: list of true value of the test_len predicted days # 2d array
    return MSE
    '''
    res = list()
    for i in range(len(test_samples)):
        y_pred = list()
        prev_data = test_samples[i]
        y_true = test_labels[i]
        for j in range(len(y_true)):
            # for pytorch mlp
            input_data = torch.Tensor(prev_data).float()
            y_pred.append(model(input_data).detach().numpy()[0])

            #y_pred.append(model.predict([prev_data])[0])

            prev_data.append(y_pred[-1])
            prev_data = prev_data[1:]

        err = compute_err(np.array(y_pred), np.array(y_true))
        res.append(err)

        # visualize
        if regions[i] in check:
            #train_pred = model.predict(train_samples[i*train_len:i*train_len+train_len])
            train_pred = model(torch.Tensor(train_samples[i*train_len:i*train_len+train_len]).float()).detach().numpy()
            entire_data = train_labels[i*train_len:i*train_len+train_len]+ y_true
            #entire_data = true_signal[i]
            time = [i for i in range(len(entire_data))]
            plt.plot(time, entire_data, c='b', label='true_signal')
            plt.plot(time[len(time) - test_len:], y_pred, c='r', label='test_pred')
            plt.plot(time[:len(time)-test_len], train_pred, c='g', label='train_pred')
            plt.legend()
            plt.xlabel('days')
            plt.ylabel('New number of confirmed cases')
            plt.title(regions[i])
            plt.show()

    return sum(res) / len(res)

if __name__ == '__main__':
    data = None
    with open('src/feature_engineering/time_series.pkl', 'rb') as f:
        data = pickle.load(f)
    
    look_back = 14
    test_len = 7
    train_len = -1
    
    train_samples = list()
    train_labels = list()
    test_samples = list()
    test_labels = list()

    regions = list()
    check = ['China_Beijing', 'China_Chongqing', 'China_Sichuan', 'China_Hainan', 'US_nan', 'Russia_nan', 'Japan_nan', 'Korea, South_nan']
    #check = []

    # confirmed cases only, i.e. ind 0
    for region in data:
        # if 'China' not in region:
        #     continue
        x = data[region][0]
        regions.append(region)

        # split data
        train_x = x[:len(x)-test_len]
        test_x = test_x = x[len(x)-test_len:]
        
        # append train data
        train_len = len(train_x) - 1 - look_back
        for i in range(len(train_x) - 1 - look_back):
            train_samples.append(train_x[i:i+look_back])
            train_labels.append(train_x[i+look_back])
        
        # append test data
        test_samples.append(train_x[len(train_x) - look_back:])
        test_labels.append([i for i in test_x])
    
    # build model
    print("Train model...")
    # lam = 10 ** 1
    # model = Ridge(alpha=lam).fit(train_samples, train_labels)

    # train mlp
    model = train_mlp_linear(train_samples, train_labels)

    err = model_eval(model, test_samples, test_labels, test_len, mean_square_error, train_samples, train_labels, train_len, regions, check)
    print('err', err)
