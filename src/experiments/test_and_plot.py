import numpy as np
import torch
from torch import nn, autograd
import pickle
import matplotlib.pyplot as plt

def compute_MAE(y_pred, y_true):
    diff = np.abs(y_pred - y_true)
    return np.sum(diff) / len(diff)

if __name__ == '__main__':
    check = ['China_Beijing', 'China_Chongqing', 'China_Sichuan', 'China_Hainan', 'US_nan', 'Russia_nan', 'Japan_nan', 'Korea, South_nan']

    data = None
    data_filename = 'data/train_data/train_and_test.pkl'
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)
    
    model = None
    model_filename = 'data/models/simple_mlp.pkl'
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    # testing
    lookback = len(data[check[0]]["test_input"])
    test_len = len(data[check[0]]["test_labels"])
    for region in data:
        for i in range(len(data[region]["test_labels"])):
            test_input = torch.Tensor([data[region]["test_input"][i:]]).float()
            
            # if use linear model
            test_input = test_input.view(1, -1)
            # print("test input shape", test_input.shape)

            out = model(test_input)
            # print('out shape', out.shape)

            data[region]["test_input"].append(out.detach().numpy().tolist()[0])
        #exit()
    
    # plotting
    time = [i for i in range(len(data[check[0]]["labels"]))]
    for region in check:
        fig, axs = plt.subplots(1, 3)
        titles = ['confirmed', 'death', 'recovered']
        for j in range(3):
            true_signal = [i[j] for i in data[region]["labels"]]
            train_out = [i[j] for i in data[region]["last_out"].detach().numpy().tolist()]
            axs[j].plot(time[:len(true_signal)], true_signal, label='True_new_cases', c='b')
            axs[j].plot(time[:len(train_out)], train_out, label='train_pred', c='g')

            test_out = [i[j] for i in data[region]["test_input"][lookback:]]
            axs[j].plot(time[len(true_signal)-test_len:], test_out, label='test_pred', c='r')

            #axs[j].plot(time[len(time) - pred_len - len(y_pred[j]):len(time) - pred_len], y_pred[j], label='test_pred', c='r')
            #axs[j].plot(time[len(time) - pred_len:], future_pred[j], label='future_pred', c='y')
            axs[j].legend()
            axs[j].set_title('New '+ titles[j] + ' cases, in ' + region)
        for ax in axs.flat:
            ax.set(xlabel='Days->', ylabel='Number of cases')
        plt.show()