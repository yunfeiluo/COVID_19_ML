import numpy as np
import torch
# from torch import nn, autograd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def compute_MAE(y_pred, y_true):
    diff = np.abs(y_pred - y_true)
    return np.sum(diff) / len(diff)

if __name__ == '__main__':
    check = ['China_Beijing', 'China_Chongqing', 'China_Sichuan', 'China_Hainan', 'US_nan', 'Russia_nan', 'Japan_nan', 'Korea, South_nan']

    # read model and data
    data = None
    data_filename = 'data/train_data/train_out/mlp_train_and_test.pkl'
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)
    
    model = None
    model_filename = 'data/models/gm_net_325epoch_-3lr_64hidden.pkl'
    print('model path', model_filename)
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    # print(model)

    up_to_date_data = None
    with open('data/train_data/newly_generated_data.pkl', 'rb') as f:
        up_to_date_data = pickle.load(f)
    
    print("newly added days", len(up_to_date_data[check[0]]["labels"]) - len(data[check[0]]["labels"]))
    # testing and predictin
    future_pred_len = 30
    err = np.array([0.0, 0.0, 0.0])
    future_err = np.array([0.0, 0.0, 0.0])
    lookback = len(data[check[0]]["test_input"])
    test_len = len(data[check[0]]["test_labels"])
    train_len = len(data[check[0]]["train_labels"])
    for region in data:
        for i in range(test_len + future_pred_len):
            test_input = torch.Tensor([data[region]["test_input"][i:]]).float()
            
            # if use linear model
            # test_input = test_input.view(1, -1)
            # print("test input shape", test_input.shape)

            out = model(region, test_input).detach().numpy()
            if i < test_len:
                # err += np.abs(data[region]["scaler"].inverse_transform([data[region]["test_labels"][i] - out[0]])[0])
                err += (np.abs(data[region]["test_labels"][i] - out[0]) ** 2)
            else:
                try:
                    future_err += (np.abs(np.array(up_to_date_data[region]["labels"][train_len + i]) - out[0]) ** 2)
                except:
                    curr = None
            # print('out shape', out.shape)

            data[region]["test_input"].append(out.tolist()[0])
        #exit()
    err /= len(data)
    future_err /= len(data)
    print('test err: ')
    print('confirmed MAE', err[0])
    print('death MAE', err[1])
    print('recovered MAE', err[2])
    print(' ')

    print('future test err: ')
    print('confirmed MAE', future_err[0])
    print('death MAE', future_err[1])
    print('recovered MAE', future_err[2])
    print(' ')
    
    exit()
    # plotting
    time = [i for i in range(len(data[check[0]]["last_out"]) + test_len + future_pred_len)]
    for region in check:
        fig, axs = plt.subplots(1, 3)
        titles = ['confirmed', 'death', 'recovered']
        last_out = data[region]["scaler"].inverse_transform(data[region]["last_out"].detach().numpy()).tolist()
        true_signal = data[region]["scaler"].inverse_transform(up_to_date_data[region]["labels"]).tolist()
        test_out = data[region]["scaler"].inverse_transform(data[region]["test_input"][lookback:lookback+test_len]).tolist()
        future_pred = data[region]["scaler"].inverse_transform(data[region]["test_input"][lookback+test_len:]).tolist()
        for j in range(3):
            curr_true_signal = [i[j] for i in true_signal]
            train_out = [i[j] for i in last_out]
            axs[j].plot(time[:len(true_signal)], curr_true_signal, label='True_new_cases', c='b')
            axs[j].plot(time[:len(train_out)], train_out, label='train_pred', c='g')

            curr_test_out = [i[j] for i in test_out]
            axs[j].plot(time[len(train_out):len(train_out) + test_len], curr_test_out, label='test_pred', c='r')

            curr_future_pred = [i[j] for i in future_pred]
            axs[j].plot(time[len(train_out) + test_len:], curr_future_pred, label='future_pred', c='y')
            axs[j].legend()
            axs[j].set_title('New '+ titles[j] + ' cases, in ' + region)
        for ax in axs.flat:
            ax.set(xlabel='Days->', ylabel='Number of cases')
        plt.show()