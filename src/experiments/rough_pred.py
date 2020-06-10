import numpy as np
from sklearn.linear_model import Ridge, Lasso
import pickle
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # read data
    data = None
    with open('data/train_data/data_for_train.pkl', 'rb') as f:
        data = pickle.load(f)

    up_to_date_data = None
    with open('data/train_data/newly_generated_data.pkl', 'rb') as f:
        up_to_date_data = pickle.load(f)
    
    # deterimine test reions
    regions = [key for key in data]
    test_len = 7
    train_len = len(data[regions[0]]["samples"]) - 7
    lookback = len(data[regions[0]]["samples"][0])

    # form train and test sampels and labels
    samples = list()
    labels = list()
    for region in regions:
        # train samples and labels
        data[region]["train_samples"] = data[region]["samples"][:train_len]
        data[region]["train_samples"] = torch.Tensor(data[region]["train_samples"]).float()
        data[region]["train_samples"] = data[region]["train_samples"].view(train_len, -1).detach().numpy().tolist() # if use linear model
        samples += data[region]["train_samples"]

        data[region]["train_labels"] = data[region]["labels"][:train_len]
        labels += data[region]["train_labels"]
        #data[region]["train_labels"] = torch.Tensor(data[region]["train_labels"]).float()

        # # check
        # print(data[region]["train_samples"].shape)
        # print(data[region]["train_labels"].shape)
        # print(' ')

        # test input and labels
        data[region]["test_input"] = data[region]["samples"][train_len - 1]
        data[region]["test_labels"] = data[region]["labels"][train_len:]
    
    # train code
    # hyper params
    regu_lam = 10 ** -4

    #model = Ridge(alpha=regu_lam)
    model = Lasso(alpha=regu_lam)
    model.fit(samples, labels)

    # test
    check = ['China_Beijing', 'China_Chongqing', 'China_Sichuan', 'China_Hainan', 'US_nan', 'Russia_nan', 'Japan_nan', 'Korea, South_nan']
    
    print("newly added days", len(up_to_date_data[check[0]]["labels"]) - len(data[check[0]]["labels"]))

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
            test_input = test_input.view(1, -1).detach().numpy()
            # print("test input shape", test_input.shape)

            out = model.predict(test_input)
            # err += np.abs(data[region]["scaler"].inverse_transform([data[region]["test_labels"][i] - out[0]])[0])
            if i < test_len:
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
    print('confirmed MAE', err[0])
    print('death MAE', err[1])
    print('recovered MAE', err[2])
    print(' ')
    print('future test err: ')
    print('confirmed MAE', future_err[0])
    print('death MAE', future_err[1])
    print('recovered MAE', future_err[2])
    print(' ')

    #exit()
    # plotting
    time = [i for i in range(len(data[check[0]]["train_samples"]) + test_len + future_pred_len)]
    for region in check:
        fig, axs = plt.subplots(1, 3)
        titles = ['confirmed', 'death', 'recovered']
        last_out = data[region]["scaler"].inverse_transform(model.predict(data[region]["train_samples"])).tolist()
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