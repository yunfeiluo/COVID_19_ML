import matplotlib.pyplot as plt
import pickle
import numpy as np

def compute_MAE(y_pred, y_true):
    diff = np.abs(np.array(y_pred) - np.array(y_true))
    return np.sum(diff) / len(diff)

def check_pred():
    # read the previous results
    y_preds, y_trues, future_preds, x_preds, x_trues, pred_len = None, None, None, None, None, None
    with open('src/experiments/pred_check.pkl', 'rb') as f:
        y_preds, y_trues, future_preds, x_preds, x_trues, pred_len = pickle.load(f)

    # specify regions need to check
    check = ['China_Beijing', 'China_Chongqing', 'China_Sichuan', 'China_Hainan', 'US_nan', 'Russia_nan', 'Japan_nan', 'Korea, South_nan']

    # read new data
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
    
    look_back = 14
    prev_len = len(x_preds[0][0]) + len(y_trues[0][0]) + look_back
    future_trues = [[list() for i in range(3)] for j in range(len(future_preds))]
    regions = list()
    for i in range(3):
        j = 0
        for region in data:
            x = data[region][i]
            if i == 0:
                regions.append(region)
            
            future_trues[j][i] = x[prev_len:]
            j += 1
            
    confirmed_err, death_err, recovered_err = list(),list(),list()
    days = len(future_trues[0][0])
    # visualize
    for i in range(len(y_preds)):
        y_pred, y_true, future_pred, x_pred, x_true = y_preds[i], y_trues[i], future_preds[i], x_preds[i], x_trues[i]
        future_true = future_trues[i]

        # compute error
        confirmed_err.append(compute_MAE(future_pred[0][:len(future_true[0])], future_true[0]))
        death_err.append(compute_MAE(future_pred[1][:len(future_true[1])], future_true[1]))
        recovered_err.append(compute_MAE(future_pred[2][:len(future_true[2])], future_true[2]))

        # extract required series
        if regions[i] in check:
            # plot
            time = [i for i in range(len(x_pred[0]) + len(y_true[0]) + pred_len)]
            fig, axs = plt.subplots(1, 3)
            titles = ['confirmed', 'death', 'recovered']
            for j in range(3):
                true_signal = x_true[j] + y_true[j] + future_true[j]
                axs[j].plot(time[:len(true_signal)], true_signal, label='True_new_cases', c='b')

                axs[j].plot(time[len(time) - pred_len - len(y_pred[j]):len(time) - pred_len], y_pred[j], label='test_pred', c='r')
                axs[j].plot(time[:len(time) - pred_len - len(y_true[j])], x_pred[j], label='train_pred', c='g')
                axs[j].plot(time[len(time) - pred_len:], future_pred[j], label='future_pred', c='y')
                axs[j].legend()
                axs[j].set_title('New '+ titles[j] + ' cases, in ' + regions[i])
            for ax in axs.flat:
                ax.set(xlabel='Days->', ylabel='Number of cases')
            plt.show()

    # print error
    print('confirmed err', sum(confirmed_err) / len(confirmed_err))
    print('max confirmed err', max(confirmed_err))
    print('max region', regions[np.argmax(np.array(confirmed_err))])
    print('min confirmed err', min(confirmed_err))
    print(' ')

    print('death err', sum(death_err) / len(death_err))
    print('max death err', max(death_err))
    print('max region', regions[np.argmax(np.array(death_err))])
    print('min death err', min(death_err))
    print(' ')

    print('recovered err', sum(recovered_err) / len(recovered_err))
    print('max recovered err', max(recovered_err))
    print('max region', regions[np.argmax(np.array(recovered_err))])
    print('min recovered err', min(recovered_err))

if __name__ == '__main__':
    check_pred()