import numpy as np
from sklearn.svm import SVR
import pickle
import matplotlib.pyplot as plt

def svm_forcasting():
    data = None
    with open('src/feature_engineering/time_series.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # x = data['China_Beijing'][0]
    #x = data['Russia_nan'][2]
    time = [i for i in range(len(x))]

    look_back = 14
    test_len = 35
    train_x = x[:len(x)-test_len]
    test_x = x[len(x)-test_len:]

    samples = list()
    labels = list()

    for i in range(len(train_x) - 1 - look_back):
        samples.append(train_x[i:i+look_back])
        labels.append(train_x[i+look_back])
    
    model = SVR(kernel='rbf', gamma=10 ** -6, C=10 ** 3)
    model = model.fit(samples, labels)
    y_pred = list()
    prev_data = train_x[len(train_x) - look_back:]
    for i in range(len(test_x)):
        y_pred.append(model.predict([prev_data])[0])

        prev_data.append(y_pred[-1])
        prev_data = prev_data[1:]
    
    train_pred = model.predict(samples)

    plt.plot(time, x, c='b')
    plt.plot(time[len(time)-test_len:], y_pred, c='r')
    plt.plot(time[look_back+1:len(time)-test_len], train_pred, c='g')
    plt.show()

if __name__ == '__main__':
    svm_forcasting()