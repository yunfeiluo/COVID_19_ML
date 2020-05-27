import numpy as np
from sklearn.linear_model import Ridge, Lasso
import pickle
import torch

if __name__ == '__main__':
    # read data
    data = None
    with open('data/train_data/data_for_train.pkl', 'rb') as f:
        data = pickle.load(f)
    
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
    regu_lam = 10 ** 0

    #model = Ridge(alpha=regu_lam)
    model = Lasso(alpha=regu_lam)
    model.fit(samples, labels)

    # test
    check = ['China_Beijing', 'China_Chongqing', 'China_Sichuan', 'China_Hainan', 'US_nan', 'Russia_nan', 'Japan_nan', 'Korea, South_nan']
    
    err = np.array([0.0, 0.0, 0.0])
    lookback = len(data[check[0]]["test_input"])
    test_len = len(data[check[0]]["test_labels"])
    for region in data:
        for i in range(test_len):
            test_input = torch.Tensor([data[region]["test_input"][i:]]).float()
            
            # if use linear model
            test_input = test_input.view(1, -1).detach().numpy()
            # print("test input shape", test_input.shape)

            out = model.predict(test_input)
            err += np.abs(data[region]["scaler"].inverse_transform([data[region]["test_labels"][i] - out[0]])[0])
            # print('out shape', out.shape)

            data[region]["test_input"].append(out.tolist()[0])
        #exit()
    err /= len(data)
    print('confirmed MAE', err[0])
    print('death MAE', err[1])
    print('recovered MAE', err[2])