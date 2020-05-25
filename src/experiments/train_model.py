import numpy as np
import torch
from torch import nn, autograd, optim
import pickle
import random
import matplotlib.pyplot as plt

from src.model.mlp import mlp_linear, multitask_mlp

if __name__ == '__main__':
    # read data
    data = None
    with open('data/train_data/samples_labels_and_covars.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # deterimine test reions
    regions = [key for key in data]
    random.shuffle(regions)
    test_len = 7
    train_len = len(data[regions[0]]["samples"]) - 7
    lookback = len(data[regions[0]]["samples"][0])

    # form train and test sampels and labels
    for region in regions:
        # train samples and labels
        data[region]["train_samples"] = data[region]["samples"][:train_len]
        data[region]["train_samples"] = torch.Tensor(data[region]["train_samples"]).float()
        data[region]["train_samples"] = autograd.Variable(data[region]["train_samples"]).float()
        # data[region]["train_samples"] = data[region]["train_samples"].view(train_len, -1) # if use linear model

        data[region]["train_labels"] = data[region]["labels"][:train_len]
        data[region]["train_labels"] = torch.Tensor(data[region]["train_labels"]).float()
        data[region]["train_labels"] = autograd.Variable(data[region]["train_labels"]).float()

        # # check
        # print(data[region]["train_samples"].shape)
        # print(data[region]["train_labels"].shape)
        # print(' ')

        # test input and labels
        data[region]["test_input"] = data[region]["samples"][train_len - 1]
        data[region]["test_labels"] = data[region]["labels"][train_len:]

    
    # training code
    #input_size = 3 * lookback # if use linear model
    input_size = 3
    # hyper-param
    hidden_size = 128
    output_size = 3
    step_size = 10 ** -3
    regu_lam = 10 ** -4
    epochs = 500
    regionalize = False

    # build model
    #model = mlp_linear(input_size, hidden_size, output_size)
    model = multitask_mlp(regions, input_size, hidden_size, output_size, regionalize=regionalize)
    loss_func = torch.nn.MSELoss()
    opt = optim.Adam(params=model.parameters(), lr=step_size, weight_decay=regu_lam)

    for region in regions:
        data[region]["last_out"] = None
        data[region]["min_loss"] = np.inf
    # training
    for epoch in range(epochs):
        acc = 0
        for region in regions:
            model.zero_grad()
            #out = model(data[region]["train_samples"])
            out = model(region, data[region]["train_samples"])
            # print('out shape', out.shape)
            # print('label shape', data[region]["train_labels"].shape)
            loss = loss_func(out, data[region]["train_labels"])
            loss.backward()
            opt.step()

            acc += loss.item()

            if loss.item() < data[region]["min_loss"]:
                data[region]["min_loss"] = loss.item()
                data[region]["last_out"] = out
        
        if epoch % 10 == 0:
            print('epoch {}, loss {}'.format(epoch, acc / len(regions)))
    
    # saving data and models for testing and ploting
    data_filename = 'data/train_data/train_and_test.pkl'
    with open(data_filename, 'wb') as f:
        print("write data to file", data_filename)
        pickle.dump(data, f)
    
    model_filename = 'data/models/simple_mlp.pkl'
    with open(model_filename, 'wb') as f:
        print("write model to file", model_filename)
        pickle.dump(model, f)
