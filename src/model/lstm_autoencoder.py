import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing

class lstm_autoencoder(nn.Module):
    # TODO
    def __init__(self, input_size, bottleneck_size, encode_only=False):
        super(lstm_autoencoder, self).__init__()
        self.encode_only = encode_only
        self.input_size = input_size # batch size
        self.num_layers = 1
        self.bottleneck_size, self.hidden_dim = bottleneck_size, 2 * bottleneck_size

        self.rnn1 = nn.LSTM(input_size=self.input_size, hidden_size=self.bottleneck_size, num_layers=self.num_layers, batch_first=True)
        #self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=self.num_layers, batch_first=True)

    def forward(self, input_seq):
        #x = x.reshape((1, 14, self.input_size))
        encoded_seq, hidden = self.rnn1(input_seq)
        print('x shape after rnn1', encoded_seq.shape)
        #x, (hidden_n, _) = self.rnn2(x)
        #print('x shape after rnn2', x.shape)
        #return hidden_n.reshape((self.input_size, self.embedding_dim))
        return encoded_seq

def train_lstm_autoencoder(train_samples):
    random.shuffle(train_samples)
    #test_len = int(len(train_samples) * 0.2)
    test_len = 0
    print('num of test', test_len)

    new_train_samples = train_samples[:5]

    # reshape test_samples to be (batch size, sequence length, input dimension (num features))
    reshaped_samples = list()
    for sample in new_train_samples:
        re_sample = list()
        for step in sample:
            re_sample.append([step, step, step])
        reshaped_samples.append(re_sample)

    samples = reshaped_samples[:len(reshaped_samples)-test_len]
    #samples = preprocessing.normalize(samples)
    samples = torch.Tensor(samples).float()
    samples = autograd.Variable(samples).float()

    test_samples = torch.Tensor(reshaped_samples[len(reshaped_samples)-test_len:]).float()

    input_size = len(reshaped_samples[0][0]) # num of features at each time step
    # hyper-param
    reduction_factor = 0.5 # 0.3
    bottleneck_size = int(input_size * reduction_factor)
    print('bottleneck_size', bottleneck_size)
    step_size = 10 ** -3
    regu_lam = 10 ** -4
    epochs = 1 # 750
    
    print('input shape', samples.shape)
    # build model
    model = lstm_autoencoder(input_size, bottleneck_size)
    loss_func = nn.MSELoss()
    opt = optim.Adam(params=model.parameters(), lr=step_size, weight_decay=regu_lam)

    loss_epoch = list()
    # training
    print('####### Start training linear mlp autoencoder ###############')
    for epoch in range(epochs):
        model.zero_grad()
        out = model(samples)
        print('output shape', out.shape)
        loss = loss_func(out, samples)
        loss.backward()
        opt.step()
        
        # validation, for inspecting proper epoch nums
        # test_out = model(test_samples)
        # test_loss = loss_func(test_out, test_samples)
        # loss_epoch.append(test_loss.item())
    
    # plt.plot([i for i in range(epochs)], loss_epoch)
    # plt.show()

    ###################
    # # validation
    # test_out = model(test_samples)
    # loss = loss_func(test_out, test_samples)
    # print('test loss', loss.item())
    # test_out = test_out.detach().numpy()
    # test_samples = test_samples.detach().numpy()

    # for i in range(len(test_samples)):
    #     time = [i for i in range(len(test_samples[i]))]
    #     plt.plot(time, test_samples[i], c='b')
    #     plt.plot(time, test_out[i], c='r')
    #     plt.show()

    # print('######### training complete ###############')
    # model.encode_only = True
    # return model
    