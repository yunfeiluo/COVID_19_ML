import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing

class dense_mlp_autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size, encode_only=False):
        super(dense_mlp_autoencoder, self).__init__()
        self.encode_only = encode_only

        self.encoder_liner = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size),
            #nn.ReLU(),
        )
        self.decoder_liner = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
        )
    
    def forward(self, input_seq):
        compress = self.encoder_liner(input_seq)
        if self.encode_only:
            return compress
        reconstruct = self.decoder_liner(compress)
        return reconstruct

def train_dense_mlp_autoencoder(train_samples):
    random.shuffle(train_samples)
    #test_len = int(len(train_samples) * 0.2)
    test_len = 0
    print('num of test', test_len)

    samples = train_samples[:len(train_samples)-test_len]
    #samples = preprocessing.normalize(samples)
    samples = torch.Tensor(samples).float()
    samples = autograd.Variable(samples).float()

    test_samples = torch.Tensor(train_samples[len(train_samples)-test_len:]).float()

    input_size = len(train_samples[0])
    # hyper-param
    hidden_size = 128
    reduction_factor = 0.3
    bottleneck_size = int(input_size * reduction_factor)
    print('bottleneck_size', bottleneck_size)
    step_size = 10 ** -3
    regu_lam = 10 ** -4
    epochs = 500
    
    # build model
    model = dense_mlp_autoencoder(input_size, hidden_size, bottleneck_size)
    loss_func = nn.MSELoss()
    opt = optim.Adam(params=model.parameters(), lr=step_size, weight_decay=regu_lam)

    # training
    print('####### Start training linear mlp autoencoder ###############')
    for epoch in range(epochs):
        model.zero_grad()
        out = model(samples)
        loss = loss_func(out, samples)
        loss.backward()
        opt.step()
        
        if epoch % 10 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
    
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

    print('######### training complete ###############')
    model.encode_only = True
    return model
    