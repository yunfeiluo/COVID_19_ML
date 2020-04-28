import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import kaggle

class mlp_linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_linear, self).__init__()
        self.sequential_liner = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.sequential_liner(x)

class conv_unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_unit,self).__init__()
        self.sequential_liner = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.sequential_liner(x)

class mlp_conv(nn.Module):
    def __init__(self, input_size, conv_size1, conv_size2, num_classes):
        super(mlp_conv, self).__init__()
        self.sequential_liner1 = nn.Sequential(
            conv_unit(in_channels=3, out_channels=conv_size1), # -> conv_size1*32*32
            conv_unit(in_channels=conv_size1, out_channels=conv_size2), # -> conv_size2 * 32 * 32
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # -> conv_size*16*16
        )
        
        self.sequential_liner2 = nn.Sequential(
            nn.Linear(in_features=conv_size2*16*16, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = num_classes)
        )

    def forward(self, x):
        out = self.sequential_liner1(x)
        out = out.view(-1,conv_size2*16*16)
        out = self.sequential_liner2(out)
        return out

# helper function(s)
def read_data():
    print('Reading image data ...')
    temp = np.load('../../Data/data_train.npy')
    train_x = temp
    temp = np.load('../../Data/train_labels.npy')
    train_y = temp
    temp = np.load('../../Data/data_test.npy')
    test_x = temp
    return train_x, train_y, test_x

def write_data(pred, filename):
    print('Writing output to ', filename)
    kaggle.kaggleize(pred, filename)

# main driver
if __name__ == '__main__':
    # get data
    train_x, train_y, test_x = read_data()

    # convert data to tensor, and add autograd
    train_x = torch.Tensor(train_x).float()
    train_x = train_x.view(len(train_x), 3, 32, 32)
    train_x = autograd.Variable(train_x).float()

    train_y = torch.Tensor(train_y).long()
    #train_y = train_y.view(-1, 1)

    test_x = torch.Tensor(test_x).float()
    test_x = test_x.view(len(test_x), 3, 32, 32)
    
    input_size = len(train_x[0])

    # hyper parameter
    conv_size1 = 18
    conv_size2 = 32
    num_classes = 4
    learning_rate = 0.001
    lam = 0.0001
    epochs = 10

    # build model
    model = mlp_conv(input_size, conv_size1, conv_size2, num_classes)
    #criterion = torch.nn.MSELoss()
    #criterion = lambda out, tar: torch.mean(torch.abs(out - tar))
    criterion = nn.CrossEntropyLoss()
    # optimizer
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=lam)
    print('start training ...')
    # train model
    batch_size = 1000
    for epoch in range(epochs):
        print('epoch: ' + str(epoch + 1) + '/' + str(epochs))
        i = 0
        it = 1
        #while i < 6000:
        obj = 0
        while i < len(train_x):
        #while i < 1000:
            #print('batch ' + str(it))
            model.zero_grad()
            out = model(train_x[i:i+batch_size])
            loss = criterion(out, train_y[i:i+batch_size])
            loss.backward()
            opt.step()
            
            obj += loss.item()
            # print('loss: ' + str(obj))

            i += batch_size
            it += 1
        print('loss: ' + str(obj))
        print(' ')
    
    # made prediction
    # predicted_y = model(test_x).detach().numpy()
    # predicted_y = np.argmax(predicted_y, axis=1)
    # write_data(predicted_y, '../Predictions/best.csv')
    