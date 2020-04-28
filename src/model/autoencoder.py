import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input_seq):
        tt = torch.cuda if self.isCuda else torch
        # h0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, input_seq.size(0), self.hidden_size))
        # c0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, input_seq.size(0), self.hidden_size))

        encoded_input, hidden = self.lstm(input_seq)
        encoded_input = self.relu(encoded_input)
        return encoded_input

    def get_encoded_input_size(self):
        return self.hidden_size


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        """

        @param hidden_size: Hidden size is the size of the encoded input, usually encoder hidden size.
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda

        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.output_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        # h0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        # c0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output


class LSTMAE(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, isCuda=False):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size,
                                  hidden_size,
                                  num_layers,
                                  isCuda)
        self.decoder = DecoderRNN(self.encoder.get_encoded_input_size(),
                                  input_size,
                                  num_layers,
                                  isCuda)

    def forward(self, input_seq):
        encoded_input = self.encoder(input_seq)
        decoded_output = self.decoder(encoded_input)
        return decoded_output

    def get_bottleneck_features(self, input_seq):
        return self.encoder(input_seq)

# ordinal lstm model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[len(input_seq) - 7:].view(1, -1)

# main driver
if __name__ == '__main__':
    data = None
    with open('src/feature_engineering/time_series.pkl', 'rb') as f:
        data = pickle.load(f)
    
    x = data['China_Beijing'][2]
    y = [i for i in range(len(x))]

    test_len = 7
    train_x = x[:len(x) - test_len]
    train_num = len(train_x)
    test_x = x[len(x) - test_len:]
    
    train_x = torch.Tensor(train_x).float()
    train_x = autograd.Variable(train_x).float()
    test_x = torch.Tensor(test_x).float()

    model = LSTM(1, 10, 1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10.0)

    loss_thres = 5000.0
    min_loss = loss_thres
    
    samples = list()
    labels = list()
    for i in range(len(train_x) - test_len - 30):
        x_train = train_x[i:i+30]
        x_train = autograd.Variable(x_train).float()
        samples.append(x_train)
        label = train_x[i+30:i+30+test_len]
        labels.append(label)

    epochs = 100
    for epoch in range(epochs):
        print('epoch', epoch)
        last_loss = -1
        for i in range(len(samples)):
            optimizer.zero_grad()
            y_pred = model(samples[i])[0]

            loss = loss_function(y_pred, labels[i])
            loss.backward(retain_graph=True)

            if loss.item() < min_loss:
                min_loss = loss.item()
                last_loss = loss.item()
                print('break point', loss.item())
                break
            if loss.item() < loss_thres:
                loss_thres /= 2
                for param in optimizer.param_groups:
                    param['lr'] /= 2

            optimizer.step()

            last_loss = loss.item()
        print('loss epoch {}, {}'.format(epoch, last_loss))
    y_pred = model(train_x[train_num - test_len:])[0].detach().numpy()
    x = [i for i in range(test_len)]
    plt.scatter(x, y_pred, c='r')
    plt.plot(x, y_pred, c='r')

    plt.scatter(x, test_x, c='b')
    plt.plot(x, test_x, c='b')
    plt.show()

    # # convert data to tensor, and add autograd
    # #train_y = train_y.view(-1, 1)

    # # build model
    # print('start training ...')
    # # train model
    # batch_size = 1000
    # for epoch in range(epochs):
    #     print('epoch: ' + str(epoch + 1) + '/' + str(epochs))
    #     i = 0
    #     it = 1
    #     #while i < 6000:
    #     obj = 0
    #     while i < len(train_x):
    #     #while i < 1000:
    #         #print('batch ' + str(it))
    #         model.zero_grad()
    #         out = model(train_x[i:i+batch_size])
    #         loss = criterion(out, train_y[i:i+batch_size])
    #         loss.backward()
    #         opt.step()
            
    #         obj += loss.item()
    #         # print('loss: ' + str(obj))

    #         i += batch_size
    #         it += 1
    #     print('loss: ' + str(obj))
    #     print(' ')
    
    # # made prediction
    # # predicted_y = model(test_x).detach().numpy()
    # # predicted_y = np.argmax(predicted_y, axis=1)
    # # write_data(predicted_y, '../Predictions/best.csv')
    