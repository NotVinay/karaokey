from torch.nn import Module, LSTM, Linear, Parameter
import torch.nn.functional as F
import torch
import numpy as np


class LSTM_Model(Module):
    def __init__(self,
                 nb_features,
                 nb_frames,
                 hidden_size=256,
                 nb_layers=1,
                 input_mean=None,
                 input_scale=None,
                 output_mean=None):

        super(LSTM_Model, self).__init__()

        # set the hidden size
        self.hidden_size = hidden_size

        # create parameters with torch tensors for mean and scale
        self.input_mean = Parameter(torch.from_numpy(np.copy(input_mean).astype(np.float32)))

        self.input_scale = Parameter(torch.from_numpy(np.copy(input_scale).astype(np.float32)))

        # fully connected dense layer for input dimensionality reduction
        self.fc_dr = Linear(
            in_features=nb_features,
            out_features=hidden_size
        )

        # LSTM layer
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=nb_layers,
            batch_first=True,
            bidirectional=False
        )

        # fully connected dense layer for input dimensionality expansion
        self.fc_de = Linear(
            in_features=hidden_size,
            out_features=nb_features
        )

        self.output_scale = Parameter(
            torch.ones(nb_features)
        )

        self.output_mean = Parameter(
            torch.from_numpy(np.copy(output_mean).astype(np.float32))
        )

    def forward(self, x):
        nb_batches, nb_frames, nb_features = x.data.shape
        x -= self.input_mean
        x /= self.input_scale

        # reduce input dimensionality
        x = self.fc_dr(x.reshape(-1, nb_features))

        # tanh squashing range ot [-1, 1]
        x = torch.tanh(x)

        # making sure that the shape of the tensors are correct to be feed into LSTM
        x = x.reshape(nb_batches, nb_frames, self.hidden_size)

        # feed into LSTM layer(s)
        x, state = self.lstm(x)

        # making sure that the shape of the tensors are correct to be feed into next FC layer
        x = x.reshape(-1, self.hidden_size)

        # dimensionality expansion layer using fully connected dense layer to regain the original shape
        x = self.fc_de(x)

        # reshaping the
        x = x.reshape(nb_batches, nb_frames, nb_features)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        x = F.relu(x)

        return x


class Generalised_LSTM_Model(Module):
    def __init__(self,
                 nb_features,
                 nb_frames,
                 hidden_size=256,
                 nb_layers=1,
                 bidirectional=False,
                 input_mean=None,
                 input_scale=None,
                 output_mean=None,
                 activation_function="relu"):

        super(Generalised_LSTM_Model, self).__init__()

        # set the hidden size
        self.hidden_size = hidden_size

        # create parameters with torch tensors for mean and scale
        self.input_mean = Parameter(torch.from_numpy(np.copy(input_mean).astype(np.float32)))

        self.input_scale = Parameter(torch.from_numpy(np.copy(input_scale).astype(np.float32)))

        # fully connected dense layer for input dimensionality reduction
        self.fc_dr = Linear(
            in_features=nb_features,
            out_features=hidden_size
        )

        # LSTM layer
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=nb_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # fully connected dense layer for input dimensionality expansion
        self.fc_de = Linear(
            in_features=hidden_size,
            out_features=nb_features
        )

        self.output_scale = Parameter(
            torch.ones(nb_features)
        )

        self.output_mean = Parameter(
            torch.from_numpy(np.copy(output_mean).astype(np.float32))
        )

        activation_functions = {'relu':F.relu, 'tanh': torch.tanh}
        self.activation_function = activation_functions[activation_function]

    def forward(self, x):
        nb_batches, nb_frames, nb_features = x.data.shape
        x -= self.input_mean
        x /= self.input_scale

        # reduce input dimensionality
        x = self.fc_dr(x.reshape(-1, nb_features))

        # tanh squashing range ot [-1, 1]
        x = self.activation_function(x)

        # making sure that the shape of the tensors are correct to be feed into LSTM
        x = x.reshape(nb_batches, nb_frames, self.hidden_size)

        # feed into LSTM layer(s)
        x, state = self.lstm(x)

        # making sure that the shape of the tensors are correct to be feed into next FC layer
        x = x.reshape(-1, self.hidden_size)

        # dimensionality expansion layer using fully connected dense layer to regain the original shape
        x = self.fc_de(x)

        # reshaping the
        x = x.reshape(nb_batches, nb_frames, nb_features)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        x = self.activation_function(x)

        return x

