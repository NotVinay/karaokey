from torch.nn import Module, LSTM, GRU, RNN, Linear, Parameter
import torch.nn.functional as F
import torch
import numpy as np
import ipdb

__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Development"


class Generalised_Recurrent_Model(Module):
    """
    Generalised Model for neural networks with recurrent layers.

    The structure of the Model is:

    - **Fully Connected Layer** Dimensionality Reduction layer used for reducing the dimensionality of the input features. If recurrent layer features are too many it can have problems with long term dependencies (RNN will suffer from it the most).
    - **Recurrent Layer** It learns the time dependencies over the sample frames. LSTMs and GRUs will provide the best output, however RNN can be efficient and could provide good result over
    - **Fully Connected Layer** Dimensionality Expansion layer used for transforming the learned features to original shape.

    Attributes
    ----------
    nb_features: int
        number of features(nb of bins) of input data.
    nb_frames: int
        number of frames or samples in each batch.
    nb_layers: int
        number of layers for recurrent layer.
    hidden_size: int
        number of hidden cells in the recurrent layer.
    mixture_mean: ndarray, shape(nb_features)
        mean of mixture data.
    mixture_scale: ndarray, shape(nb_features)
        scaled data of mixture.
    label_mean: ndarray, shape(nb_features)
        mean of label data.
    activation_function: str
        activation function to use. "relu" by default, if "tanh" than tanh is used/
    recurrent_layer: str
        recurrent layer to use. "lstm" by defalt, use "rnn" for RNN and "gru" for GRU.
    """
    def __init__(self,
                 nb_features,
                 nb_frames,
                 nb_layers,
                 hidden_size,
                 bidirectional=False,
                 mixture_mean=None,
                 mixture_scale=None,
                 label_mean=None,
                 activation_function="relu",
                 recurrent_layer="lstm"):
        super(Generalised_Recurrent_Model, self).__init__()

        # set the hidden size
        self.hidden_size = hidden_size

        # create parameters with torch tensors for mean and scale
        self.mixture_mean = Parameter(torch.from_numpy(np.copy(mixture_mean).astype(np.float32)))

        self.label_scale = Parameter(torch.from_numpy(np.copy(mixture_scale).astype(np.float32)))

        # fully connected dense layer for input dimensionality reduction
        self.fc_dr = Linear(in_features=nb_features,
                            out_features=hidden_size)

        # different recurrent layers
        recurrent_layers = {'lstm': LSTM(input_size=hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=nb_layers,
                                         batch_first=True,
                                         bidirectional=bidirectional),
                            'gru': GRU(input_size=hidden_size,
                                       hidden_size=hidden_size,
                                       num_layers=nb_layers,
                                       batch_first=True,
                                       bidirectional=bidirectional),
                            'rnn': RNN(input_size=hidden_size,
                                       hidden_size=hidden_size,
                                       num_layers=nb_layers,
                                       batch_first=True,
                                       bidirectional=bidirectional)}
        # recurrent layer
        self.recurrent_layer = recurrent_layers[recurrent_layer]

        self.lstm_output = hidden_size * 2 if bidirectional else hidden_size

        # fully connected dense layer for input dimensionality expansion
        self.fc_de = Linear(
            in_features=self.lstm_output,
            out_features=nb_features
        )

        # output label scaling
        self.label_scale = Parameter(torch.ones(nb_features))

        # output label mean
        self.label_mean = Parameter(torch.from_numpy(np.copy(label_mean).astype(np.float32)))

        # activation function
        activation_functions = {'relu': F.relu, 'tanh': torch.tanh}
        self.activation_function = activation_functions[activation_function]

    def forward(self, x):
        """
        Every call of the forward pass.

        Parameters
        ----------
        x : input data to the model

        Returns
        -------
        tensor (Tensor)
            output data from forward pass.
        """
        # ipdb.set_trace()
        nb_batches, nb_frames, nb_features = x.data.shape
        x -= self.mixture_mean
        x /= self.label_scale

        # flattens the inputs so that it can be used for fully connected layer
        # shape (nb_batches*nb_frames, nb_features)
        x = x.reshape(-1, nb_features)

        # reduce input dimensionality
        # parameters are reduced to make sure the recurrent layer do not have problems learning long term dependencies.
        x = self.fc_dr(x)

        # tanh: squashing range ot [-1, 1]
        # relu: x if (x>0) or 0 otherwise
        x = self.activation_function(x)

        # making sure that the shape of the tensors are correct to be feed into recurrent layer
        x = x.reshape(nb_batches, nb_frames, self.hidden_size)

        # feeded the reduced inputs into recurrent layer(s)
        x, state = self.recurrent_layer(x)

        # making sure that the shape of the tensors are correct to be feed into next FC layer
        # shape (nb_batches*nb_frames, lstm_output)
        x = x.reshape(-1, self.lstm_output)

        # dimensionality expansion layer using fully connected dense layer to regain the original shape
        x = self.fc_de(x)

        # reshaping the
        x = x.reshape(nb_batches, nb_frames, nb_features)

        # apply output scaling
        x *= self.label_scale
        x += self.label_mean

        x = self.activation_function(x)
        return x
