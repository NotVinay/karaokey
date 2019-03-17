from torch.nn import Module, LSTM, Linear, Parameter
import torch.nn.functional as F


class Vanilla(Module):
    def __init__(
            self, nb_features, nb_frames, hidden_size=256, nb_layers=1,
            input_mean=None, input_scale=None, output_mean=None
    ):
        super(Vanilla, self).__init__()

        self.hidden_size = hidden_size

        self.input_mean = Parameter(
            torch.from_numpy(np.copy(input_mean)).float()
        )
        self.input_scale = Parameter(
            torch.from_numpy(np.copy(input_scale)).float(),
        )

        self.encode_fc = Linear(
            nb_features, hidden_size
        )

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=nb_layers,
            bidirectional=False,
            batch_first=False
        )

        self.fc = Linear(
            in_features=hidden_size,
            out_features=nb_features
        )

        self.output_scale = Parameter(
            torch.ones(nb_features).float()
        )

        self.output_mean = Parameter(
            torch.from_numpy(np.copy(output_mean)).float()
        )

    def forward(self, x):
        nb_frames, nb_batches, nb_features = x.data.shape

        # debugger: ipdb.set_trace()
        x -= self.input_mean
        x /= self.input_scale
        # reduce input dimensionality
        x = self.encode_fc(x.reshape(-1, nb_features))
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        x, state = self.lstm(x.reshape(nb_frames, nb_batches, self.hidden_size))

        x = self.fc(x.reshape(-1, self.hidden_size))

        x = x.reshape(nb_frames, nb_batches, nb_features)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        x = F.relu(x)

        return x