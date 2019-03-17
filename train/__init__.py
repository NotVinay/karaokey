import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optimizer
import time, datetime
from torch.autograd import Variable
from train.model import LSTM_Model
from tensorboardX import SummaryWriter
from config.model_config import TRAIN_CONFIG
from dataset import Dataset

__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Development"

def random_batch_sampler(
    dataset, nb_frames=128
):
    while True:
        i = np.random.randint(0, len(dataset))
        X, Y = dataset[i]
        nb_total_frames, nb_bins, nb_channels = X.shape
        start = np.random.randint(0, X.shape[0] - nb_frames)
        cur_X = X[start:start+nb_frames, :, 0]
        cur_Y = Y[start:start+nb_frames, :, 0]
        yield dict(X=cur_X, Y=cur_Y)

if __name__ == '__main__':
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M')
    print(timestamp)
    MODEL_NAME = timestamp+'_LSTM_B' + str(TRAIN_CONFIG.NB_BATCHES) + '_H' + str(TRAIN_CONFIG.HIDDEN_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    dataset = Dataset(dir_path=r"C:\Users\w1572032.INTRANET.001\Desktop\pro_dataset",
                      sub_set="train",
                      source_label="vocals",
                      lazy_load=True)
    print(dataset.mixture_scaler.mean_.shape)
    print(dataset.mixture_scaler.scale_.shape)

    dnn_model = LSTM_Model(
        nb_features=TRAIN_CONFIG.NB_BINS,
        nb_frames=TRAIN_CONFIG.NB_SAMPLES,
        hidden_size=TRAIN_CONFIG.HIDDEN_SIZE,
        input_mean=dataset.mixture_scaler.mean_,
        input_scale=dataset.mixture_scaler.scale_,
        output_mean=dataset.label_scaler.mean_,
    ).to(device)

    optimizer = optimizer.RMSprop(dnn_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_gen = random_batch_sampler(dataset, nb_frames=TRAIN_CONFIG.NB_SAMPLES)

    # -----------------TRAINING---------------------

    # setting model on training mode so that weights and parameters can be updated
    dnn_model.train()

    # initialize tensorboard graph
    dummy_batch = Variable(torch.rand(TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))
    writer = SummaryWriter(log_dir="runs/"+MODEL_NAME)
    writer.add_graph(LSTM_Model(nb_features=TRAIN_CONFIG.NB_BINS, nb_frames=TRAIN_CONFIG.NB_SAMPLES, hidden_size=TRAIN_CONFIG.HIDDEN_SIZE), dummy_batch, True)

    # add batch dimension 1
    X = np.zeros((TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))
    Y = np.zeros((TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))
    for i in tqdm.tqdm(range(TRAIN_CONFIG.STEPS)):

        # assemble batch
        for k in range(TRAIN_CONFIG.NB_BATCHES):
            train_sample = next(train_gen)
            X[k] = np.copy(train_sample['X'])
            Y[k] = np.copy(train_sample['Y'])

        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Yt = torch.tensor(Y, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        Y_hat = dnn_model(Xt)

        loss = criterion(Y_hat, Yt)
        writer.add_scalar('loss', loss.item(), i)
        loss.backward()
        optimizer.step()

    torch.save(dnn_model, "./models/" + MODEL_NAME + ".pt")


    writer.close()
