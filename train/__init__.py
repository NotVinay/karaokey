import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optimizer
from torch.autograd import Variable
from train.Model import LSTM_Model
from tensorboardX import SummaryWriter
from config.model_config import TRAIN_CONFIG
from dataset import Dataset



def random_sampler(
    dataset, nb_frames=128
):
    while True:
      X, Y = dataset[np.random.randint(0, len(dataset))]
      nb_total_frames, nb_bins = X.shape

      start = np.random.randint(0, X.shape[0] - nb_frames)
      cur_X = X[start:start+nb_frames, :]
      cur_Y = Y[start:start+nb_frames, :]
      yield dict(X=cur_X, Y=cur_Y)


if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(42)
    if cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = LSTM_Model(
        nb_features=TRAIN_CONFIG.NB_BINS,
        nb_frames=TRAIN_CONFIG.NB_SAMPLES,
        hidden_size=TRAIN_CONFIG.HIDDEN_SIZE,
        input_mean=None,
        input_scale=None,
        output_mean=None,
    ).to(device)

    optimizer = optimizer.RMSprop(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    dataset = Dataset(dir_path=r"C:\Users\w1572032.INTRANET.000\Desktop\pro_dataset",
                      set="train",
                      source_label="vocals",
                      lazy_load=True)

    train_gen = random_sampler(dataset, nb_frames=TRAIN_CONFIG.NB_SAMPLES)

    # set the training mode on pytorch
    model.train()
    dummy_batch = Variable(torch.rand(TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))

    writer = SummaryWriter(comment='LSTM_Model')
    writer.add_graph(LSTM_Model(nb_features=TRAIN_CONFIG.NB_BINS, nb_frames=TRAIN_CONFIG.NB_SAMPLES, hidden_size=TRAIN_CONFIG.HIDDEN_SIZE), dummy_batch, True)

    # add batch dimension 1
    X = np.zeros((TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))
    Y = np.zeros((TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))
    for i in tqdm.tqdm(range(1000)):

        # assemble batch
        for k in range(TRAIN_CONFIG.NB_BATCHES):
            train_sample = next(train_gen)
            X[k] = np.copy(train_sample['X'])
            Y[k] = np.copy(train_sample['Y'])

        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Yt = torch.tensor(Y, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        Y_hat = model(Xt)

        loss = criterion(Y_hat, Yt)
        writer.add_scalar('loss', loss.item(), i)
        loss.backward()
        optimizer.step()

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json(r"H:/Graph/all_scalars.json")
    writer.close()
