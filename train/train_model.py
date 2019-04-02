import numpy as np
import librosa as lib
import tqdm
import torch
import torch.nn.functional as F
import time, datetime
from torch.autograd import Variable
from train.model import LSTM_Model, Generalised_Recurrent_Model
from tensorboardX import SummaryWriter
from config.model_config import TRAIN_CONFIG, DATASET_CONFIG
from dataset import Dataset
import museval
import norbert
import preprocess.utility as sp
from preprocess.preprocess_tools import Scaler, STFT
from preprocess.data import Data
import common.input_handler as uin
import os

def random_batch_sampler(dataset, nb_frames=128):
    """
    Random batch sampler.

    Parameters
    ----------
    dataset : Dataset
        Dataset object
    nb_frames : int
        number of samples or frames per batch.

    Returns
    -------
    dict
        dictionary with mixture and label
    """
    while True:
        i = np.random.randint(0, len(dataset))
        mixture, label = dataset[i]
        nb_total_frames, nb_bins, nb_channels = mixture.shape
        start = np.random.randint(0, mixture.shape[0] - nb_frames)
        cur_mixture = mixture[start:start + nb_frames, :, 0]
        cur_label = label[start:start + nb_frames, :, 0]
        yield dict(mixture=cur_mixture, label=cur_label)


def train(dnn_model,
          device,
          dataset_loader,
          loss_function,
          optimizer,
          writer):
    """
    trains the `dnn_model`

    Parameters
    ----------
    dnn_model : Generalised_Recurrent_Model
        Model to Train
    device : torch.device
        device to use
    dataset_loader : object
        Dataset batch generator
    loss_function : object
        loss function either relu or tanh from PyTorch
    optimizer : object
        optimizer to use for example, Adam or RMSProp
    writer : SummaryWriter
        summary writer for writing TensorBoard summaries
    """
    # setting training mode on
    dnn_model.train()
    # batch holders
    X = np.zeros((TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))
    Y = np.zeros((TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))

    # performing training steps
    for i in tqdm.tqdm(range(TRAIN_CONFIG.STEPS)):
        # assembling batches
        for k in range(TRAIN_CONFIG.NB_BATCHES):
            train_sample = next(dataset_loader)
            X[k] = np.copy(train_sample['mixture'])
            Y[k] = np.copy(train_sample['label'])

        # converting to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)

        # resets the gradients for new epoch.
        optimizer.zero_grad()
        # forward pass on X data
        Y_hat = dnn_model(X_tensor)
        # calculating loss
        loss = loss_function(Y_hat, Y_tensor)

        # logging loss
        writer.add_scalar('loss', loss.item(), i)

        # backward propagation of algorithm
        loss.backward()

        # optimizer.step updates the parameters to optimize the loss
        optimizer.step()


def evaluation(dnn_model,
               device,
               test_tracks,
               writer,
               full_evaluation=True,
               trained_on="vocals"):
    """

    Parameters
    ----------
    dnn_model : Generalised_Recurrent_Model
        Model to use for prediction
    test_tracks :
    device : torch.device
        device to use
    writer : SummaryWriter
        summary writer for writing TensorBoard summaries
    full_evaluation : bool
        True if full evaluation is to be performed, False if only one track needs to be evaluated
    trained_on : str
        Labels of the trained model "vocals" or "accompaniment"

    """
    # setting the evaluation mode
    dnn_model.eval()

    # gradients are not needed for evaulation to turning it off
    with torch.no_grad():
        sdr_means = []
        sir_means = []
        isr_means = []
        sar_means = []
        # iterate over sample the tracks
        for track_number, track in enumerate(test_tracks):
            # getting predicted estimates of accompaniment and vocals
            acc_estimate, vocals_estimate = predict(dnn_model,
                                                    device,
                                                    data=track.mixture.data,
                                                    sr=track.mixture.sr,
                                                    trained_on=trained_on)
            # adding it to list for evaluating metrics
            estimates_list = np.array([vocals_estimate, acc_estimate])
            reference_list = np.array(
                [np.copy(track.sources["vocals"].data), np.copy(track.sources["accompaniment"].data)])

            # evaluating the metrics
            SDR_mean, SIR_mean, ISR_mean, SAR_mean = museval.evaluate(reference_list, estimates_list, win=track.mixture.data.shape[0], hop=track.mixture.data.shape[0])

            # getting mean of the metrics
            # SDR_mean = np.mean(SDR, axis=1)
            # SIR_mean = np.mean(SIR, axis=1)
            # ISR_mean = np.mean(ISR, axis=1)
            # SAR_mean = np.mean(SAR, axis=1)
            print(track_number, ": ", SDR_mean.shape, ", ", SDR_mean.shape)

            # logging METRICS for vocals
            writer.add_scalar('vocals/SDR_mean', SDR_mean[0], track_number)
            writer.add_scalar('vocals/SIR_mean', SIR_mean[0], track_number)
            writer.add_scalar('vocals/SAR_mean', SAR_mean[0], track_number)
            writer.add_scalar('vocals/ISR_mean', ISR_mean[0], track_number)

            # logging METRICS for accompaniment
            writer.add_scalar('accompaniment/SDR_mean', SDR_mean[1], track_number)
            writer.add_scalar('accompaniment/SIR_mean', SIR_mean[1], track_number)
            writer.add_scalar('accompaniment/SAR_mean', SAR_mean[1], track_number)
            writer.add_scalar('accompaniment/ISR_mean', ISR_mean[1], track_number)

            # appending it to the means of the all tracks
            sdr_means.append(SDR_mean)
            sir_means.append(SIR_mean)
            isr_means.append(ISR_mean)
            sar_means.append(SAR_mean)
            # saving the first sample
            if track_number == 0:
                mono_vocals_estimate_normalized = lib.util.normalize(sp.to_mono(vocals_estimate))
                print("normalized")
                writer.add_audio(tag="vocals",
                                 snd_tensor=torch.from_numpy(mono_vocals_estimate_normalized),
                                 global_step=1,
                                 sample_rate=track.mixture.sr)
                mono_acc_estimate_normalized = lib.util.normalize(sp.to_mono(acc_estimate))
                print("saved")
                writer.add_audio(tag="accompaniment",
                                 snd_tensor=torch.from_numpy(mono_acc_estimate_normalized),
                                 global_step=1,
                                 sample_rate=track.mixture.sr)
                print("FIRST TRACK EVALUATION COMPLETE")
            if not full_evaluation:
                print("ENDING FULL EVALUATION!!")
                break
            # END OF FOR of test samples

        # calculating mean over all tracks and saving it
        sdr_total_mean = np.mean(np.array(sdr_means), axis=0)
        sir_total_mean = np.mean(np.array(sir_means), axis=0)
        isr_total_mean = np.mean(np.array(isr_means), axis=0)
        sar_total_mean = np.mean(np.array(sar_means), axis=0)
        writer.add_text('Model_Configuration', TRAIN_CONFIG.__str__(), 0)
        writer.add_text('sdr_total_mean',
                        "accompaniment: " + str(sdr_total_mean[1]) + "  \n vocals: " + str(sdr_total_mean[0]),
                        0)
        writer.add_text('sir_total_mean',
                        "accompaniment: " + str(sir_total_mean[1]) + "  \n vocals: " + str(sir_total_mean[0]),
                        0)
        writer.add_text('isr_total_mean',
                        "accompaniment: " + str(isr_total_mean[1]) + "  \n vocals: " + str(isr_total_mean[0]),
                        0)
        writer.add_text('sar_total_mean',
                        "accompaniment: " + str(sar_total_mean[1]) + "  \n vocals: " + str(sar_total_mean[0]),
                        0)
        # END OF CONTEXT torch.no_grad()


def predict(dnn_model,
            device,
            data,
            sr,
            trained_on="vocals"):
    """
    Predicts the estimes of vocals and accompaniment using

    Parameters
    ----------
    dnn_model : Generalised_Recurrent_Model
        model to use for prediction
    device : torch.device
        device to use
    data : ndarray(nb_samples, nb_channels)
        data of mixture track in time series
    sr : int
        sampling rate of the mixture track
    trained_on : str
        Labels of the trained model "vocals" or "accompaniment"

    Returns
    -------
    acc_estimate: ndarray, shape(nb_samples, nb_channels)
        Accompaniment estimates in time series
    vocals_estimate: ndarray, shape(nb_samples, nb_channels)
        Vocals estimates in time series
    """
    # transformation object
    transform = STFT(sr=DATASET_CONFIG.SR,
                     n_per_seg=DATASET_CONFIG.N_PER_SEG,
                     n_overlap=DATASET_CONFIG.N_OVERLAP)

    # Scaler object
    scaler = Scaler()

    # convert track to mono track
    if data.shape[1] != 1:
        data = sp.to_mono(data)

    nb_samples, nb_channels = data.shape

    # generate STFT of time series data, shape(nbframes, nb_bins, nb_channels)
    mixture_tf = transform.stft(data.T)

    # get spectrogram of STFT i.e., |Xi|, shape(nbframes, nb_bins, nb_channels)
    mixture_stft = np.abs(mixture_tf)

    # scaling the values to 0 to 1, shape(nbframes, nb_bins, nb_channels)
    X_scaled = scaler.scale(mixture_stft)

    # transposing the matrix to make it in shape (nb_batch, nb_frames, nb_bins)
    X_scaled = np.transpose(X_scaled, (2, 0, 1))

    mixture_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device).to(device)
    estimate = dnn_model(mixture_tensor)

    # output tensor shape (nb_batch, nb_frames, nb_bins)
    estimate_np = estimate[0].cpu().detach().numpy()

    # stacking the output to make it in stereo shape
    # and transposing it back to shape (nb_frames, nb_bins, nb_channels)
    estimate_stereo = np.stack([estimate_np, estimate_np]).transpose(1, 2, 0)
    # intensifies the signal
    estimate_stereo = estimate_stereo[..., None] ** 2

    # stacking the mixture stft to make it in stereo shape
    # and transposing it back to shape (nb_frames, nb_bins, nb_channels)
    mixture_tf_squeeze = np.squeeze(mixture_tf)
    mixture_tf_stereo = np.stack([mixture_tf_squeeze, mixture_tf_squeeze]).transpose(1, 2, 0)

    # models the estimates to stft, frequency wise.
    estimate_residual = norbert.residual(estimate_stereo, mixture_tf_stereo)
    # applying wiener filers to get the sources
    estimate_filter_results = norbert.wiener(np.copy(estimate_residual), np.copy(mixture_tf_stereo))

    # return the estimates based on the source type of the labels
    if trained_on == "vocals":
        vocals_estimate = transform.istft(estimate_filter_results[..., 0]).T
        acc_estimate = transform.istft(estimate_filter_results[..., 1]).T
        return acc_estimate, vocals_estimate
    else:
        acc_estimate = transform.istft(estimate_filter_results[..., 0]).T
        vocals_estimate = transform.istft(estimate_filter_results[..., 1]).T
        return acc_estimate, vocals_estimate


def main():
    # current timestamp
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M')
    use_default_config = uin.get_confirmation(msg="Use default train config",
                                              error_msg="Please enter y or n")
    # getting parameters through console.
    if not use_default_config:
        TRAIN_CONFIG.TRAINED_ON = uin.get_input_str(msg="Please select the label (vocals/accompaniment)",
                                                    only_accept=['vocals', 'accompaniment'],
                                                    error_msg="Please enter vocals or accompaniment")
        TRAIN_CONFIG.NB_BATCHES = uin.get_input_int(msg="Please enter batch size",
                                                    only_accept=[8, 16, 32, 64, 128],
                                                    error_msg="Please enter valid number of layers")
        TRAIN_CONFIG.NB_SAMPLES = uin.get_input_int(msg="Please enter number of samples",
                                                    only_accept=[64, 128, 256, 512],
                                                    error_msg="Please enter valid number of batches")
        TRAIN_CONFIG.HIDDEN_SIZE = TRAIN_CONFIG.NB_SAMPLES * 2
        TRAIN_CONFIG.RECURRENT_LAYER = uin.get_input_str(msg="Please select recurrent layer (rnn/lstm/gru)",
                                                             only_accept=['rnn', 'lstm', 'gru'],
                                                             error_msg="Please select rnn or lstm or gru")
        TRAIN_CONFIG.ACTIVATION_FUNCTION = uin.get_input_str(msg="Please select activation function (relu/tanh)",
                                                             only_accept=['relu', 'tanh'],
                                                             error_msg="Please select relu or tanh")
        TRAIN_CONFIG.NB_LAYERS = uin.get_input_int(msg="Please enter number of LSTM Layers [1 to 3]",
                                                   only_accept=range(1, 3),
                                                   error_msg="Please enter valid number of layers")
        TRAIN_CONFIG.BIDIRECTIONAL = uin.get_confirmation(msg="Use bidirectional LSTM?",
                                                          error_msg="Please enter y or n")
        TRAIN_CONFIG.OPTIMIZER = uin.get_input_str(msg="Please select optimizer (adam/rmsprop)",
                                                   only_accept=['adam', 'rmsprop'],
                                                   error_msg="Please adam or rmsprop")
        TRAIN_CONFIG.STEPS = uin.get_input_int(msg="Please enter number of steps",
                                               only_accept=[1000, 2000, 5000, 7500, 10000],
                                               error_msg="Please enter valid number of steps")

    print(TRAIN_CONFIG.__str__())

    if not uin.get_confirmation(msg="Proceed with above configurations?", error_msg="Please enter y or n"):
        print("TERMINATED")
        return None

    # model name for saving logs and trained model
    MODEL_NAME = timestamp + '_Generalised_Recurrent_Model_' + str(TRAIN_CONFIG.ACTIVATION_FUNCTION) + "_" + str(
        TRAIN_CONFIG.TRAINED_ON) + '_B' + str(
        TRAIN_CONFIG.NB_BATCHES) + '_H' + str(TRAIN_CONFIG.HIDDEN_SIZE) + '_S' + str(TRAIN_CONFIG.STEPS) + "_" + TRAIN_CONFIG.OPTIMIZER

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    dataset = Dataset(dir_path=r"C:\Users\w1572032.INTRANET.001\Desktop\pro_dataset",
                      sub_set="train",
                      source_label=TRAIN_CONFIG.TRAINED_ON,
                      lazy_load=True)
    print(dataset.mixture_scale.shape)
    print(dataset.mixture_mean.shape)

    # initialising the objects
    dnn_model = Generalised_Recurrent_Model(nb_features=TRAIN_CONFIG.NB_BINS,
                                            nb_frames=TRAIN_CONFIG.NB_SAMPLES,
                                            nb_layers=TRAIN_CONFIG.NB_LAYERS,
                                            hidden_size=TRAIN_CONFIG.HIDDEN_SIZE,
                                            bidirectional=TRAIN_CONFIG.BIDIRECTIONAL,
                                            mixture_mean=dataset.mixture_mean,
                                            mixture_scale=dataset.mixture_scale,
                                            label_mean=dataset.label_mean,
                                            activation_function=TRAIN_CONFIG.ACTIVATION_FUNCTION,
                                            recurrent_layer=TRAIN_CONFIG.RECURRENT_LAYER).to(device)

    optimizer_choices = {'adam': torch.optim.Adam,
                         'rmsprop': torch.optim.RMSprop}
    loss_function = torch.nn.MSELoss()
    optimizer = optimizer_choices[TRAIN_CONFIG.OPTIMIZER](dnn_model.parameters(), lr=0.001)
    dataset_loader = random_batch_sampler(dataset, nb_frames=TRAIN_CONFIG.NB_SAMPLES)
    writer = SummaryWriter(log_dir="runs/" + MODEL_NAME)

    # initialize tensorboard graph
    # dummy_batch = Variable(torch.rand(TRAIN_CONFIG.NB_BATCHES, TRAIN_CONFIG.NB_SAMPLES, TRAIN_CONFIG.NB_BINS))
    # writer.add_graph(Generalised_Recurrent_Model(nb_features=TRAIN_CONFIG.NB_BINS,
    #                                             nb_frames=TRAIN_CONFIG.NB_SAMPLES,
    #                                             nb_layers=TRAIN_CONFIG.NB_LAYERS,
    #                                             hidden_size=TRAIN_CONFIG.HIDDEN_SIZE,
    #                                             bidirectional=TRAIN_CONFIG.BIDIRECTIONAL,
    #                                             mixture_mean=dataset.mixture_mean,
    #                                             mixture_scale=dataset.mixture_scale,
    #                                             label_mean=dataset.label_mean,
    #                                             activation_function=TRAIN_CONFIG.ACTIVATION_FUNCTION,
    #                                             recurrent_layer=TRAIN_CONFIG.RECURRENT_LAYER.to(device)
    #                  , dummy_batch, True)

    # train the model
    train(dnn_model, device, dataset_loader, loss_function, optimizer, writer)

    # save the model
    torch.save(dnn_model, "./models/" + MODEL_NAME + ".pt")

    # TESTING THE MODEL
    data = Data(dataset_path=DATASET_CONFIG.PATH)
    test_tracks = data.get_tracks(sub_set="test", labels={'vocals', 'accompaniment'})

    # evaluate the model
    evaluation(dnn_model,
               device,
               test_tracks,
               writer,
               full_evaluation=True,
               trained_on=TRAIN_CONFIG.TRAINED_ON)

    # close the summary writer
    writer.close()


if __name__ == '__main__':
    # run main method if this is ran as alone an script
    main()
