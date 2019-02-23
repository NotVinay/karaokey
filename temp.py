from dataset import Dataset
import numpy as np

def create_batches(x, y, nb_frames):
    """
    Create samples of n frames from the data
    Parameters
    ----------
    x : input data
    y : label data
    n : sample size

    Returns
    -------
    ndarray

    """
    x_batch = []
    y_batch = []
    for i in range(0, len(x), nb_frames):
        if (i + nb_frames) > len(x):
            x_batch.append(x[-nb_frames:])
            y_batch.append(x[-nb_frames:])
        else:
            x_batch.append(x[i:i + nb_frames])
            y_batch.append(y[i:i + nb_frames])
    return np.array(x_batch), np.array(y_batch)


dataset = Dataset(dir_path=r"C:\Users\w1572032\Desktop\short_pro_dataset",
                  set="train",
                  source_label="vocals",
                  lazy_load=False)
mixtures = dataset.mixtures
labels = dataset.labels

for X, Y in dataset:
    print(np.mean(X))
    print(np.mean(Y))
    print("-----------------------------------------------------------------------")


# X = []
# Y = []
# for i, val in enumerate(mixtures):
#     print(i)
#     x_b, y_b = create_batches(mixtures[i], labels[i], nb_frames=128)
#     for x_s in x_b:
#         X.append(x_s)
#     for y_s in y_b:
#         Y.append(y_s)
# X=np.array(X)
# Y=np.array(Y)
# np.save(r"C:\Users\w1572032.INTRANET.000\Desktop\sampled_dataset_x.npy", X)
# np.save(r"C:\Users\w1572032.INTRANET.000\Desktop\sampled_dataset_y.npy", Y)
# print(X.shape)
# print(Y.shape)