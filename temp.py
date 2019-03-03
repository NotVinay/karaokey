from dataset import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

# def create_batches(x, y, nb_frames):
#     """
#     Create samples of n frames from the data
#     Parameters
#     ----------
#     x : input data
#     y : label data
#     n : sample size
#
#     Returns
#     -------
#     ndarray
#
#     """
#     x_batch = []
#     y_batch = []
#     for i in range(0, len(x), nb_frames):
#         if (i + nb_frames) > len(x):
#             x_batch.append(x[-nb_frames:])
#             y_batch.append(x[-nb_frames:])
#         else:
#             x_batch.append(x[i:i + nb_frames])
#             y_batch.append(y[i:i + nb_frames])
#     return np.array(x_batch), np.array(y_batch)


dataset = Dataset(dir_path=r"C:\Users\w1572032.INTRANET.000\Desktop\pro_dataset",
                  sub_set="train",
                  source_label="vocals",
                  lazy_load=False)

# dataset_test = Dataset(dir_path=r"C:\Users\w1572032.INTRANET.000\Desktop\pro_dataset_test",
#                   set="train",
#                   source_label="vocals",
#                   lazy_load=True)
#
# for i in range(len(dataset)):
#     X, Y = dataset[i]
#     X_test, Y_test = dataset_test[i]
#     if np.array_equal(X, X_test) and np.array_equal(Y, Y_test):
#         print("array equal")
#
#     if np.array_equal(dataset.mixture_scaler.mean_,  dataset_test.mixture_scaler.mean_):
#         print("mixture mean equal")
#     if np.array_equal(dataset.mixture_scaler.scale_, dataset_test.mixture_scaler.scale_):
#         print("mixture scale equal")
#
#     if np.array_equal(dataset.label_scaler.mean_, dataset_test.label_scaler.mean_):
#         print("label mean equal")

# mixtures = dataset.mixtures
# labels = dataset.labels

# X = []
# Y = []
#
# for i, val in enumerate(mixtures):
#     print(i)
#     x_b, y_b = create_batches(mixtures[i], labels[i], nb_frames=128)
#     for x_s in x_b:
#         X.append(x_s)
#     for y_s in y_b:
#         Y.append(y_s)
#
# X=np.array(X)
# Y=np.array(Y)
# np.save(r"C:\Users\w1572032.INTRANET.000\Desktop\sampled_dataset_x.npy", X)
# np.save(r"C:\Users\w1572032.INTRANET.000\Desktop\sampled_dataset_y.npy", Y)
#
# print(X.shape)
# print(Y.shape)