from dataset import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

dataset_test = Dataset(dir_path=r"C:\Users\w1572032.INTRANET.001\Desktop\pro_dataset_test",
                  sub_set="test",
                  source_label="accompaniment",
                  lazy_load=True)
mixture_scaler = np.load(r"C:\Users\w1572032.INTRANET.001\Desktop\pro_dataset\test_metadata\mixture_scaler.npy").item()
label_scaler = np.load(r"C:\Users\w1572032.INTRANET.001\Desktop\pro_dataset\test_metadata\accompaniment_scaler.npy").item()

if np.array_equal(mixture_scaler.mean_, dataset_test.mixture_mean):
    print("mixture mean equal")

if np.array_equal(mixture_scaler.scale_,  dataset_test.mixture_scale):
    print("mixture scale equal")

if np.array_equal(label_scaler.mean_, dataset_test.label_mean):
    print("label mean equal")

if np.array_equal(label_scaler.scale_, dataset_test.label_scale):
    print("label scale equal")
