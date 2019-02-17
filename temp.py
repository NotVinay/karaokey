from dataset import Dataset
import numpy as np
dataset = Dataset(dir_path=r"C:\Users\w1572032.INTRANET.000\Desktop\pro_dataset",
                  set="train",
                  source_label="vocals",
                  lazy_load=False)
mixtures = dataset.mixtures
print(dataset.tracks[59])
labels = dataset.labels
for i, mixture in enumerate(mixtures):
    print(i)
    print(mixture.shape)
print(mixtures.shape)