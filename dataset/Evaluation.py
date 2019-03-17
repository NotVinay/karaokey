import museval
import preprocess.utility as util
import os
import numpy as np

loc = os.path.abspath(r"C:\Users\w1572032.INTRANET.001\Downloads")

acc_estimates, sr = util.read(os.path.join(loc, "accompaniment_estimate.wav"))
vocals_estimates, sr = util.read(os.path.join(loc, "vocals_estimate.wav"))
estimates = np.array([acc_estimates, vocals_estimates])
acc, sr = util.read(os.path.join(loc, "accompaniment.wav"))
vocals, sr = util.read(os.path.join(loc, "vocals.wav"))
references = np.array([acc, vocals])
SDR, SIR, ISR, SAR = museval.evaluate(references, estimates)
SDR_mean = np.mean(SDR, axis=1)
print(SDR)
print(SDR_mean)
print(np.mean(SIR, axis=1))
print(np.mean(ISR, axis=1))
print(np.mean(SAR, axis=1))