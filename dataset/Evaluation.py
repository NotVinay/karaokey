import museval
import preprocess.utility as util
import os
import numpy as np

loc = os.path.abspath(r"C:\Users\w1572032.INTRANET.001\Downloads")

acc, sr = util.read(os.path.join(loc, "accompaniment.wav"))
vocals, sr = util.read(os.path.join(loc, "vocals.wav"))
references = np.array([acc, vocals])
acc_estimates, sr = util.read(os.path.join(loc, "accompaniment_estimate.wav"))
vocals_estimates, sr = util.read(os.path.join(loc, "vocals_estimate.wav"))
estimates = np.array([acc_estimates[0:acc.shape[0], :],
                      vocals_estimates[0:vocals.shape[0], :]])
print(references.shape)
print(estimates.shape)
SDR, SIR, ISR, SAR = museval.evaluate(references, estimates, win=sr, hop=sr)
print(SDR)
print(np.mean(SDR, axis=1))
print(np.mean(SIR, axis=1))
print(np.mean(ISR, axis=1))
print(np.mean(SAR, axis=1))