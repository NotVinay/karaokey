import museval
from mir_eval.separation import bss_eval_images_framewise as bss_eval
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
SDR_whole, SIR_whole, ISR_whole, SAR_whole, _ = bss_eval(references,
                                                         estimates,
                                                         window=sr,
                                                         hop=sr,
                                                         compute_permutation=False)
#SDR_whole, SIR_whole, ISR_whole, SAR_whole = museval.evaluate(references, estimates, win=acc_estimates.shape[0], hop=acc_estimates.shape[0])
print(SDR_whole)
print(np.mean(SDR_whole, axis=1))
print(np.mean(SIR_whole, axis=1))
print(np.mean(ISR_whole, axis=1))
print(np.mean(SAR_whole, axis=1))
