# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import wfdb

# Retrieve ECG data from data folder
ecg_signal = nk.data(dataset="ecg_1000hz")
ecg_signal = ecg_signal[:10000]

# Set sample-rate
sample_rate = 1000

cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=sample_rate)
_, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=sample_rate)

# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)

plt.show()

# Zooming into the first 5 R-peaks
plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:], ecg_signal)

plt.show()

# Delineate the ECG signal
_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sample_rate, method="peak")


s, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sample_rate, method="peak")
print(s.columns)
print(s)
s.to_csv("data/s.csv", index=False)
print(waves_peak)

print(waves_peak['ECG_P_Peaks'])
print("")
print(waves_peak['ECG_Q_Peaks'])
print("")
print(rpeaks['ECG_R_Peaks'])
print("")
print(waves_peak['ECG_S_Peaks'])
print("")
print(waves_peak['ECG_T_Peaks'])

# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:],
                       waves_peak['ECG_P_Peaks'][:],
                       waves_peak['ECG_Q_Peaks'][:],
                       waves_peak['ECG_S_Peaks'][:]], ecg_signal)

plt.show()

# Delineate the ECG signal and visualizing all peaks of ECG complexes
_, waves_peak = nk.ecg_delineate(ecg_signal,
                                 rpeaks,
                                 sampling_rate=sample_rate,
                                 method="peak",
                                 show=True,
                                 show_type='peaks')

plt.show()