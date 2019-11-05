import numpy as np
import math
from scipy import fftpack
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

path = "./100hz_data/black_huawei/black_gyro_100hz_14102019_143651.csv"
columns = ["timestamp", "x", "y", "z"]
new_columns = ["timestamp", "fft_x", "fft_y", "fft_z"]
df = pd.read_csv(path, names=columns)
new_df = pd.DataFrame(columns=new_columns)
new_df["timestamp"] = df["timestamp"]
new_df["fft_x"] = pd.Series(fftpack.fft(np.asarray(df["x"])))
new_df["fft_y"] = pd.Series(fftpack.fft(np.asarray(df["y"])))
new_df["fft_z"] = pd.Series(fftpack.fft(np.asarray(df["z"])))

WINDOW_SIZE = len(df)
order = 3
f_sample = 100
f_cutoff = 30
nyquist_rate = f_cutoff / (f_sample / 2)

N = WINDOW_SIZE  # samples
f_s = 100  # frequency
t_n = N / f_s  # sec
T = t_n / N

# Plotting of fft
N = len(df)
# Period
T = 0.01
# Create x-axis for time length of signal
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
yf = np.asarray(new_df["fft_z"])
yf = 2.0 / N * np.abs(yf[0:N // 2])
plt.plot(xf, abs(yf[0:N//2]))
plt.xlabel('Frequency')
plt.ylabel(r'Spectral Amplitude')
plt.legend(loc=1)
plt.show()


