import numpy as np
import wave
from scipy.io import wavfile
import matplotlib.pyplot as plt

sampling_freq, audio = wavfile.read(r"C:\Users\Gtuff\Desktop\DSP-python\sound_me.wav")   # 读取文件

audio = audio / np.max(audio)   # 归一化，标准化

# 应用傅里叶变换
fft_signal = np.fft.fft(audio)

fft_signal = abs(fft_signal)

# 建立频率轴
fft_signal=np.fft.fftshift(fft_signal)
Freq = np.arange(-sampling_freq/2,sampling_freq/2,sampling_freq/len(fft_signal))

# 绘制语音信号的
plt.figure()
plt.plot(Freq, fft_signal, color='blue')
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude')
plt.show()