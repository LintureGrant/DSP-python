from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
# 产生一个测试信号，振幅为2的正弦波，其频率在3kHZ缓慢调制，振幅以指数形式下降的白噪声
sampling_freq, audio = wavfile.read(r"C:\Users\Gtuff\Desktop\DSP-python\sound_Lnoise.wav")
T=20 #短时傅里叶变换的时长 单位 ms
fs = sampling_freq #采样频率是这么多，那么做出来的频谱宽度就是这么多
N = len(audio)  #采样点的个数
audio=audio*1.0/(max(abs(audio)))
print(N)

# 计算并绘制STFT的大小
f, t, Zxx = signal.stft(audio, fs, nperseg=T*fs/1000)

plt.pcolormesh(t, f, np.abs(Zxx), vmin = 0, vmax =0.1)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()