import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

Fs=1000;T=1/Fs

wp=200*np.pi/500
wst=300*np.pi/500

Rp=1
As=25

omiga_p=(2/T)*np.tan(wp/2)
omiga_st=(2/T)*np.tan(wst/2)

N, Wn=signal.buttord(omiga_p, omiga_st, Rp, As, True)
filts = signal.lti(*signal.butter(N, Wn, btype='lowpass',
                                  analog=True))
filtz = signal.lti(*signal.bilinear(filts.num, filts.den, Fs))
wz, hz = signal.freqz(filtz.num, filtz.den)
ws, hs = signal.freqs(filts.num, filts.den, worN=Fs*wz)
plt.semilogx(wz*Fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
             label=r'$|H_z(e^{j \omega})|$')
plt.semilogx(wz*Fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
             label=r'$|H(j \omega)|$')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.show()
plt.grid()