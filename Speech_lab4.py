from cProfile import label
from distutils.command.build_ext import build_ext
from lib2to3.pgen2.token import NT_OFFSET
from operator import ge
from signal import signal
from turtle import color, right
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns; sns.set()
import IPython.display as ipd
import librosa
import librosa.display
from scipy import signal

# t = np.linspace(0, 2, 500)
# saw_tooth = 2*(t-np.floor(t))-1

# f0 = 1 # frequency in Hz for scipy samtooth
# saw_tooth2 = signal.sawtooth(2 * np.pi * f0 * t)

# plt.plot(t, saw_tooth, label='$2$ ($t$ - floor($t$)) - $1$')
# plt.plot(t, saw_tooth2, '--', label='scipy sawtooth');
# plt.xlabel('time $t$ in seconds'); plt.ylabel('$x(t)$')
# plt.legend();


# fs = 8000                  # sampling frequency
# t = np.arange(0,2,1/fs)  # time vector

# f0 = 200                  # frequency in 200 Hz for scipy sawtooth
# saw_tooth = signal.sawtooth(2*np.pi*f0*t)

# # plot first 20 ms (=160 samples at sampling frequency of 8000 Hz)
# plt.subplot(1,2,1)
# plt.plot(t[0:160], saw_tooth[0:160], '--', label='scipy sawtooth');
# plt.xlabel('time $t$ in seconds'); plt.ylabel('$x(t)$')
# plt.legend();

# # calculate the spectum (frequency domain representation)
# FFT_length = 2**15 # take a power of two which is larger than the signal length
# f = np.linspace(0, fs/2, num=int(FFT_length/2+1))
# spectrum = np.abs(np.fft.rfft(saw_tooth,n=FFT_length))

# # plot the spectrum
# plt.subplot(1,2,2)
# plt.plot(f,spectrum)
# plt.xlabel('frequency $f$ in Hz');plt.ylabel('$x(f)$')

# plt.tight_layout() # this allowes for some space for the title text.

# # playback sound file (if you want)
# ipd.Audio(saw_tooth, rate=fs)

# fs = 8000 # Sampling frequency
# t = np.arange(0,2,1/fs) # time vector 2
# f0 = 2                 # fundamental frequency in Hz

# sin1 = np.sin(2*np.pi*f0*t)
# sin2 = np.sin(2*np.pi*2*f0*t)/2
# sin3 = np.sin(2*np.pi*3*f0*t)/3
# sin4 = np.sin(2*np.pi*4*f0*t)/4

# plt.figure(figsize=(8,6))
# plt.subplot(4,2,1)
# plt.plot(t,sin1,label='$\mathrm{sin}(\omega_0 t$)')
# plt.ylabel('$x_1(t)$')
# plt.legend() 

# plt.subplot(4,2,3)
# plt.plot(t,sin2,label='$\mathrm{sin}(2 \omega_0 t$)/2')
# plt.ylabel('$x_2(t)$')
# plt.legend()

# plt.subplot(4,2,4)
# plt.plot(t,sin1+sin2,label='$x_1(t)+x_2(t)$')
# plt.legend()

# plt.subplot(4,2,5)
# plt.plot(t,sin3,label='$\mathrm{sin}(3 \omega_0 t$)/3')
# plt.ylabel('$x_3(t)$')
# plt.legend()

# plt.subplot(4,2,6)
# plt.plot(t,sin1+sin2+sin3,label='$x_1(t)+x_2(t)+x_3(t)$')
# plt.legend()

# plt.subplot(4,2,7)
# plt.plot(t,sin4,label='$\mathrm{sin}(4 \omega_0 t$)/4')
# plt.ylabel('$x_4(t)$')
# plt.xlabel('time $t$ in seconds')
# plt.legend()

# plt.subplot(4,2,8)
# plt.plot(t,sin1+sin2+sin3+sin4,label='$x_1(t)+x_2(t)+x_3(t)+x_4(t)$')
# plt.xlabel('time $t$ in seconds')
# plt.legend()

# plt.tight_layout()



def generateSawTooth(f0=2, length = 2, fs=8000, order=10, height=1):
    """
    Return a saw-tooth signal with given parameters.
    
    Parameters
    ----------
    f0 : float, optional
        fundamental frequency $f_0$ of the signal to be generated,
        default: 1 Hz
    length : float, optional
        length of the signal to be generated, default: 2 sec.
    fs : float, optional
        sampling frequency $f_s$, default: 8000 Hz
    order : int, optional
        number of sinosuids to approximate saw-tooth, default: 10
    height : float, optional
        height of saw-tooth, default: 1

    Returns
    -------
    sawTooth
        generated sawtooth signal
    t
        matching time vector
    """
        
    t=np.arange(0,length,1/fs) # time vector
    sum = np.zeros(len(t))
    for ii in range(order):
        jj=ii+1
        sum += np.sin(2*np.pi*jj*f0*t)/jj
    return 2*height*sum/np.pi, t

saw,t = generateSawTooth(order=100)
plt.plot(t,saw)
plt.xlabel('time $t$ in seconds')
plt.show()