from lib2to3.pgen2.token import NT_OFFSET
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns; sns.set()
import librosa
import librosa.display
import IPython.display as ipd
import sys
from scipy import signal
# 读取音频wav文件       
audio_path = r"C:\Users\Richard\Desktop\COM3502代码\chinese_song.wav"
s,sr = librosa.load(audio_path, sr=None, mono=True)
ipd.Audio(s,rate=sr)

figure = plt.figure(figsize=(12,6))
LDFT = 512
plt.subplot(1,2,1)
plt.specgram(s,Fs=sr,NFFT=LDFT) # NFFT default:256
plt.title('spectrogram with DFT length of ' + str(LDFT) + ' (high time res.)')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')
plt.colorbar(label='dB')
plt.clim(-150,0)

LDFT=8192

plt.subplot(1,2,2)
plt.specgram(s, Fs=sr, NFFT=LDFT); # NFFT default: 256
plt.title('spectrogram with DFT length of ' + str(LDFT) + ' (high freq. res.)')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')
plt.grid(None)  # no grid (in case you used seaborn)

plt.colorbar(label='dB');
plt.clim(-150,0)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
def plot_tolerance_scheme(Wp=0.25,Ws=0.3,Rp_lin=0.9,Rs_lin=0.1):
    """
    Plots a tolerance scheme for a (low-pass) filter design.
    
    Parameters
    ----------
    Wp : float, optional (but recommended)
        pass limit frequency (normalised) $W_p = \Omega / \pi$
    Ws : float, optional (but recommended)
        stop limit frequency (normalised)
    Rp_lin : float, optional (but recommended)
        allowed ripple range in pass band (linear), default 0.9
    Rs_lin : float, optional (but recommended)
        allowed ripple range in pass band (linear), default 0.1

    Example use:
    -------
    signal
        plot_tolerance_scheme(Wp=0.25,Ws=0.3,Rp_lin=0.9,Rs_lin=0.1)
    """
    dh1x=[0,Ws];  dh1y=[1,1];            # (x,y) coordinates of lines
    dh2x=[0,Wp];  dh2y=[Rp_lin,Rp_lin]; 
    dv2x=[Wp,Wp]; dv2y=[0,Rp_lin];   
    sh1x=[Ws,1];  sh1y=[Rs_lin,Rs_lin]; 
    sh2x=[Wp,1];  sh2y=[0,0]; 
    svx=[Ws,Ws];  svy=[Rs_lin,1];  
    # plot the actual lines
    plt.plot(dh1x,dh1y,'k--',dh2x,dh2y,'k--',dv2x,dv2y,'k--',sh1x,sh1y,'k--',
             sh2x,sh2y,'k--',svx,svy,'k--');
    plt.xlabel('Frequency $\Omega/\pi$');




Wp=0.25;    # passband edge frequency 
Ws=0.3;     # stopband edge frequency 
Rp_lin=0.9; # allowed ripples in the pass band area
Rs_lin=0.1; # allowed ripples in the stop band area

plot_tolerance_scheme(Wp,Ws,Rp_lin,Rs_lin)




Wp = 2*(2000*(1-0.05))/sr # ?
Ws = 1-Wp
Rp_lin = 0.95
Rs_lin = 0.05
plot_tolerance_scheme(Wp,Ws,Rp_lin,Rs_lin)



def zplane(z, p, title='Poles and Zeros'):
    "Plots zeros and poles in the complex z-plane"
    ax = plt.gca()

    ax.plot(np.real(z), np.imag(z), 'bo', fillstyle='none', ms=10)
    ax.plot(np.real(p), np.imag(p), 'rx', fillstyle='none', ms=10)
    unit_circle = plt.Circle((0, 0), radius=1, fill=False,
                             color='black', ls='--', alpha=0.9)
    ax.add_patch(unit_circle)

    plt.title(title)
    plt.xlabel('Re{$z$}')
    plt.ylabel('Im{$z$}')
    plt.axis('equal')

poles = [0.5-0.5j,0.5+0.5j]
zeros = [-0.9,0.8]

zplane(zeros,poles)



Wp = 0.25
Ws = 0.3
Rp_lin = 0.9
Rs_lin = 0.1

Rp = -20*np.log10(Rp_lin)
Rs = -20*np.log10(Rs_lin)

N,Wn = signal.buttord(Wp,Ws,Rp,Rs)

b,a = signal.butter(N,Wn,'low')

f,h = signal.freqz(b,a)
omega = np.linspace(0,1,len(f))

plot_tolerance_scheme(Wp,Ws,Rp_lin,Rs_lin)
plt.plot([Wn,Wn],[0,1],color='red',ls=':',label='cutoff frequency')
plt.plot(omega,abs(h),lw = 2,label='Butterworth low-pass')
plt.title('Butterworth low-pass filter of order ' + str(N))
plt.ylabel('Amplitude $|h(e^{j \Omega})|$')
plt.xlabel('Frequency $\Omega/\pi$')
plt.legend()



# plot zeros and poles in the z plane
zplane(-1, np.roots(a))
plt.text(-0.95,0.1,str(N));


Wp = 0.475
Ws = 0.525
Rp_lin = 0.95
Rs_lin = 0.05

Rp = -20*np.log10(Rp_lin)
Rs = -20*np.log10(Rs_lin)

N,Wn = signal.buttord(Wp,Ws,Rp,Rs)

b,a = signal.butter(N,Wn,'low')

f,h = signal.freqz(b,a)
omega = np.linspace(0,1,len(f))

plot_tolerance_scheme(Wp,Ws,Rp_lin,Rs_lin)
plt.plot([Wn,Wn],[0,1],color='red',ls=':',label='cutoff frequency')
plt.plot(omega,abs(h),lw = 2,label='Butterworth low-pass')
plt.title('Butterworth low-pass filter of order ' + str(N))
plt.ylabel('Amplitude $|h(e^{j \Omega})|$')
plt.xlabel('Frequency $\Omega/\pi$')
plt.legend()

# plot zeros and poles in the z plane
zplane(-1, np.roots(a))
plt.text(-0.95,0.1,str(N));



from scipy import signal
s_filtered = signal.filtfilt(b,a,s)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1) # 过滤前
plt.plot(s,label = "$s[k]$ (before filtering)")
plt.plot(s_filtered,label = 'after filtering')
plt.legend()

plt.subplot(1,2,2)
start_idx = 16000
end_idx = start_idx+1500
plt.plot(np.arange(start_idx,end_idx),s[start_idx:end_idx],label = "$s[k]$ (before filtering)")
plt.plot(np.arange(start_idx,end_idx),s_filtered[start_idx:end_idx],label="after filtering")
plt.title('Cut-out showing a harmonic (voiced) piece and a noisy (unvoiced) one')
plt.legend()
# ipd.Audio(s,rate=sr)
ipd.Audio(s_filtered,rate=sr)




fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.specgram(s,Fs=sr)
plt.title('spectrogram of signal} before filtering')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')
plt.clim(-180,-30)
plt.colorbar(label ='dB')

plt.subplot(1,2,2)
plt.specgram(s_filtered,Fs=sr)
plt.title('spectrogram of signal after filtering')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')
plt.clim(-180,-30)
plt.colorbar(label ='dB')
plt.tight_layout()


Wp = 0.25
Ws = 0.3
Rp_lin = 0.9
Rs_lin = 0.1

# Rp = -20*np.log10(Rp_lin)
# Rs = -20*np.log10(Rs_lin)

N,Wn = signal.cheb1ord(Wp,Ws,Rp,Rs)
b,a = signal.cheby1(N,Rp,Wn,'low')
h = np.abs(np.fft.fft(b,1024))/np.abs(np.fft.fft(a,1024))
h = h[0:513] # only show first half (positive frequencies
omega = np.linspace(0,1,513)

# plot_tolerance_scheme(Wp,Ws,Rp_lin,Rs_lin)
plt.figure(figsize=(12,12))

plt.plot(omega,abs(h),lw = 2,label='Butterworth low-pass')
plt.title('Chebyshev type II low-pass filter of order ' + str(N))


s_filtered = signal.filtfilt(b,a,s)
fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.specgram(s,Fs=sr)
plt.title('spectrogram of signal before filtering')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')
plt.colorbar(label='dB')
plt.clim(-180,-30)

plt.subplot(1,2,2)
plt.specgram(s_filtered,Fs=sr)
plt.title('spectrogram of signal before filtering')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')
plt.colorbar(label='dB')
plt.clim(-180,-30)
plt.tight_layout()

Wp = 0.25
Ws = 0.3
Rp_lin = 0.9
Rs_lin = 0.1
N,Wn = signal.ellipord(Wp,Ws,Rp,Rs)
b, a = signal.ellip(N,Rp,Rs,Wn,'low')

h = np.abs(np.fft.fft(b,1024))/np.abs(np.fft.fft(a,1024))
h = h[0:513]
omega = np.linspace(0,1,513)

plot_tolerance_scheme()
plt.plot(omega,abs(h),lw = 2)
plt.title('Cauer (elliptical) low-pass filter of order ' + str(N))
plt.ylabel('Amplitude $|h(e^{j \Omega})|$')


b, a = signal.ellip(N,Rp,Rs,Wn,'low')
zplane(np.roots(b), np.roots(a))

Wp = [0.25, 0.5]       # pass-band frequency limits (normalised)
Ws = [0.2,  0.6]       # stop-band frequency limits (normalised)
Rp = 1                 # we allow 1 dB ripple in pass-band
Rs = 40                # we's like to have 40dB attenuation
N, Wn = signal.buttord(Wp, Ws, Rp, Rs)
b, a = signal.butter(N, Wn, 'band')
f,h=signal.freqz(b,a)
omega=np.linspace(0,1,len(f))
plt.plot(omega, 20*np.log10(np.abs(h)))
plt.grid(True,which='both', axis='both')
plt.ylabel('Amplitude $|h(e^{j \Omega})|$ in dB')
plt.xlabel('Frequency $\Omega / \pi$')
plt.title('Butterworth bandpass filter')

plt.fill([0,     Ws[0],  Ws[0],   0],    [-Rs, -Rs, 2, 2], '0.7', lw=0) # stop
plt.fill([Wp[0], Wp[0],  Wp[1],  Wp[1]], [-100, -Rp, -Rp, -100], '0.7', lw=0) # pass
plt.fill([Ws[1], Ws[1], 1, 1],           [2, -Rs, -Rs, 2], '0.7', lw=0) # stop
plt.axis([0, 1, -60, 2]);



Wp = [0.25, 0.5]       # pass-band frequency limits (normalised)
Ws = [0.2,  0.6]       # stop-band frequency limits (normalised)
Rp = 1                 # we allow 1 dB ripple in pass-band
Rs = 40                # we's like to have 40dB attenuation
Rs_lin= 10**(-Rs/20)   # transforming dB back to linear
Rp_lin= 10**(-Rp/20)   # transforming dB back to linear

# determine necessary filter order as well as cut-off frequencies
N, Wn = signal.buttord(Wp, Ws, Rp, Rs)
print('The minimum possible filter order to fulfil the tolerance scheme is '+str(N)+'.')
print('The 1st cut-off frequency which will be {:.2f}.'.format(Wn[0])) 
print('The 2nd cut-off frequency which will be {:.2f}.'.format(Wn[1])) 



b,a = signal.butter(N,Wn,'band')
f,h=signal.freqz(b,a)
omega=np.linspace(0,1,len(f))
# plot filter
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(omega, np.abs(h))
#plt.title('Butterworth bandpass filter fit to constraints')
plt.ylabel('Amplitude $|h(e^{j \Omega})|$')
plt.xlabel('Frequency $\Omega / \pi$')
plt.grid(True,which='both', axis='both')

plt.fill([0,     Ws[0], Ws[0], 0],     [Rs_lin, Rs_lin, 1.1, 1.1], '0.7', lw=0) # stop
plt.fill([Wp[0], Wp[0], Wp[1], Wp[1]], [-.1, Rp_lin, Rp_lin, -.1], '0.7', lw=0) # pass
plt.fill([Ws[1], Ws[1], 1, 1],         [1.1, Rs_lin, Rs_lin, 1.1], '0.7', lw=0) # stop
plt.axis([0, 1, -0.1, 1.1]);

# plot filter again (this time in dB)
plt.subplot(1,3,2)
plt.plot([Wn[0],Wn[0]],[-100,2],color='r',ls=':',label='cutoff frequency1')
plt.plot([Wn[1],Wn[1]],[-100,2],color='g',ls=':',label='cutoff frequency2')
plt.plot(omega, 20*np.log10(np.abs(h)),label='filter transfer function')
plt.ylabel('Amplitude $|h(e^{j \Omega})|$ in dB')
plt.xlabel('Frequency $\Omega / \pi$')
plt.legend(loc='lower right')
plt.fill([0,     Ws[0],  Ws[0],   0],    [-Rs, -Rs, 2, 2],       '0.7', lw=0) # stop
plt.fill([Wp[0], Wp[0],  Wp[1],  Wp[1]], [-100, -Rp, -Rp, -100], '0.7', lw=0) # pass
plt.fill([Ws[1], Ws[1], 1, 1],           [2, -Rs, -Rs, 2],       '0.7', lw=0) # stop
plt.axis([0, 1, -70, 2]);

# plot zeros and poles in the z plane
plt.subplot(1,3,3)
zplane(np.roots(b), np.roots(a))

plt.tight_layout()

ipd.Audio(s, rate=sr)
ipd.Audio(s_filtered, rate=sr)

























