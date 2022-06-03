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
# Task1 读取音频并且画出正弦波信号
audio_path = r"C:\Users\Richard\speech_8kHz_murder.wav"
s,fs = librosa.load(audio_path,sr=None,mono=True)

print('File '+audio_path+'loaded. Its sampling rate is '+ str(fs)+' Hz')

# listen to the sound file (if you want)
# ipd.Audio(s,rate=fs)

# start_sample = int(10*fs) # start at 10 sec
# number_of_samples = 4096 # the number of samples we want to cut
# end_sample = start_sample + number_of_samples # last sample to be cut out
# sample_vec = np.linspace(start_sample,end_sample,number_of_samples) # vector of sample
# x1 = s[start_sample:end_sample] # do a slice operation

# plt.figure(figsize=(8,5))
# plt.subplot(2,1,1)
# plt.plot(np.arange(0,len(s)),s)
# plt.ylabel('$x_1[k]$')
# plt.plot(sample_vec,x1,'r')
# plt.subplot(2,1,2)
# plt.plot(sample_vec,x1,'r')
# plt.xlabel('$k$')
# plt.ylabel('$x_1$['+str(start_sample)+'...'+str(end_sample)+']')
# plt.title('$x_1[k]$ for '+str(len(x1))+' samples between '+str(start_sample)+' to '+str(end_sample)+' ($f_s$='+str(fs)+')')
# plt.tight_layout()
# ipd.Audio(x1,rate=fs)


# Task2
# lets cut out a piece of the data
# Lw = 512
# Lov = 1
# Lhop = int(np.round(Lw/Lov))

# 为子图创建轴的网格
# plt.figure(figsize=(12,6))
# ax1 = plt.subplot2grid(shape=(2,5),loc=(0,0),colspan=5)
# ax2 = plt.subplot2grid(shape=(2,5),loc=(1,0),colspan=1)
# ax3 = plt.subplot2grid(shape=(2,5),loc=(1,1),colspan=1)
# ax4 = plt.subplot2grid(shape=(2,5),loc=(1,2),colspan=1)
# ax5 = plt.subplot2grid(shape=(2,5),loc=(1,3),colspan=1)
# ax6 = plt.subplot2grid(shape=(2,5),loc=(1,4),colspan=1)
# ax_blocks = [ax2,ax3,ax4,ax5,ax6]

# 在上面板上绘制信号部分（轴ax1）。
# ax1.plot(sample_vec,x1,'r')
# ax1.set_xlabel('$k$')
# ax1.set_ylabel('$x_1$['+str(start_sample)+'...'+str(end_sample)+']')
# ax1.set_title('A piece between sample '+str(start_sample)+' and '+str(end_sample)+'(of length '+str(len(x1))+')'+' from the 1st channel ($f_s$='+str(fs)+')')

# 在下面的面板上绘制单个区块 
# colors = ['g','y','m','b','c','k'] # 定义一个颜色的向量
# for ii,k in enumerate(range(start_sample,start_sample+5*Lhop,Lhop)):
#     block_k_vec = np.arange(k,k+Lw)
#     block_sig_vec = x1[ii*Lhop:ii*Lhop+Lw]
#     ax1.plot(block_k_vec,block_sig_vec,colors[ii])

#     ax_blocks[ii].plot(block_k_vec,block_sig_vec,colors[ii])
#     ax_blocks[ii].set_xlabel('$k$')
#     ax_blocks[ii].set_ylim(-0.35, 0.35)
#     ax_blocks[ii].set_title('Block ' + str(ii))


# 自动调整水平方向的padding 
# 以及垂直方向。
# plt.tight_layout()
# plt.show()

audio_path_1 = r"C:\Users\Richard\voiced_unvoiced_e.wav"
e,fs_e = librosa.load(audio_path_1,sr=None,mono=True)
print('File'+audio_path_1+'loaded. It has a samlping rate of f_s ='+str(fs_e)+'Hz')
audio_path_2 = r"C:\Users\Richard\voiced_unvoiced_z.wav"
z,fs_z = librosa.load(audio_path_2,sr=None,mono=True)
print('File'+audio_path_2+'loaded. It has a samlping rate of f_s ='+str(fs_z)+'Hz')
ipd.Audio(e,rate=fs_e)
ipd.Audio(z,rate=fs_z)

# 定义一个计算短期信号的函数
def calc_STE(signal, sampsPerFrame):
    nFrames = int(len(signal)/sampsPerFrame) # 不重叠的数量 
    E = np.zeros(nFrames)
    for frame in range (nFrames):
        startIdx = frame * sampsPerFrame
        stopIdx = startIdx + sampsPerFrame
        E[frame] = np.sum(signal[startIdx:stopIdx]**2)
    return E

# signal = e
# sampsPerFrame = int(0.02*fs_e) # 20ms

# 为语音文件e 创建子版块创建网格
# plt.figure(figsize=(12,6))
# plt.subplot(2,1,1)
# plt.plot(signal)
# plt.subplot(2,1,2)
# plt.plot(calc_STE(signal,sampsPerFrame))
# plt.title('(Short-term) Energy per block ($L_{\mathrm{B1}} ='+str(sampsPerFrame)+'$,which is '+ str(sampsPerFrame/fs_e*1000)+'ms @ $f_s='+str(fs_e)+'$)')
# plt.tight_layout()
# ipd.Audio(e,rate=fs_e)

# signal = z
# sampsPerFrame = int(0.02*fs_z) # 20ms

# 为语音文件z 创建子版块创建网格
# plt.figure(figsize=(12,6))
# plt.subplot(2,1,1)
# plt.plot(signal)
# plt.subplot(2,1,2)
# plt.plot(calc_STE(signal,sampsPerFrame))
# plt.title('(Short-term) Energy per block ($L_{\mathrm{B1}} ='+str(sampsPerFrame)+'$,which is '+ str(sampsPerFrame/fs_z*1000)+'ms @ $f_s='+str(fs_z)+'$)')
# plt.tight_layout()
# ipd.Audio(z,rate=(fs_z))

# signal = s
# sampsPerFrame = int(0.02*fs) # 20ms

# plot
# plt.figure(figsize=(12,6))
# plt.subplot(2,1,1)
# plt.plot(signal)
# plt.subplot(2,1,2)
# plt.plot(calc_STE(signal,sampsPerFrame))
# plt.title('(Short-term) Energy per block ($L_{\mathrm{B1}} ='+str(sampsPerFrame)+'$,which is '+ str(sampsPerFrame/fs*1000)+'ms @ $f_s='+str(fs)+'$)')
# plt.tight_layout()
# ipd.Audio(s,rate=(fs))



# 定义一个计算零交叉率的函数
def calc_ZCR(signal,sampsPerFrame):
    nFrames = int(len(signal)/sampsPerFrame)
    ZCR = np.zeros(nFrames)
    for frame in range (nFrames):
        startIdx = frame * sampsPerFrame
        stopIdx = startIdx + sampsPerFrame
        signalframe = signal[startIdx:stopIdx]
        for k in range (1,len(signalframe)):
            ZCR[frame] += 0.5*abs(np.sign(signalframe[k])-np.sign(signalframe[k-1]))
    return ZCR

signal = e
sampsPerFrame = int(0.02*fs_e) # 20ms

# plot
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(signal)
plt.subplot(2,1,2)
plt.plot(calc_ZCR(signal,sampsPerFrame))
plt.tight_layout()
ipd.Audio(e,rate=fs_e)

signal = z
sampsPerFrame = int(0.02*fs_z) # 20ms

# plot
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(signal)
plt.subplot(2,1,2)
plt.plot(calc_ZCR(signal,sampsPerFrame))
plt.tight_layout()
ipd.Audio(z,rate=fs_z)

signal = s
sampsPerFrame = int(0.02*fs) # 20ms

# plot
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(signal)
plt.subplot(2,1,2)
plt.plot(calc_ZCR(signal,sampsPerFrame))
plt.tight_layout()
ipd.Audio(s,rate=fs)
plt.show()



