from lib2to3.pgen2.token import NT_OFFSET
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns; sns.set()
import librosa
import librosa.display
import IPython.display as ipd
import sys
# 时间规格
second_per_sample = 8000       # 每秒钟的样本
sample_per_second = 1/second_per_sample     # 每个样本的秒数
StopTime = 0.25             # 声音的终点
time = np.arange(0,StopTime,sample_per_second)         # 以秒为单位的时间向量

# 频率为250赫兹的正弦信号
f = 250                     # 赫兹的频率
x = np.sin(2*np.pi*f*time)    # 正弦波信号

# 信号x（t）可以通过matplotlib.pyplot库中的plot()命令来显示。
plt.plot(time,x)
plt.ylabel("$x(t)$")
plt.xlabel("time $t$ in second")
# plt.show()

ipd.Audio(x/10,rate=second_per_sample) # 将rate除以10可以确保播放时不会太大声

# -------------------------------------------------------------------------------

StopTime1 = 0.5         # 声音的终点1
time1 = np.arange(0,StopTime1,sample_per_second)  # 以秒为单位的时间向量1

f1 = 60                     # 赫兹的频率1
phi1 = np.pi/2             # 相位(在0和2π之间)
x1 = np.sin(2*np.pi*f1*time1+phi1) # 正弦波信号1

f2 = 200            # 赫兹的频率1
phi2 = 0           # 相位(在0和2π之间)
x2 = np.sin(2*np.pi*f2*time1+phi2) # 正弦波信号2

# 绘制出信号的时间图 （Task1）
fig = plt.figure(figsize=(12,5)) # 创建一个尺寸为12x5英寸的图形
plt.subplot(2,1,1)              # 在一个2x1的子图阵列中的第一个子图
plt.plot(time1,x1,lw=2,label='$x_1(t)$=sin($2 \cdot \pi \cdot ' + str(f1) + '$ Hz $\cdot t)+ $' + str(phi1))
plt.ylabel("$x_1(t)$");         # 我们可以使用LaTeX符号，例如LaTeX数学模式（使用$...$）。
plt.legend()                   # 将标签信息显示为图例     
plt.subplot(2,1,2)            # 在一个2x1的子图阵列中的第二个子图
plt.plot(time1,x2,'r',lw=2, label='$x_2(t)$=sin($2 \cdot \pi \cdot ' + str(f2) + '$ Hz $\cdot t) + $' + str(phi2))
plt.ylabel('$x_2(t)$');
plt.legend()
plt.xlabel('time $t$ in seconds');
# plt.show()

ipd.Audio(x1/10,rate=second_per_sample) # 将rate除以10可以确保播放时不会太大声
ipd.Audio(x2/10,rate=second_per_sample) # 将rate除以10可以确保播放时不会太大声

#--------------------------------------------------------------

# 添加之前已经创建的两个信号 x1，x2
x3 = x1 + x2

# 绘制出信号的时间图（Task2）
new_fig = plt.figure(figsize=(12,8))        # 创建一个尺寸为12x8英寸的图形
plt.subplot(3,1,1)                         # 在一个3x1的子图阵列中的第一个子图
plt.plot(time1,x1,lw=2,label='$x_1(t)$=sin($2 \cdot \pi \cdot ' + str(f1) + '$ Hz $\cdot t)+ $' + str(phi1))
plt.ylabel("$x_1(t)$");                    # 我们可以使用LaTeX符号，例如LaTeX数学模式（使用$...$）。
plt.xlabel("time $t$ in seconds");
plt.legend()                             # 将标签信息显示为图例
plt.subplot(3,1,2)                      # 在一个3x1的子图阵列中的第二个子图
plt.plot(time1,x2,'r',lw=2, label='$x_2(t)$=sin($2 \cdot \pi \cdot ' + str(f2) + '$ Hz $\cdot t) + $' + str(phi2))
plt.ylabel("$x_2(t)$")
plt.xlabel("time $t$ in seconds");
plt.legend()
plt.subplot(3,1,3)
plt.plot(time1,x3,lw=2,color='g',label='sin1+sin2')
plt.ylabel("$x_1(t)+x_2(t)$")
plt.xlabel("time $t$ in seconds");
plt.tight_layout()                    # 这个命令确保/试图使所有的标签文本都能很好地阅读
# plt.show()

ipd.Audio(x3/10,rate=second_per_sample) # 将rate除以10可以确保播放时不会太大声

#--------------------------------------------------------------

# 读取音频wav文件(Task3)       
audio_path = r"C:\Users\Richard\speech.wav"
s,sr = librosa.load(audio_path, sr=None, mono=True) # Lab上给的读取音频的方法不能使用，我选择将音频文件拷贝到本地文件夹"C:\Users\Richard\speech.wav"中，使用librosa方法读取

# # plt画图
plt.figure(figsize=(6,4))           # 创建一个尺寸为12x8英寸的图形
plt.plot(s)
plt.ylabel('$s[k]$');
plt.xlabel('sample $k$');

# 可选的，我们可以创建一个匹配的时间向量
plt.figure(figsize=(6,4))
time2 = np.linspace(0,len(s)/second_per_sample,len(s))
plt.plot(time2,s)
plt.xlabel('$t$ in seconds');
plt.ylabel('$s(t)$');

# plt.show()
ipd.Audio(s,rate=second_per_sample)

#-------------------------------------------------------------
# 以线性振幅和对数振幅分贝（dB）绘制信号图 (Task4)
plt.subplot(2,1,1)
plt.plot(s)
plt.ylabel('$s[k]$');

epsilon = 1e-10 # 非常小的数字，以避免log10(0)。
s_dB = 20*np.log10(np.abs(s)+epsilon)     

plt.subplot(2,1,2)
plt.plot(s_dB)
plt.ylabel('$s_{\mathrm{dB}}[k]$')
plt.xlabel('sample $k$')
plt.ylim((-40,0)); # 定义Y轴的哪一部分应该被 "放大"。
plt.show() 




