from cProfile import label
from distutils.command.build_ext import build_ext
from lib2to3.pgen2.token import NT_OFFSET
from operator import ge
from turtle import color, right
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns; sns.set()
import IPython.display as ipd
import librosa
import librosa.display

def get_sine_wave(frequency_hz, length_s=1, sample_rate_hz=8000):
    """
    Return a sine wave with given parameters.
    
    Parameters
    ----------
    frequency_hz : float
        frequency $f$ of the sinus to be generated 
    length_s : float, optional
        length of the sine signal to be generated, default: 1 sec.
    sample_rate_hz : float, optional
        sampling frequency $f_s$, default: 8000 Hz

    Returns
    -------
    signal
        generated sinus signal
    """
    # Task1 创建时间向量$t$
    time_points = np.linspace(0,length_s,
                                int(sample_rate_hz*length_s),
                                endpoint=False)
    # 返回频率为$f$的正弦波，即可变频率_hz
    return np.sin(2*np.pi*frequency_hz*time_points)

# 建立一个样本频率为8000Hz，时间长度为2s，音高为440Hz的正弦波信号
concert_pitch = get_sine_wave(440,2,8000)

# Task2 取这个正弦波信号的一部分，30ms（240/8000=0.03s）
# plt.plot(concert_pitch[:240])
# plt.xlabel('sample $k$')
# plt.ylabel('sine of $440$ Hz')
# plt.title('First 240 samples of sine of 440 Hz')
# plt.show()

# 聆听这个正弦波信号
ipd.Audio(concert_pitch, rate=8000, normalize=False)

# Task3 串联正弦波信号
half_concert_pitch = get_sine_wave(220,2,8000)
# plt.plot(half_concert_pitch[:160],color="red",label="sin with $f=200$ Hz")
# plt.plot(concert_pitch[:160],color="green",label="sin with $f=400$ Hz")
# plt.xlabel("samples $k$")
# plt.legend(loc="upper right") # 创建一个显示线条标签的图例

alternation = np.concatenate((half_concert_pitch, concert_pitch, half_concert_pitch, concert_pitch))
ipd.Audio(alternation,rate=8000)

# plt.plot(half_concert_pitch[:160],color="red",label="sin with $f=200$ Hz",alpha = 0.3)
# plt.plot(concert_pitch[:160],color="green",label="sin with $f=400$ Hz",alpha = 0.3)
# plt.plot(half_concert_pitch[0:160]+concert_pitch[0:160])
# plt.show()

# Task4 叠加、可视化并播放两个正弦信号的叠加（加法）
signal_1 = get_sine_wave(440,2,8000)
signal_2 = get_sine_wave(400,2,8000)
samples = 800

# plt.plot(signal_1[0:samples],color = "red",alpha = 0.3)
# plt.plot(signal_2[0:samples],color = "green",alpha = 0.3)
# plt.plot(signal_1[0:samples]+signal_2[0:samples])

ipd.Audio(signal_1+signal_2,rate=8000)

# Task5 根据之前创建的正弦波信号创建一个f = 2.5Hz 时间长度为1s的阻尼波
fs = 8000
t = np.arange(0,1,1/fs) # 时间向量的长度为1秒
f = 2.5               # 频率为2.5Hz
alpha = 1            # 破坏因素（玩一玩，看看会发生什么）。

sin = np.sin(2*np.pi*f*t)
damp = np.exp(-alpha*t)
x = sin * damp 

# plt.subplot(2,1,1)
# plt.plot(t,sin)
# plt.ylabel('sin($t$)')

# plt.subplot(2,1,2)
# plt.plot(t,x, label='sin$(2 \pi \cdot ' + str(f) + ' \mathrm{Hz} \cdot t) \cdot \mathrm{e}^{- a t}$')
# plt.plot(t,damp, '--', label='$\mathrm{e}^{- a t}$')
# plt.legend()
# plt.xlabel('$t$ in seconds')
# plt.show()

concert_pitch_lsec = get_sine_wave(440,1,8000)
half_concert_pitch_lsec = get_sine_wave(220,1,8000)

alternation_lsec = np.concatenate((half_concert_pitch_lsec,concert_pitch_lsec,half_concert_pitch_lsec,concert_pitch_lsec))

# 输出图像
# plt.plot(alternation_lsec)
ipd.Audio(alternation_lsec,rate=8000)

alternation_damp = np.concatenate((half_concert_pitch_lsec*damp,concert_pitch_lsec*damp,half_concert_pitch_lsec*damp,concert_pitch_lsec*damp))
# plt.plot(alternation_damp)
ipd.Audio(alternation_damp,rate=8000)
# plt.show()

# Task6-Task7

# 创建音符
# g = get_sine_wave(196.00)
# a = get_sine_wave(220.00)
# b = get_sine_wave(246.94)
# c = get_sine_wave(261.63)
# d = get_sine_wave(293.66)
# e = get_sine_wave(329.63)

# 一起应用，形成曲子+和弦 
# tune = [b,d,a,g,a,b,d,(g+b+d)]
# tune = np.concatenate(tune)

# 输出曲子图像
# plt.plot(tune)
# plt.show()

# 播放这个曲子
# ipd.Audio(tune,rate=8000)

# 创造受阻的音符 
g = get_sine_wave(196.00)*damp
a = get_sine_wave(220.00)*damp
b = get_sine_wave(246.94)*damp
c = get_sine_wave(261.63)*damp
d = get_sine_wave(293.66)*damp
e = get_sine_wave(329.63)*damp

#一起应用，形成曲子+和弦 
tune = [b,d,a,g,a,b,d,(g+b+d)]
tune = np.concatenate(tune)

#输出曲子图像
# plt.plot(tune)
# plt.show()

# 播放这个曲子
ipd.Audio(tune,rate=8000)

# 右手边的音符
right_hand_notes=   [("E5", 1 / 16),
    ("D#5", 1 / 16),
    ("E5", 1 / 16),
    ("D#5", 1 / 16),
    ("E5", 1 / 16),
    ("B4", 1 / 16),
    ("D5", 1 / 16),
    ("C5", 1 / 16),
    ("A4", 1 / 8),
    ("Pause", 1 / 16),
    ("C4", 1 / 16),
    ("E4", 1 / 16),
    ("A4", 1 / 16),
    ("B4", 1 / 8),
    ("Pause", 1 / 16),
    ("E4", 1 / 16),
    ("G#4", 1 / 16),
    ("B4", 1 / 16),
    ("C4", 1 / 8),
    ("Pause", 1 / 16),
    ("E4", 1 / 16),
    ("E5", 1 / 16),
    ("D#5", 1 / 16),
    ("E5", 1 / 16),
    ("D#5", 1 / 16),
    ("E5", 1 / 16),
    ("B4", 1 / 16),
    ("D5", 1 / 16),
    ("C5", 1 / 16),
    ("A4", 1 / 8),
    ("Pause", 1 / 16),
    ("C4", 1 / 16),
    ("E4", 1 / 16),
    ("A4", 1 / 16),
    ("B4", 1 / 8),
    ("Pause", 1 / 16),
    ("E4", 1 / 16),
    ("C5", 1 / 16),
    ("B4", 1 / 16),
    ("A4", 1 / 4),
]

# 左手边的音符
left_hand_notes = [
    ("Pause", 1 / 8),
    ("Pause", 3 / 8),
    ("A2", 1 / 16),
    ("E3", 1 / 16),
    ("A3", 1 / 16),
    ("Pause", 3 / 16),
    ("E2", 1 / 16),
    ("E3", 1 / 16),
    ("G#3", 1 / 16),
    ("Pause", 3 / 16),
    ("A2", 1 / 16),
    ("E3", 1 / 16),
    ("B3", 1 / 16),
    ("Pause", 3 / 16),
    ("Pause", 3 / 8),
    ("A2", 1 / 16),
    ("E3", 1 / 16),
    ("A3", 1 / 16),
    ("Pause", 3 / 16),
    ("E2", 1 / 16),
    ("E3", 1 / 16),
    ("G#3", 1 / 16),
    ("Pause", 3 / 16),
    ("A2", 1 / 16),
    ("E3", 1 / 16),
    ("B3", 1 / 16),
    ("Pause", 1 / 16),
]

# 我们需要一个简单的附加功能来生成更多的进阶歌曲，就是一个生成沉默的函数。下面的函数get_silence生成了一个所需长度的0值数组。
def get_silence(length_s, sample_rate_hz):
    """Return silence for the given length at the given sample rate."""
    return np.zeros(int(length_s*sample_rate_hz))

# 现在我们可以定义一个函数，它可以创建所需的到或暂停，由上面对left_hand_notes和right_hand_notes的描述定义。
def create_tone(note,duration):
    
    tempo = 5
    if note == "Pause":
        return get_silence(length_s=duration*tempo, sample_rate_hz=8000)
    
    note_position = {
        "C": -9,
        "C#": -8,
        "D": -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        "G": -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,   
    }

    octave = int(note[-1])
    key = note[:-1]

    frequency_hz = 440*2**((note_position[key]/12)+(octave-4))

    return get_sine_wave(frequency_hz=frequency_hz, length_s=duration*tempo, sample_rate_hz=8000)

# 创建一个测试音，看看上面的函数是否按预期工作。
# test_tone = create_tone("A4",1/8)
# plt.plot(test_tone)
# plt.show()
# ipd.Audio(test_tone,rate=8000)

right_hand = np.concatenate(
    [create_tone(note, duration) for note, duration in right_hand_notes]
)

left_hand = np.concatenate(
    [create_tone(note, duration) for note, duration in left_hand_notes]
)

song = left_hand + right_hand
ipd.Audio(song, rate=8000)

# Download the wave file `https://staffwww.dcs.shef.ac.uk/people/S.Goetze/sound/music_44k.wav`
# TMD jupyter运行不了这个音频文件，我选择继续使用librosa
audio_path = r"C:\Users\Richard\music_44k.wav"
mus_44k, sr = librosa.load(audio_path, sr=None, mono=True)
# plt.plot(mus_44k)
# plt.show()

# 将样本从44.1kHz 缩减至 8KHz
mus_8k = librosa.resample(mus_44k,sr,8000) # 缩减至8kHz

# 将样本从8kHz 复原至 44.1KHz
mus_44k_2 = librosa.resample(mus_8k,8000,sr)

# 可视化
figure = plt.figure(figsize=(12,6)) # 创建一个尺寸为12x6英尺的图像

plt.subplot(1,2,1)
plt.specgram(mus_44k, Fs=sr)
plt.title('original')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')

plt.colorbar(label='dB')
plt.clim(-150,0)

plt.subplot(1,2,2)
plt.specgram(mus_44k_2, Fs=sr)
plt.title('after down-/upsampling')
plt.xlabel('time $t$')
plt.ylabel('frequency $f$')

plt.colorbar(label='dB')
plt.clim(-150,0)

ipd.Audio(mus_44k,rate=sr)
ipd.Audio(mus_44k_2,rate=sr)    
plt.show()






