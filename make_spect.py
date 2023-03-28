"""
这个就是语音处理
wav ---> 频谱

所以可以将这个改成生成数据的

1. 读取到视频, 将视频中的图像和音频进行分离。
2. 对图像进行人脸检测, 和关键点检测。
3. 对分离出来的音频的数据进行预处理和与视频图像进行对齐。

"""
import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256, fps=25, sample_rate=16000):
    """
    这里是控制最后的长度的
    可以通过控制hop_length来是的最后出来的数据的shape和帧数是对应的

    f: frame
    fps: 帧率
    1s有16000个因素
    1s有30帧
    第一帧的图像是对应着第一帧的因素
    最后一帧的图像是对应着最后一帧的因素
    Args:
        x:
        fft_length:
        hop_length:

    Returns:
    """

    # 这个是我写的计算shape的方法
    hop_length = int(sample_rate / fps)  # 这个可以说是固定的, 将视频转为25帧, 在获得音频的时候使用16000的采样率
    lenght = x.shape[0]
    shape = (lenght // hop_length + 1, fft_length)

    # 先在x的两边进行一个padding, padding的长度是fft_length//2, 就是窗口大小的1/2
    x = np.pad(x, int(fft_length//2), mode='reflect')

    # noverlap = fft_length - hop_length
    # shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length,
    #                       fft_length
    #                       )

    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])

    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)    
    
    
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
rootDir = './wavs'
# spectrogram directory
targetDir = './spmel'


dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName, subdir)))

    # 生成一个随机的算子
    prng = RandomState(int(subdir[1:]))

    for fileName in sorted(fileList):
        # Read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for model roubstness
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        # Compute spect
        D = pySTFT(wav).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)

        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)    
        # save spect
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)
