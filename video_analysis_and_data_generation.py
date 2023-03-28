"""
错误: No module named numba.decorators错误解决
参考博客: https://blog.csdn.net/July_Wander/article/details/106857289
解决方案: pip install numba==0.48.0 --user
"""
import cv2
import os
import tqdm
import cv2
import skvideo.io
import numpy as np
import sys
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import get_window
from face_align import Face_Align
import os
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState

class face_process():
    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )

        self.fa = Face_Align()

        self.size = 336

        self.mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
        self.min_level = np.exp(-100 / 20 * np.log(10))
        self.b, self.a = self.butter_highpass(30, 16000, order=5)
        self.prng = RandomState()

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def pySTFT(self, x, fft_length=1024, fps=25, frames=2860, sample_rate=16000):
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
            fft_length: 采样的窗口的长度
            hop_length: 步长

        Returns:
        """

        # 这个是我写的计算shape的方法
        hop_length = int(sample_rate / fps)  # 这个可以说是固定的, 将视频转为25帧, 在获得音频的时候使用16000的采样率

        # 可以直接就使用frames, 而不通过下面的计算, 使用下面的计算的话会除不尽
        # lenght = x.shape[0]
        # shape = (lenght // hop_length + 1, fft_length)

        shape = (frames, fft_length)

        # 先在x的两边进行一个padding, padding的长度是fft_length//2, 就是窗口大小的1/2
        x = np.pad(x, int(fft_length // 2), mode='reflect')

        # noverlap = fft_length - hop_length
        # shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length,
        #                       fft_length
        #                       )

        strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])

        result = np.lib.stride_tricks.as_strided(x,
                                                 shape=shape,
                                                 strides=strides)

        fft_window = get_window('hann', fft_length, fftbins=True)
        result = np.fft.rfft(fft_window * result, n=fft_length).T

        return np.abs(result)

    def __run__(self, video_file):
        videogen, video_profile_info = self.makevideo(video_file)
        face_images = []
        kps_list = []

        for img_RGB in tqdm.tqdm(videogen):
            # skvideo读取数据为RGB
            face_bbox = self.face_mask_google_mediapipe(self.face_detection, img_RGB)

            x1, y1, x2, y2 = face_bbox
            h, w = img_RGB.shape[:2]
            x1, y1, x2, y2 = self.bbox_expansion_rate(x1, y1, x2, y2, h, w, 0.8)
            face_image = img_RGB[y1:y2, x1:x2, :]

            face_image = cv2.resize(face_image, (self.size, self.size))

            # 保存关键点
            kps = self.fa.__run__(face_image)
            kps_list.append(kps[np.newaxis, :, :])

            # 绘制图像
            face_image = np.zeros(face_image.shape)
            face_image = self.fa.drawFace(kps, face_image)
            face_images.append(face_image)

        kps_np = np.concatenate(kps_list, axis=0)
        return face_images, video_profile_info, kps_np

    def face_mask_google_mediapipe(self,
                                   face_detection,
                                   image,  # cv2读取并转RGB的图像
                                   ):
        """
        Returns a list of images with mask on the face parts.
        """

        face_bbox = []
        h, w, c = image.shape

        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                x_min = int(
                    detection.location_data.relative_bounding_box.xmin * w
                )
                y_min = int(
                    detection.location_data.relative_bounding_box.ymin * h
                )
                width = int(
                    detection.location_data.relative_bounding_box.width * w
                )
                height = int(
                    detection.location_data.relative_bounding_box.height * h
                )

                x1 = max(0, x_min)
                y1 = max(0, y_min)
                x2 = min(w, x_min + width)
                y2 = min(w, y_min + height)
                face_bbox.append([x1, y1, x2, y2])

        if len(face_bbox) > 0:
            return face_bbox[0]
        else:
            return []


    def bbox_expansion_rate(self, x1, y1, x2, y2, h, w, rate):
        """
        对边框进行外扩, 尽量要漏出头顶
        """
        b_h = y2 - y1
        b_w = x2 - x1

        y1 = max(0, int(y1 - rate * (2 / 3) * b_h))
        y2 = min(h, int(y2 + rate * (1 / 3) * b_h))
        x1 = max(0, int(x1 - rate * (1 / 2) * b_w))
        x2 = min(w, int(x2 + rate * (1 / 2) * b_w))

        return x1, y1, x2, y2

    def makevideo(self, video_file):
        """
        cv2 获取视频基本信息
        skvideo 读取视频每一帧(注意: 不会漏帧)

        :param video_file:
        :param _model:
        :return:
        """
        capture = cv2.VideoCapture(video_file)
        # 记录下当前视频中的info
        video_profile_info = {
            "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "channel": capture.get(cv2.CAP_PROP_CHANNEL),
            "fps": capture.get(cv2.CAP_PROP_FPS),  # frames/s
            "num_frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        }
        capture.release()

        videogen = skvideo.io.vreader(video_file)

        return videogen, video_profile_info

    def audioprocess(self, wav_path, fps, frames, sample_rate):
        x, fs = sf.read(wav_path)
        # Remove drifting noise
        y = signal.filtfilt(self.b, self.a, x)
        # Ddd a little random noise for model roubstness
        wav = y * 0.96 + (self.prng.rand(y.shape[0]) - 0.5) * 1e-06
        # Compute spect
        D = self.pySTFT(wav, fps=fps, frames=frames, sample_rate=sample_rate).T
        # Convert to mel and normalize
        D_mel = np.dot(D, self.mel_basis)
        D_db = 20 * np.log10(np.maximum(self.min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)

        # (2860, 80)
        return S


if __name__ == "__main__":
    video_data_dir_list = [
        "E:/datasets/audio2face/cctv_short_video_bilibili_small/Av416828403-P1.mp4",
    ]

    generate_wavs_dir = "generate_wavs"
    if not os.path.exists(generate_wavs_dir):
        os.makedirs(generate_wavs_dir)

    generate_lm_dir = "generate_lm"
    if not os.path.exists(generate_lm_dir):
        os.makedirs(generate_lm_dir)

    generate_audio_dir = "generate_audio"
    if not os.path.exists(generate_audio_dir):
        os.makedirs(generate_audio_dir)

    generate_lm_np_dir = "generate_lm_np"
    if not os.path.exists(generate_lm_np_dir):
        os.makedirs(generate_lm_np_dir)

    face_process = face_process()

    for video_path in video_data_dir_list:

        # 得到文件名
        abs_prefix, suffix = os.path.splitext(video_path)
        prefix, _ = os.path.splitext(os.path.split(video_path)[-1])

        # 给sub lm创建一个子文件夹
        sub_generate_lm_dir = os.path.join(generate_lm_dir, prefix)
        if not os.path.exists(sub_generate_lm_dir):
            os.makedirs(sub_generate_lm_dir)

        # 得到关键点的数据
        face_images, video_profile_info, kps_np = face_process.__run__(video_path)
        for i, image in enumerate(face_images):
            dst_img_path = os.path.join(sub_generate_lm_dir, str(i).zfill(6) + "_" + prefix + ".png")
            image = np.array(image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_img_path, image)
        kps_np_file = os.path.join(generate_lm_np_dir, prefix + ".npy")
        np.save(kps_np_file, kps_np.astype(np.float32), allow_pickle=False)


        # 得到音频数据
        sample_rate = 16000
        dst_wav_file = os.path.join(generate_wavs_dir, prefix + ".wav")
        os.system("ffmpeg -i {} -f wav -ar {} -ac 1 {} -y".format(video_path, str(sample_rate), dst_wav_file))
        fps = int(video_profile_info['fps'])
        frames = int(video_profile_info['num_frames'])
        audio_data = face_process.audioprocess(dst_wav_file, fps, frames, sample_rate)
        audio_file = os.path.join(generate_audio_dir, prefix + ".npy")
        np.save(audio_file, audio_data.astype(np.float32), allow_pickle=False)