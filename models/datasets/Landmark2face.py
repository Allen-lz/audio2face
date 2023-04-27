# -*-coding:utf-8 -*-

"""
# File   : image_translation_dataset.py
# Time   : 2023/4/11 14:44
# Author : luzeng
# version: python 3.9
"""

import os
import glob2
from collections import namedtuple

import cv2
import skimage.io
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

class ImageTranslationDataset(data.Dataset):
    """
    Online landmark extraction with AWings
    Landmark setting: 98 landmarks
    """

    def __init__(self, data_dir, max_num_per_video=-1):
        super(ImageTranslationDataset, self).__init__()
        # 这是一个数据文件夹, 里面有多个视频子文件夹, 每个视频属于同一个人在连续的时间段说话视频
        self.data_dir = data_dir
        self.max_num_per_video = max_num_per_video
        self.input_size = (512, 512)

        # self.margin_t = 25
        self.margin_t = 20

        self.rotate_prob = 0.15

        self.face_mesh_predict_model = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            #refine_landmarks=True,
            min_detection_confidence=0.5)

        # 这是从mediapipe中提取的mesh点中对应的关键点
        self.face_68_landmarks_ind = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107,
            336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387,
            263, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87
        ]

        self.images = []
        # 得到每个的subdir中的图片
        for sub_dir in os.listdir(self.data_dir):
            image_paths = glob2.glob(os.path.join(self.data_dir, sub_dir, '*.jpg'))
            # 有的视频段的视频可能过多, 所以就取得前max_num_per_video个画面的图片
            if max_num_per_video != -1:
                # 这里用的是extend而不是append, 所以不会增加维度
                self.images.extend(image_paths[:self.max_num_per_video])
            else:
                self.images.extend(image_paths)

    def __len__(self):
        return len(self.images)

    def __down_face_mask(self, img, fl):
        """
        使用多个点points连接成了一个poly, poly框出一个连通域, 给这个连通域赋予权重
        """
        down_face_mask = np.zeros_like(img)
        # 下面的都是表述面部轮廓的点
        poly = np.array([
            fl[33],
            fl[4],
            fl[5],
            fl[6],
            fl[7],
            fl[8],
            fl[9],
            fl[10],
            fl[11],
            fl[12],
        ], dtype=np.int32)[:, :2]  # 不需要深度
        # 这里得到左上右下, 一个矩形的区域
        (min_x, min_y), (max_x, max_y) = poly.min(0), poly.max(0)
        # self.margin_t: 区域外扩的尺度
        # 将区域内的权重设置为1
        down_face_mask[max(min_y - self.margin_t, 0): min(max_y + self.margin_t + 1, img.shape[0]),
                       max(min_x - self.margin_t, 0): min(max_x + self.margin_t + 1, img.shape[1])] = 1.

        return down_face_mask

    def __down_teeth_mask(self, img, fl):
        down_teeth_mask = np.zeros_like(img)
        poly = np.array([fl[48, :2], fl[67, :2], fl[66, :2], fl[65, :2], fl[64, :2]], dtype=np.int32)
        cv2.fillPoly(down_teeth_mask, [poly], (1, 1, 1))
        return down_teeth_mask

    def __mouth_mask(self, img, fl):
        mouth_mask = np.zeros_like(img)
        poly = fl[48:68, :2].copy().astype(np.int32)
        (min_x, min_y), (max_x, max_y) = poly.min(0), poly.max(0)

        _margin = [max(min_y - self.margin_t, 0), min(max_y + self.margin_t + 1, img.shape[0]),
                   max(min_x - self.margin_t, 0), min(max_x + self.margin_t + 1, img.shape[1])]

        mouth_mask[_margin[0]: _margin[1], _margin[2]: _margin[3]] = 1.
        return mouth_mask, _margin

    def __getitem__(self, item):
        # 得到一张图片
        img_path = self.images[item]
        img = skimage.io.imread(img_path)

        # 对图片数据沿着水平方向进行分割
        src1, src2, _, dst = np.split(img, 4, axis=1)

        need_rotate = np.random.rand() < self.rotate_prob

        _margin = [0, dst.shape[0], 0, dst.shape[1]]

        # 数据增强, 对dst图片进行随机旋转
        if need_rotate:
            # 随机旋转20度以内的角度
            M = cv2.getRotationMatrix2D((dst.shape[0] // 2, dst.shape[1] // 2),
                                        np.random.rand() * 40 - 20,  # (-20, 20)
                                        1)
            dst = cv2.warpAffine(src=dst, M=M, dsize=(dst.shape[0], dst.shape[1]), borderValue=(0, 0, 0))

        # 人脸关键点检测
        src_predict_result = self.face_mesh_predict_model.process(src1)
        dst_predict_result = self.face_mesh_predict_model.process(dst)
        # 但凡src或者dst中有一个是None的话, 就直接全返回None
        if dst_predict_result is None or dst_predict_result.multi_face_landmarks is None or \
                src_predict_result is None or src_predict_result.multi_face_landmarks is None:
            return None, None, None, None, None, None, None

        # 从src和dst中得到68个点
        src_face_mesh_pts = np.array([
            [landmark.x * self.input_size[0], landmark.y * self.input_size[1], landmark.z * self.input_size[0], ]
            for idx, landmark in enumerate(src_predict_result.multi_face_landmarks[0].landmark)
        ])
        src_face_landmarks = src_face_mesh_pts[self.face_68_landmarks_ind, :2]
        dst_face_mesh_pts = np.array([
            [landmark.x * self.input_size[0], landmark.y * self.input_size[1], landmark.z * self.input_size[0], ]
            for idx, landmark in enumerate(dst_predict_result.multi_face_landmarks[0].landmark)
        ])
        dst_face_landmarks = dst_face_mesh_pts[self.face_68_landmarks_ind, :2]

        # 权重
        down_face_mask = self.__down_face_mask(dst, dst_face_landmarks)
        mouth_mask, _margin = self.__mouth_mask(dst, dst_face_landmarks)  # 这里返回了一个嘴部的框
        down_teeth_mask = self.__down_teeth_mask(dst, dst_face_landmarks)

        _weight = np.ones_like(dst) * 0.1
        _weight[down_face_mask == 1] = 1.0
        _weight[mouth_mask == 1] = 3.0
        _weight[down_teeth_mask == 1] = 10.0

        # 针对每个不同区域的权重进行可视化
        # plt.subplot(141), plt.title("down_face_mask"), plt.imshow(down_face_mask[:, :, 0])
        # plt.subplot(142), plt.title("mouth_mask"), plt.imshow(mouth_mask[:, :, 0])
        # plt.subplot(143), plt.title("down_teeth_mask"), plt.imshow(down_teeth_mask[:, :, 0])
        # plt.subplot(144), plt.title("_weight"), plt.imshow(_weight[:, :, 0])
        # plt.show()

        # 数据输入
        _in = src1.astype(np.float32).transpose((2, 0, 1)) / 255.0
        _out = dst.astype(np.float32).transpose((2, 0, 1)) / 255.0
        down_face_mask = down_face_mask.transpose((2, 0, 1))
        _weight = _weight.transpose((2, 0, 1))

        return _in, src_face_landmarks.astype(np.float32), dst_face_landmarks.astype(np.float32), \
               _out, _weight, np.array(_margin, dtype=np.int16), down_face_mask

    def collate(self, batch):
        """
        经过getitem之后还要走一遍collate, 这里的collate主要是格式化一下
        """
        data_list = [[], [], [], [], [], [], []]
        for sample in batch:
            if sample[0] is None:
                continue
            else:
                for i, d in enumerate(sample):
                    data_list[i].append(d)
        return default_collate(list(zip(*data_list)))


def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''

    tmp_shape = shape.copy()
    tmp_shape[60] = tmp_shape[48]
    tmp_shape[64] = tmp_shape[54]

    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (tmp_shape[i, 0], tmp_shape[i, 1]), (tmp_shape[i + 1, 0], tmp_shape[i + 1, 1]), color,
                     lineWidth)
        if (loop):
            cv2.line(img, (tmp_shape[idx_list[0], 0], tmp_shape[idx_list[0], 1]),
                     (tmp_shape[idx_list[-1] + 1, 0], tmp_shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))
