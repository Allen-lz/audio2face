# -*-coding:utf-8 -*-

"""
# File   : image_translation_dataset_for_LIA.py
# Time   : 2023/4/11 17:10
# Author : luzeng
# version: python 3.9
"""

import os
import glob2
import cv2
import skimage.io
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt


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

class ImageTranslationDatasetForLIA(data.Dataset):
    """
    Online landmark extraction with AWings
    Landmark setting: 98 landmarks
    """

    def __init__(self, data_dir, exp_data_dir, noise_prob=0.2, img_size=(512, 512)):
        super(ImageTranslationDatasetForLIA, self).__init__()

        self.data_dir = data_dir
        self.exp_data_dir = exp_data_dir
        self.exp_data_images_path = []
        self.get_all_img(self.exp_data_dir)

        self.noise_prob = noise_prob
        self.img_size = img_size

        self.margin_t = 10

        self.data = []
        for sub_dir in os.listdir(self.data_dir):
            if not os.path.exists(os.path.join(self.data_dir, sub_dir, "images")):
                continue

            for chunk_dir in os.listdir(os.path.join(self.data_dir, sub_dir, "images")):
                key = "{}-{}".format(sub_dir, chunk_dir)
                # 得到一个chunk子文件夹中所有图片
                image_paths = glob2.glob(os.path.join(self.data_dir, sub_dir, "images", chunk_dir, "*.jpg"))
                filter_image_paths = []
                # 图片筛选, 不是所有图片都有face landmark
                for path in image_paths:
                    fl_path = path.replace(os.path.join("images", chunk_dir),
                                           os.path.join("landmarks", chunk_dir)).replace(".jpg", ".npy")
                    # filter_image_paths中只保留有landmark的图片路径
                    if os.path.exists(fl_path):
                        filter_image_paths.append(path)

                # 得到image对应的face landmark文件
                landmarks_paths = [
                    path.replace(os.path.join("images", chunk_dir),
                                 os.path.join("landmarks", chunk_dir))
                        .replace(".jpg", ".npy")
                    for path in filter_image_paths
                ]
                # 一个chunk是self.data中的一个元素
                if len(filter_image_paths) > 0:
                    self.data.append([key, filter_image_paths, landmarks_paths])

    def __len__(self):
        return len(self.data)

    def get_all_img(self, data_dir):
        """
        得到一个文件夹中的所有图片文件
        Returns:
        """
        items = os.listdir(data_dir)

        for item in items:
            if os.path.isdir(os.path.join(data_dir, item)):
                dir_path = os.path.join(data_dir, item)
                self.get_all_img(dir_path)
            else:
                suffix = os.path.splitext(item)[-1]
                if suffix.lower() == ".png" or suffix.lower() == ".jpg" or suffix.lower() == ".jpeg":
                    image_path = os.path.join(data_dir, item)
                    self.exp_data_images_path.append(image_path)


    def __down_face_mask(self, img, fl):
        down_face_mask = np.zeros_like(img)
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
        ], dtype=np.int32)[:, :2]
        (min_x, min_y), (max_x, max_y) = poly.min(0), poly.max(0)
        down_face_mask[max(min_y - self.margin_t, 0): min(max_y + self.margin_t + 1, img.shape[0]),
        max(min_x - self.margin_t, 0): min(max_x + self.margin_t + 1, img.shape[1])] = 1.
        return down_face_mask

    def __mouth_oral_mask(self, img, fl):
        down_teeth_mask = np.zeros_like(img)
        poly = np.array([fl[48, :2], fl[67, :2], fl[66, :2], fl[65, :2],
                         fl[64, :2], fl[63, :2], fl[62, :2], fl[61, :2]], dtype=np.int32)
        # poly = np.array([fl[48, :2], fl[51, :2], fl[64, :2], fl[57, :2]], dtype=np.int32)
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
        chunk = self.data[item]
        image_path_list = chunk[1]
        fl_path_list = chunk[2]

        # 随机抽取2帧, 一帧作为src一帧作为dst
        rand_ind = np.random.randint(len(image_path_list) - 1, size=2)

        rand_exp_ind = np.random.randint(len(self.exp_data_images_path) - 1, size=1)

        fl_path = fl_path_list[rand_ind[1]]
        src = skimage.io.imread(image_path_list[rand_ind[0]])
        dst = skimage.io.imread(image_path_list[rand_ind[1]])
        exp_ref = skimage.io.imread(self.exp_data_images_path[rand_exp_ind[0]])  # numpy RGB

        # resize
        ori_size = (src.shape[1], src.shape[0])  # 这里先记录下src原来的大小
        src = cv2.resize(src, self.img_size)
        dst = cv2.resize(dst, self.img_size)
        exp_ref = cv2.resize(exp_ref, self.img_size)

        # dst landmarks, 归一化之后再映射到目标尺度
        fl = np.load(fl_path)
        norm_fl = fl / [ori_size[0], ori_size[1], ori_size[0]]
        suitable_fl = norm_fl * [self.img_size[0], self.img_size[1], self.img_size[0]]

        # 这里的权重的生成方式和最初的是一样的
        # mask
        down_face_mask = self.__down_face_mask(dst, suitable_fl)
        mouth_mask, _margin = self.__mouth_mask(dst, suitable_fl)
        mouth_oral_mask = self.__mouth_oral_mask(dst, suitable_fl)

        # loss权重
        _weight = np.ones_like(dst)
        _weight[down_face_mask == 1] = 2.0
        _weight[mouth_mask == 1] = 5.0
        _weight[mouth_oral_mask == 1] = 10.0

        _in = (src.astype(np.float32).transpose((2, 0, 1)) / 255.0 - 0.5) * 2.
        _out = (dst.astype(np.float32).transpose((2, 0, 1)) / 255.0 - 0.5) * 2.
        _exp_ref = exp_ref.astype(np.float32).transpose((2, 0, 1)) / 255.0




        _weight = _weight.transpose((2, 0, 1))

        return _in, norm_fl[:, :2].reshape(-1), _out, _weight, np.array(_margin, dtype=np.int16), _exp_ref
