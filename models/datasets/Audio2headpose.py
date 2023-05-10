"""
这个是由音频到头部姿态的数据加载脚本

1. gt是当前帧的head pose与第一帧的差值
2. 这个的T也是18帧
"""


import torch
import torch.utils.data as data
import numpy as np
import os
import random
import sys
sys.path.append(".")
from models.utils.landmark_normal import norm_input_face
import matplotlib.pyplot as plt
import cv2


from models._3dmm.inference_3dmm import FaceRecon

def close_input_face_mouth(shape_3d, p1=0.7, p2=0.5):
    """
    使用规则将人的关键点的嘴闭上
    """

    # 这是我写的一个很简单的方法
    # shape_3d = shape_3d.reshape(68, 3)
    # y_61_67 = (shape_3d[61, 1] + shape_3d[67, 1]) // 2
    # y_62_66 = (shape_3d[62, 1] + shape_3d[66, 1]) // 2
    # y_63_65 = (shape_3d[63, 1] + shape_3d[65, 1]) // 2
    #
    # shape_3d[61, 1] = y_61_67
    # shape_3d[67, 1] = y_61_67
    #
    # shape_3d[62, 1] = y_62_66
    # shape_3d[66, 1] = y_62_66
    #
    # shape_3d[63, 1] = y_63_65
    # shape_3d[65, 1] = y_63_65

    # 这个是原来的方法
    shape_3d = shape_3d.reshape((1, 68, 3))
    index1 = list(range(60 - 1, 55 - 1, -1))
    index2 = list(range(68 - 1, 65 - 1, -1))
    mean_out = 0.5 * (shape_3d[:, 49:54] + shape_3d[:, index1])
    mean_in = 0.5 * (shape_3d[:, 61:64] + shape_3d[:, index2])
    shape_3d[:, 50:53] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, list(range(59 - 1, 56 - 1, -1))] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d[:, 49] -= (shape_3d[:, 61] - mean_in[:, 0]) * p2
    shape_3d[:, 53] -= (shape_3d[:, 63] - mean_in[:, -1]) * p2
    shape_3d[:, 59] -= (shape_3d[:, 67] - mean_in[:, 0]) * p2
    shape_3d[:, 55] -= (shape_3d[:, 65] - mean_in[:, -1]) * p2
    # shape_3d[:, 61:64] = shape_3d[:, index2] = mean_in
    shape_3d[:, 61:64] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, index2] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d = shape_3d.reshape((68, 3))
    return shape_3d

def _3d_vis(points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2], zdir='z', c='c')
    plt.show()

def std_mean_calculation(au_data_list):
    all_au_data = []
    for au in au_data_list:
        all_au_data.append(au)
    all_au_data = np.concatenate(all_au_data, axis=0)
    au_std = np.std(all_au_data, axis=0)
    au_mean = np.mean(all_au_data, axis=0)

    return au_mean, au_std

class Audio2headpose_Dataset(data.Dataset):
    def __init__(self, num_window_frames, num_emb_window_frames, num_size):
        """
        加载每个子文件夹下的所有图片并使用face_recon得到每张图片的pose

        取出第一帧为β0, 然后使用当前帧的β与第一帧的做个差



        Args:
            num_window_frames:
            num_emb_window_frames:
            num_size:
        """
        self.num_window_frames = num_window_frames  # 18
        self.num_emb_window_frames = num_emb_window_frames  # 18
        self.num_size = num_size

        self.au_data_dir = "generate_audio"
        self.fl_data_dir = "generate_lm_np"

        # update
        self.face_data_dir = "generate_face"
        self.face_coeff_dir = "generate_face_coeff"
        self.face_recon = FaceRecon("models/_3dmm/BFM/epoch_20.pth",
                                    "models/_3dmm/BFM")

        self.au_data_list = []
        self.fl_data_list = []
        self.fl_std_list = []

        # update
        self.face_coeff_list = []
        self.face_coeff_std_list = []

        for file_name in os.listdir(self.au_data_dir):
            with open(os.path.join(self.au_data_dir, file_name), 'rb') as fp:
                au_data = np.load(fp)
                self.au_data_list.append(au_data)

        for file_name in os.listdir(self.fl_data_dir):
            with open(os.path.join(self.fl_data_dir, file_name), 'rb') as fp:
                fl_data = np.load(fp)  # 里面包含等待预测的关键点 和 speaker的embedding  [(305, 204),], [(256, )]
                # 这里是针对(1, 68, 3)的点需要先进行转换成(1, 204)
                fl_reshape_data = []
                fl_std_reshape_data = []
                fl_std = close_input_face_mouth(fl_data[0].reshape(1, 204).copy()).reshape(1, 204)
                for i, lm in enumerate(fl_data):
                    fl_reshape_data.append(fl_data[i].reshape(1, 204))
                    fl_std_reshape_data.append(fl_std)

                # 这里是append不同的视频段
                self.fl_data_list.append(np.concatenate(fl_reshape_data, axis=0))
                self.fl_std_list.append(np.concatenate(fl_std_reshape_data, axis=0))

        # ================================== update ==================================
        for dir_name in os.listdir(self.face_data_dir):
            cur_coeff_dir_path = os.path.join(self.face_coeff_dir, dir_name)
            if not os.path.exists(cur_coeff_dir_path):
                os.makedirs(cur_coeff_dir_path)

            cur_dir_path = os.path.join(self.face_data_dir, dir_name)  # subdir
            cur_image_list = os.listdir(cur_dir_path)  # subdir中的图片

            cur_dir_coeff_list = []
            cur_dir_first_coeff_list = []
            for i, cur_image in enumerate(cur_image_list):
                # 在进行推理之前先要判断一下这张图片是否已经被推理过了
                coeff_file = os.path.join(cur_coeff_dir_path, os.path.splitext(cur_image)[0] + ".npy")
                if os.path.exists(coeff_file):
                    # 这个face_coeff可以直接拿出来, 就能直接append到cur_dir_coeff_list上去
                    face_coeff = np.load(coeff_file)
                else:
                    cur_image_path = os.path.join(self.face_data_dir, dir_name, cur_image)
                    cur_image = cv2.imread(cur_image_path)
                    cur_face_coeff = self.face_recon.run(cur_image)
                    angle = cur_face_coeff["angle"]
                    s = np.array([[cur_face_coeff["trans_params"][0]]])
                    face_coeff = np.concatenate([angle, s], axis=-1)
                    np.save(coeff_file, face_coeff)

                cur_dir_coeff_list.append(face_coeff)
                # 一直append第一帧的系数
                cur_dir_first_coeff_list.append(cur_dir_coeff_list[0])

            cur_dir_coeff_list = np.concatenate(cur_dir_coeff_list, axis=0)
            cur_dir_first_coeff_list = np.concatenate(cur_dir_first_coeff_list, axis=0)

            self.face_coeff_list.append(cur_dir_coeff_list)
            self.face_coeff_std_list.append(cur_dir_first_coeff_list)
        # ============================================================================


        valid_idx = list(range(len(self.au_data_list)))

        random.seed(0)
        random.shuffle(valid_idx)

        self.au_data_list = [self.au_data_list[i] for i in valid_idx]
        self.fl_data_list = [self.fl_data_list[i] for i in valid_idx]
        self.fl_std_list = [self.fl_std_list[i] for i in valid_idx]

        # 音频数据标准化
        au_mean, au_std = std_mean_calculation(self.au_data_list)
        self.au_data_list = [(au_data - au_mean[np.newaxis, :]) / au_std[np.newaxis, :] for au_data in self.au_data_list]

        # 关键点归一化
        self.fl_data_list = [norm_input_face(fl_data) for fl_data in self.fl_data_list]
        self.fl_std_list = [norm_input_face(fl_std) for fl_std in self.fl_std_list]

    def __len__(self):
        return len(self.fl_data_list)

    def __getitem__(self, item):
        """
        这里的item指的是一个连续的音频文件及其对应的face_landmark, 而不是某个时刻的mel及其对应的lm
        Args:
            item:
        Returns:
        """
        # print('-> get item {}: {} {}'.format(item, self.fl_data[item][1][0], self.fl_data[item][1][1]))
        return self.fl_data_list[item], self.au_data_list[item], self.fl_std_list[item], self.face_coeff_list[item], self.face_coeff_std_list[item]

    def my_collate_in_segments(self, batch):
        """
        这个是没有face emb的,慢慢来,暂时不要有id emb的
        Args:
            batch: 这个是选取的wav
            length: 这是每个wav选择多少个窗口

        Returns:

        """

        fls, aus, emb_aus, std, fc, fc_std = [], [], [], [], [], []
        per_num = self.num_size // len(batch)
        for fl_data, au_data, std_data, fc_data, fc_std_data in batch:
            """
            对于某段音频, 都随机截取N / len(batch)个窗口
            """

            assert (fl_data.shape[0] == au_data.shape[0])

            fl_data = torch.tensor(fl_data, dtype=torch.float, requires_grad=False)    # torch.Size([2860, 204])
            au_data = torch.tensor(au_data, dtype=torch.float, requires_grad=False)    # torch.Size([2860, 80])
            std_data = torch.tensor(std_data, dtype=torch.float, requires_grad=False)  # torch.Size([2860, 204])

            fc_data = torch.tensor(fc_data, dtype=torch.float, requires_grad=False)
            fc_std_data = torch.tensor(fc_std_data, dtype=torch.float, requires_grad=False)

            for i in range(per_num):

                # 这样抽取数据的方式对一开始的效果不友好, 一开始肯定是要用第一帧疯狂进行padding, 所以这里要随机的是right而不是left
                left = random.randint(0, fl_data.shape[0] - self.num_window_frames)

                # window shift data
                fls += [fl_data[left:left + self.num_window_frames]]
                aus += [au_data[left:left + self.num_window_frames]]
                std += [std_data[left:left + self.num_window_frames]]

                # update
                fc += [fc_data[left:left + self.num_window_frames]]
                fc_std += [fc_std_data[left:left + self.num_window_frames]]

                # au emb
                # 并同步得到计算emb的aus, left + self.num_window_frames依旧是分段音频的终点, 从上面音频的边界往前数, 前面不够了就填充
                if left + self.num_window_frames - self.num_emb_window_frames >= 0:
                    emb_aus += [au_data[left + self.num_window_frames - self.num_emb_window_frames:left + self.num_window_frames]]
                else:
                    num_repeat = self.num_emb_window_frames // self.num_window_frames + 1
                    repeat_au_data = au_data[left:left + self.num_window_frames].repeat(num_repeat, 1)
                    repeat_au_data = repeat_au_data[repeat_au_data.shape[0] - self.num_emb_window_frames:]
                    emb_aus += [repeat_au_data]

        fls = torch.stack(fls, dim=0)
        aus = torch.stack(aus, dim=0)
        emb_aus = torch.stack(emb_aus, dim=0)
        std = torch.stack(std, dim=0)
        fc = torch.stack(fc, dim=0)
        fc_std = torch.stack(fc_std, dim=0)

        return fls, aus, emb_aus, std, fc, fc_std

class Audio2headpose_Eval_Dataset(data.Dataset):
    def __init__(self, num_window_frames, num_emb_window_frames, num_size):
        self.num_window_frames = num_window_frames  # 18
        self.num_emb_window_frames = num_emb_window_frames  # 18
        self.num_size = num_size

        self.au_data_dir = "generate_audio"
        self.fl_data_dir = "generate_lm_np"

        # ============================================ update ================================================
        self.face_data_dir = "generate_face"
        self.face_coeff_dir = "generate_face_coeff"
        self.face_recon = FaceRecon("D:/audio2face/SadTalker/checkpoints/epoch_20.pth",
                                    "D:/audio2face/SadTalker/checkpoints/BFM_Fitting")

        self.au_data_list = []
        self.fl_data_list = []
        self.fl_std_list = []

        # update
        self.face_coeff_list = []
        self.face_coeff_std_list = []

        for file_name in os.listdir(self.au_data_dir):
            with open(os.path.join(self.au_data_dir, file_name), 'rb') as fp:
                au_data = np.load(fp)
                self.au_data_list.append(au_data)

        for file_name in os.listdir(self.fl_data_dir):
            with open(os.path.join(self.fl_data_dir, file_name), 'rb') as fp:
                fl_data = np.load(fp)  # 里面包含等待预测的关键点 和 speaker的embedding  [(305, 204),], [(256, )]
                # 这里是针对(1, 68, 3)的点需要先进行转换成(1, 204)
                fl_reshape_data = []
                fl_std_reshape_data = []
                fl_std = close_input_face_mouth(fl_data[0].reshape(1, 204).copy()).reshape(1, 204)
                for i, lm in enumerate(fl_data):
                    fl_reshape_data.append(fl_data[i].reshape(1, 204))
                    fl_std_reshape_data.append(fl_std)

                self.fl_data_list.append(np.concatenate(fl_reshape_data, axis=0))
                self.fl_std_list.append(np.concatenate(fl_std_reshape_data, axis=0))

        # ================================== update ==================================
        for dir_name in os.listdir(self.face_data_dir):
            cur_coeff_dir_path = os.path.join(self.face_coeff_dir, dir_name)
            if not os.path.exists(cur_coeff_dir_path):
                os.makedirs(cur_coeff_dir_path)

            cur_dir_path = os.path.join(self.face_data_dir, dir_name)  # subdir
            cur_image_list = os.listdir(cur_dir_path)  # subdir中的图片

            cur_dir_coeff_list = []
            cur_dir_first_coeff_list = []
            for i, cur_image in enumerate(cur_image_list):
                # 在进行推理之前先要判断一下这张图片是否已经被推理过了
                coeff_file = os.path.join(cur_coeff_dir_path, os.path.splitext(cur_image)[0] + ".npy")
                if os.path.exists(coeff_file):
                    # 这个face_coeff可以直接拿出来, 就能直接append到cur_dir_coeff_list上去
                    face_coeff = np.load(coeff_file)
                else:
                    cur_image_path = os.path.join(self.face_data_dir, dir_name, cur_image)
                    cur_image = cv2.imread(cur_image_path)
                    cur_face_coeff = self.face_recon.run(cur_image)
                    angle = cur_face_coeff["angle"]
                    s = np.array([[cur_face_coeff["trans_params"][0]]])
                    face_coeff = np.concatenate([angle, s], axis=-1)
                    np.save(coeff_file, face_coeff)

                cur_dir_coeff_list.append(face_coeff)
                # 一直append第一帧的系数
                cur_dir_first_coeff_list.append(cur_dir_coeff_list[0])

            cur_dir_coeff_list = np.concatenate(cur_dir_coeff_list, axis=0)
            cur_dir_first_coeff_list = np.concatenate(cur_dir_first_coeff_list, axis=0)

            self.face_coeff_list.append(cur_dir_coeff_list)
            self.face_coeff_std_list.append(cur_dir_first_coeff_list)
        # ============================================================================

        valid_idx = list(range(len(self.au_data_list)))

        random.seed(0)
        random.shuffle(valid_idx)

        self.au_data_list = [self.au_data_list[i] for i in valid_idx]
        self.fl_data_list = [self.fl_data_list[i] for i in valid_idx]
        self.fl_std_list = [self.fl_std_list[i] for i in valid_idx]

        # 音频数据标准化
        au_mean, au_std = std_mean_calculation(self.au_data_list)
        self.au_data_list = [(au_data - au_mean[np.newaxis, :]) / au_std[np.newaxis, :] for au_data in self.au_data_list]

        # 关键点归一化, 并保存归一化时候的参数
        self.fl_norm_data_list = []
        self.fl_norm_std_list = []

        self.data_scale_list = []
        self.data_shift_list = []
        self.std_scale_list = []
        self.std_shift_list = []
        for fl_data in self.fl_data_list:
            fl_norm_data, scale_data, shift_data = norm_input_face(fl_data, phrase="val")
            self.data_scale_list.append(scale_data)
            self.data_shift_list.append(shift_data)
            self.fl_norm_data_list.append(fl_norm_data)

        for fl_std in self.fl_std_list:
            fl_norm_std, scale_std, shift_std = norm_input_face(fl_std, phrase="val")
            self.std_scale_list.append(scale_std)
            self.std_shift_list.append(shift_std)
            self.fl_norm_std_list.append(fl_norm_std)

        self.fl_data_list = self.fl_norm_data_list
        self.fl_std_list = self.fl_norm_std_list

    def __len__(self):
        return len(self.fl_data_list)

    def __getitem__(self, item):
        """
        这里的item指的是一个连续的音频文件及其对应的face_landmark, 而不是某个时刻的mel及其对应的lm
        Args:
            item:
        Returns:
        """
        return self.fl_data_list[item], self.au_data_list[item], self.fl_std_list[item], self.face_coeff_list[item], self.face_coeff_std_list[item], \
               self.data_scale_list[item], self.data_shift_list[item],\
               self.std_scale_list[item], self.std_shift_list[item],

    def my_collate_in_segments(self, batch):
        """
        这个是没有face emb的,慢慢来,暂时不要有id emb的
        Args:
            batch: 这个是选取的wav
            length: 这是每个wav选择多少个窗口

        Returns:

        """

        fls, aus, emb_aus, std, fc, fc_std = [], [], [], [], [], []
        data_scales, data_shifts, std_scales, std_shifts = [], [], [], []

        per_num = self.num_size // len(batch)
        for fl_data, au_data, std_data, fc_data, fc_std_data, fl_scale, fl_shift, std_scale, std_shift in batch:
            """
            对于某段音频, 都随机截取N / len(batch)个窗口
            """

            assert (fl_data.shape[0] == au_data.shape[0])

            fl_data = torch.tensor(fl_data, dtype=torch.float, requires_grad=False)
            au_data = torch.tensor(au_data, dtype=torch.float, requires_grad=False)
            std_data = torch.tensor(std_data, dtype=torch.float, requires_grad=False)

            fc_data = torch.tensor(fc_data, dtype=torch.float, requires_grad=False)
            fc_std_data = torch.tensor(fc_std_data, dtype=torch.float, requires_grad=False)

            fl_scale = torch.tensor(fl_scale, dtype=torch.float, requires_grad=False)
            fl_shift = torch.tensor(fl_shift, dtype=torch.float, requires_grad=False)
            std_scale = torch.tensor(std_scale, dtype=torch.float, requires_grad=False)
            std_shift = torch.tensor(std_shift, dtype=torch.float, requires_grad=False)


            for i in range(per_num):

                left = random.randint(0, fl_data.shape[0] - self.num_window_frames)

                # window shift data
                fls += [fl_data[left:left + self.num_window_frames]]
                aus += [au_data[left:left + self.num_window_frames]]
                std += [std_data[left:left + self.num_window_frames]]

                # update
                fc += [fc_data[left:left + self.num_window_frames]]
                fc_std += [fc_std_data[left:left + self.num_window_frames]]

                # 增加一些归一化的参数
                data_scales += [fl_scale[left:left + self.num_window_frames]]
                data_shifts += [fl_shift[left:left + self.num_window_frames]]
                std_scales += [std_scale[left:left + self.num_window_frames]]
                std_shifts += [std_shift[left:left + self.num_window_frames]]

                # au emb
                # 并同步得到计算emb的aus, left + self.num_window_frames依旧是分段音频的终点
                if left + self.num_window_frames - self.num_emb_window_frames >= 0:
                    emb_aus += [au_data[left + self.num_window_frames - self.num_emb_window_frames:left + self.num_window_frames]]
                else:
                    num_repeat = self.num_emb_window_frames // self.num_window_frames + 1
                    repeat_au_data = au_data[left:left + self.num_window_frames].repeat(num_repeat, 1)
                    repeat_au_data = repeat_au_data[repeat_au_data.shape[0] - self.num_emb_window_frames:]
                    emb_aus += [repeat_au_data]

        fls = torch.stack(fls, dim=0)
        aus = torch.stack(aus, dim=0)
        emb_aus = torch.stack(emb_aus, dim=0)
        std = torch.stack(std, dim=0)

        fc = torch.stack(fc, dim=0)
        fc_std = torch.stack(fc_std, dim=0)

        data_scales = torch.stack(data_scales, dim=0)
        data_shifts = torch.stack(data_shifts, dim=0)

        std_scales = torch.stack(std_scales, dim=0)
        std_shifts = torch.stack(std_shifts, dim=0)

        return fls, aus, emb_aus, std, fc, fc_std, data_scales, data_shifts, std_scales, std_shifts


if __name__ == "__main__":
    phrase = "val"
    if phrase == "train":
        train_data = Audio2headpose_Dataset(num_window_frames=18, num_emb_window_frames=128, num_size=16)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1,  # 这里的batchsize是选取几段wav的意思, 不是通常的那种batchsize
                                                        shuffle=False, num_workers=0,
                                                        collate_fn=train_data.my_collate_in_segments)

        for i in range(50000):
            for batch in train_dataloader:
                fls, aus, emb_aus, std, fc, fc_std = batch
                print(fls.shape, aus.shape, emb_aus.shape, std.shape, fc.shape, fc_std.shape)

    else:
        val_data = Audio2headpose_Eval_Dataset(num_window_frames=18, num_emb_window_frames=128, num_size=16)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                                       # 这里的batchsize是选取几段wav的意思, 不是通常的那种batchsize
                                                       shuffle=False, num_workers=0,
                                                       collate_fn=val_data.my_collate_in_segments)

        for i in range(50000):
            for batch in val_dataloader:
                fls, aus, emb_aus, std, fc, fc_std, data_scales, data_shifts, std_scales, std_shifts = batch
                print(data_scales.shape)
                print(data_shifts.shape)
