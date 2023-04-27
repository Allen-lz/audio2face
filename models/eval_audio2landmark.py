"""
增加各种学习率策略
增加学习率热身策略
"""


import cv2
import torch
import os
import numpy as np
import argparse
import torch.optim as optim
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from models.networks.audio2landmark_network import Audio2landmark_content
from models.networks.audio2landmark_speaker_aware_network import Audio2landmark_speaker_aware
from models.datasets.Audio2landmark import Audio2landmark_Dataset, Audio2landmark_Eval_Dataset
from warmup_scheduler import GradualWarmupScheduler
from model_bl import D_VECTOR
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=0., help='weight decay')
parser.add_argument('--num_window_frames', type=int, default=18, help='')
parser.add_argument('--num_emb_window_frames', type=int, default=128, help='')
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--load_a2l_C_name', type=str, default='')
parser.add_argument('--in_size', type=int, default=80)
parser.add_argument('--drop_out', type=float, default=0.5, help='drop out')
parser.add_argument('--use_prior_net', default=True, action='store_false')
parser.add_argument('--use_lip_weight', default=True, action='store_false')
parser.add_argument('--use_motion_loss', default=False, action='store_true')
parser.add_argument('--nepoch', type=int, default=50000, help='number of epochs to train for')
parser.add_argument('--lr_decay_iters', type=int, default=4000, help='lr_decay_iters')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_size', type=int, default=16, help='num size')
parser.add_argument('--gamma', type=float, default=0.1, help='lr gamma')
parser.add_argument('--lr_policy', type=str, default='step', help='lr schedule policy')
parser.add_argument('--C_ckpt', type=str, default='checkpoints/audio2landmark/best_C.pth', help='lr schedule policy')
parser.add_argument('--G_ckpt', type=str, default='checkpoints/audio2landmark/best_G.pth', help='lr schedule policy')
opt_parser = parser.parse_args()


def drawPart(face_kps, drawimg, start, end, color, closed=True):
    for i in range(start, end + 1):
        cv2.circle(drawimg, (face_kps[i, 0], face_kps[i, 1]), 2, color, thickness=1)
        if i < end:
            cv2.line(drawimg, (face_kps[i, 0], face_kps[i, 1]), (face_kps[i + 1, 0], face_kps[i + 1, 1]), color,
                     thickness=2)
        elif closed:
            cv2.line(drawimg, (face_kps[end, 0], face_kps[end, 1]), (face_kps[start, 0], face_kps[start, 1]), color,
                     thickness=2)
    return drawimg

def drawFace(kps, img):
    # 在绘制之前, 先将68点的形式转成blendershape的形式
    kps = landmark2blendershape(kps)
    img = drawPart(kps, img, 0, 4, (255, 0, 0), False)  # 左眉毛，非封闭区域
    img = drawPart(kps, img, 5, 9, (0, 255, 0), False)  # 右眉毛，非封闭区域
    img = drawPart(kps, img, 10, 15, (255, 0, 0))  # 左眼
    img = drawPart(kps, img, 16, 21, (0, 255, 0))  # 右眼
    img = drawPart(kps, img, 22, 25, (0, 0, 255), False)  # 鼻梁，非封闭区域
    img = drawPart(kps, img, 26, 30, (0, 0, 255), False)  # 鼻子，非封闭区域
    img = drawPart(kps, img, 31, 42, (0, 255, 255))  # 外嘴唇
    img = drawPart(kps, img, 43, 50, (255, 0, 255))  # 内嘴唇
    img = drawPart(kps, img, 51, 67, (255, 255, 0), False)  # 脸轮廓，非封闭区域
    return img


def landmark2blendershape(res):
    """
    0-16 --> 51-67
    17-21 --> 0-4
    22-26 --> 5-9
    27-30 --> 22-25
    31-35 --> 26-30
    36-41 --> 10-15
    42-47 --> 16-21
    48-67 --> 31-50
    :return:
    """
    new_res = np.zeros(res.shape)
    new_res[51: 68] = res[0: 17]
    new_res[0: 5] = res[17: 22]
    new_res[5: 10] = res[22: 27]

    new_res[22: 26] = res[27: 31]
    new_res[26: 31] = res[31: 36]

    new_res[10: 16] = res[36: 42]

    new_res[16: 22] = res[42: 48]
    new_res[31: 51] = res[48: 68]

    new_res = np.array(new_res, dtype=np.int32)

    return new_res

def calculate_loss(fls, fl_dis_pred, face_id, device):
    ''' lip region weight '''

    fls_gt = fls[:, -1, :]  # 序列中的最后一个才是gt吧
    # 这个应该是衡量嘴开合的幅度,
    w = torch.abs(fls[:, -1, 66 * 3 + 1] - fls[:, -1, 62 * 3 + 1])

    # 幅度越大的话权重就会越小
    w = torch.tensor([1.0]).to(device) / (w * 4.0 + 0.1)
    w = w.unsqueeze(1)
    lip_region_w = torch.ones((fls.shape[0], 204)).to(device)
    lip_region_w[:, 48 * 3:] = torch.cat([w] * 60, dim=1)  # 除了嘴部区域的权重要计算, 其余部位的都默认为1
    lip_region_w = lip_region_w.detach().clone().requires_grad_(False)

    if (opt_parser.use_lip_weight):
        # loss = torch.mean(torch.mean((fl_dis_pred + face_id - fls[:, 0, :]) ** 2, dim=1) * w)
        loss = torch.mean(torch.abs(fl_dis_pred + face_id.detach() - fls_gt) * lip_region_w)
    else:
        # loss = self.loss_mse(fl_dis_pred + face_id, fls[:, 0, :])
        loss = torch.nn.functional.l1_loss(fl_dis_pred + face_id.detach(), fls_gt)

    if (opt_parser.use_motion_loss):
        # 缩小这个运动的轨迹, 使得运动更加的平滑
        pred_motion = fl_dis_pred[:-1] - fl_dis_pred[1:]
        gt_motion = fls_gt[:-1] - fls_gt[1:]
        loss += torch.nn.functional.l1_loss(pred_motion, gt_motion)

    return loss

def draw_res(fls_gt, fl_dis_pred, face_id, data_scales, data_shifts, std_scales, std_shifts):
    # 先将关键点统一scale到一个size
    fls_gt = fls_gt[:, -1, :]
    fls_gt = fls_gt.detach().cpu()

    data_scales = data_scales[:, -1, :, :].detach().cpu()
    data_shifts = data_shifts[:, -1, :, :].detach().cpu()
    std_scales = std_scales[:, -1, :, :].detach().cpu()
    std_shifts = std_shifts[:, -1, :, :].detach().cpu()

    fl_dis_pred = fl_dis_pred.detach().cpu()
    face_id = face_id.detach().cpu()

    for fl_dis_i, fls_gt_i, face_id_i, data_scale_i, data_shift_i, std_scale_i, std_shift_i in zip(fl_dis_pred, fls_gt, face_id, data_scales, data_shifts, std_scales, std_shifts):
        """
        std_scales, std_shifts的作用仅仅是为了在训练的时候能对齐数据而已,在inference反解码的时候用的还是data_scales, data_shifts, 毕竟data是target
        """

        fl_dis_i, fls_gt_i = fl_dis_i.reshape(68, 3), fls_gt_i.reshape(68, 3)
        face_id_i = face_id_i.reshape(68, 3)

        fls_gt_i = fls_gt_i[:, :2] / data_scale_i - data_shift_i

        fl_pre_i = fl_dis_i + face_id_i
        fl_pre_i = fl_pre_i[:, :2] / data_scale_i - data_shift_i

        gt_plane = np.zeros((336, 336, 3))
        gt_img = drawFace(fls_gt_i, gt_plane)
        pred_plane = np.zeros((336, 336, 3))
        pred_img = drawFace(fl_pre_i, pred_plane)

        plt.subplot(121), plt.title("ground truth"), plt.imshow(gt_img)
        plt.subplot(122), plt.title("predict"), plt.imshow(pred_img)
        plt.show()






def eval(device):
    """
    其实这里要先训练C之后再加载训练更好的C来训练G的
    这里我为了简单方便, 就是一起train的, 之后可以对比看看效果如何
    Args:
        device:
    Returns:
    """
    C = Audio2landmark_content(num_window_frames=opt_parser.num_window_frames,
                               hidden_size=opt_parser.hidden_size,
                               in_size=opt_parser.in_size,
                               use_prior_net=opt_parser.use_prior_net,
                               bidirectional=False,
                               drop_out=opt_parser.drop_out).to(device)
    C.load_state_dict(torch.load(opt_parser.C_ckpt))

    speaker_aware = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).to(device)

    G = Audio2landmark_speaker_aware().to(device)
    G.load_state_dict(torch.load(opt_parser.G_ckpt))

    val_data = Audio2landmark_Eval_Dataset(num_window_frames=opt_parser.num_window_frames,
                                        num_emb_window_frames=opt_parser.num_emb_window_frames,
                                        num_size=opt_parser.num_size)
    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                   batch_size=opt_parser.batch_size, # 这里的batchsize是选取几段wav的意思, 不是通常的那种batchsize
                                                   shuffle=False, num_workers=0,
                                                   collate_fn=val_data.my_collate_in_segments)
    for i in range(opt_parser.nepoch):
        for batch in val_dataloader:
            """
            train_dataloader中的batch的个数是与参与构建dataset的wav个数有关的
            一个wav是同一个人在一段时间说的一段话, 
            """
            fls, aus, emb_aus, std, data_scales, data_shifts, std_scales, std_shifts = batch
            fls, aus, emb_aus, std, data_scales, data_shifts, std_scales, std_shifts = fls.to(device), aus.to(device), emb_aus.to(device), std.to(device), data_scales.to(device), data_shifts.to(device), std_scales.to(device), std_shifts.to(device)
            std = std[:, 0, :]
            fl_dis_pred, face_id = C(aus, std)

            emb = speaker_aware(emb_aus)
            emb_fl_dis_pred = G(aus, emb)

            # 加上属于这个人id的残差偏移
            fl_dis_pred += emb_fl_dis_pred

            loss = calculate_loss(fls, fl_dis_pred, face_id, device)
            draw_res(fls, fl_dis_pred, face_id, data_scales, data_shifts, std_scales, std_shifts)


if __name__ == "__main__":
    device = "cuda"
    eval(device)