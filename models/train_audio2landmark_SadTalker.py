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
from models.datasets.Audio2headpose import Audio2headpose_Dataset

from models.networks.CVAE.cvae import VAE
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
from model_bl import D_VECTOR

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
parser.add_argument('--device', type=str, default="cuda:0", help='')

# model save
parser.add_argument('--jpg_freq', type=int, default=1, help='')
parser.add_argument('--ckpt_epoch_freq', type=int, default=1000, help='')
parser.add_argument('--ckpt_save_dir', type=str, default="checkpoints/audio2landmark", help='')


# head pose cvae
parser.add_argument("--latent_size", type=int, default=6)
parser.add_argument("--audio_feat_size", type=int, default=80)
parser.add_argument("--c_enc_hidden_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--proj_dim", type=int, default=32)
parser.add_argument("--cvae_drop_out", type=int, default=0.5)
parser.add_argument("--vea_weight", default="checkpoints/audio2landmark/best_vae.pth", type=str)


opt_parser = parser.parse_args()


def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(angles.device)
    zeros = torch.zeros([batch_size, 1]).to(angles.device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        # https://blog.csdn.net/qq_40714949/article/details/126287769
        def lambda_rule(epoch):
            """
            用一个函数来表示lr的下降策略
            """
            lamda = 1.0 - max(0, epoch) / opt.n_epochs
            return lamda
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lr_lambda=lambda_rule,
                                          last_epoch=-1)
    elif opt.lr_policy == 'step':
        # https://blog.csdn.net/weixin_42305378/article/details/108740926
        lr_decay_iters = opt.nepoch // 3
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=lr_decay_iters,
                                        gamma=opt.gamma,
                                        last_epoch=-1)
    elif opt.lr_policy == 'plateau':
        # https://blog.csdn.net/weixin_40100431/article/details/84311430
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.5,
                                                   threshold=1e-4,
                                                   patience=10)
    elif opt.lr_policy == 'cosine':
        # https://blog.csdn.net/weixin_44682222/article/details/122218046
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,  # 这里一般就是给一个epoch的总数
                                                   eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def calculate_loss(fls, fl_dis_pred, face_id, rot_mat, device):
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

    # ===========================================================================================
    # 使用旋转矩阵对得到的landmark进行旋转得到旋转后的点
    # 使用rot对预测出来的关键点进行旋转
    # 旋转之后进行坐标轴还原和shape还原
    fl_pre = fl_dis_pred + face_id.detach()
    fl_pre = fl_pre.reshape(fl_pre.shape[0], 68, 3)
    fl_pre[:, 1] = 1 - fl_pre[:, 1]
    fl_pre_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, fl_pre)
    fl_pre_rotated[:, 1] = 1 - fl_pre_rotated[:, 1]
    fl_pre_rotated = fl_pre_rotated.reshape(fl_pre.shape[0], 68 * 3)
    # ===========================================================================================

    if (opt_parser.use_lip_weight):
        # loss = torch.mean(torch.mean((fl_dis_pred + face_id - fls[:, 0, :]) ** 2, dim=1) * w)
        loss = torch.mean(torch.abs(fl_pre_rotated - fls_gt) * lip_region_w)
    else:
        # loss = self.loss_mse(fl_dis_pred + face_id, fls[:, 0, :])
        loss = torch.nn.functional.l1_loss(fl_pre_rotated, fls_gt)

    if (opt_parser.use_motion_loss):
        # 缩小这个运动的轨迹, 使得运动更加的平滑, 但是这个队数据加载有一定的要求, 就是一个batch中的数据必须是连续的
        pred_motion = fl_dis_pred[:-1] - fl_dis_pred[1:]
        gt_motion = fls_gt[:-1] - fls_gt[1:]
        loss += torch.nn.functional.l1_loss(pred_motion, gt_motion)

    return loss


def train_audio2landmark(device):
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

    speaker_aware = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).to(device)
    # 在这里创建一个pose_cvae
    headpose_cvae = VAE(
        latent_size=opt_parser.latent_size,
        audio_feat_size=opt_parser.audio_feat_size,
        c_enc_hidden_size=opt_parser.c_enc_hidden_size,
        num_layers=opt_parser.num_layers,
        proj_dim=opt_parser.proj_dim,
        drop_out=opt_parser.cvae_drop_out
    ).to(device)
    headpose_cvae.load_weight(opt_parser.vea_weight)

    G = Audio2landmark_speaker_aware().to(device)

    optimizer = optim.Adam(list(C.parameters()) + list(G.parameters()), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

    # optimizer = torch.optim.AdamW(list(C.parameters()) + list(G.parameters()), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

    lr_scheduler = get_scheduler(optimizer, opt_parser)
    # https://blog.csdn.net/qq_38964360/article/details/126330336
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=500, after_scheduler=lr_scheduler)

    train_data = Audio2headpose_Dataset(num_window_frames=opt_parser.num_window_frames,
                                        num_emb_window_frames=opt_parser.num_emb_window_frames,
                                        num_size=opt_parser.num_size)
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=opt_parser.batch_size, # 这里的batchsize是选取几段wav的意思, 不是通常的那种batchsize
                                                   shuffle=False, num_workers=0,
                                                   collate_fn=train_data.my_collate_in_segments)


    best_loss = float("inf")
    for i in range(opt_parser.nepoch):
        for batch in train_dataloader:
            """
            train_dataloader中的batch的个数是与参与构建dataset的wav个数有关的
            一个wav是同一个人在一段时间说的一段话, 
            """
            fls, aus, emb_aus, std, fc, fc_std = batch
            fls, aus, emb_aus, std, fc, fc_std = fls.to(device), aus.to(device), emb_aus.to(device), std.to(device), fc.to(device), fc_std.to(device)
            std = std[:, 0, :]

            emb = speaker_aware(emb_aus)

            # =========================== 得到头部的旋转角度 =========================
            z = torch.randn((aus.shape[0], opt_parser.latent_size)).to(device)
            angle = headpose_cvae.inference(z, aus, emb)[:, :3]
            rot_mat = compute_rotation(angle)
            # =======================================================================

            emb_fl_dis_pred = G(aus, emb)
            fl_dis_pred, face_id = C(aus, std)

            # 加上属于这个人的残差偏移, 这个应该是归一化之后的点, 使用的是图像的像素坐标系
            fl_dis_pred += emb_fl_dis_pred

            # 将得到的rot_mat数据输入对关键点进行旋转之后就能得到最后的landmark
            loss = calculate_loss(fls, fl_dis_pred, face_id, rot_mat, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i % 500 == 0:
                print("epoch: {}".format(str(i).zfill(6)), round(loss.item(), 6), round(optimizer.state_dict()['param_groups'][0]['lr'], 8))

            if (i + 1) % opt_parser.ckpt_epoch_freq == 0:
                if not os.path.exists(opt_parser.ckpt_save_dir):
                    os.makedirs(opt_parser.ckpt_save_dir)
                torch.save(C.state_dict(), os.path.join(opt_parser.ckpt_save_dir, '{}_C.pth'.format(str(i + 1).zfill(5))))
                torch.save(G.state_dict(), os.path.join(opt_parser.ckpt_save_dir, '{}_G.pth'.format(str(i + 1).zfill(5))))
                if best_loss > loss.item():
                    best_loss = loss.item()
                    torch.save(C.state_dict(), os.path.join(opt_parser.ckpt_save_dir, 'best_loss_{}_C.pth'.format(str(round(best_loss, 6)))))
                    torch.save(G.state_dict(), os.path.join(opt_parser.ckpt_save_dir, 'best_loss_{}_G.pth'.format(str(round(best_loss, 6)))))

def loss_vae_fn(recon_x, x, mean, log_var):
    MAE = torch.nn.functional.l1_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (MAE + KLD) / x.size(0)

def train_pose_cvae(device):
    """
    其实这里要先训练C之后再加载训练更好的C来训练G的
    这里我为了简单方便, 就是一起train的, 之后可以对比看看效果如何
    Args:
        device:
    Returns:
    """

    # condition vae
    vae = VAE(
        latent_size=opt_parser.latent_size,
        audio_feat_size=opt_parser.audio_feat_size,
        c_enc_hidden_size=opt_parser.c_enc_hidden_size,
        num_layers=opt_parser.num_layers,
        proj_dim=opt_parser.proj_dim,
        drop_out=opt_parser.cvae_drop_out
        ).to(opt_parser.device)

    # 这个是固定住不训练的
    speaker_aware = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).to(opt_parser.device)

    # 只训练vae
    optimizer = optim.Adam(vae.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

    lr_scheduler = get_scheduler(optimizer, opt_parser)
    # https://blog.csdn.net/qq_38964360/article/details/126330336
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=500, after_scheduler=lr_scheduler)

    train_data = Audio2headpose_Dataset(num_window_frames=opt_parser.num_window_frames,
                                        num_emb_window_frames=opt_parser.num_emb_window_frames,
                                        num_size=opt_parser.num_size)
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=opt_parser.batch_size, # 这里的batchsize是选取几段wav的意思, 不是通常的那种batchsize
                                                   shuffle=False, num_workers=0,
                                                   collate_fn=train_data.my_collate_in_segments)
    best_loss = float("inf")
    for i in range(opt_parser.nepoch):
        for batch in train_dataloader:
            """
            train_dataloader中的batch的个数是与参与构建dataset的wav个数有关的
            一个wav是同一个人在一段时间说的一段话, 
            """
            fls, aus, emb_aus, std, fc, fc_std = batch
            fls, aus, emb_aus, std, fc, fc_std = fls.to(device), aus.to(device), emb_aus.to(device), std.to(device), fc.to(device), fc_std.to(device)

            emb = speaker_aware(emb_aus)

            gt_fc = fc[:, -1, :]
            gt_fc_std = fc_std[:, -1, :]

            gt_fc_diff = gt_fc - gt_fc_std

            fc_dis_pred, mean, log_var = vae(gt_fc_diff, aus, emb)

            # 加上属于这个人的残差偏移
            fc_pred = gt_fc_std + fc_dis_pred

            loss = loss_vae_fn(fc_pred, gt_fc, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i % 500 == 0:
                print("epoch: {}".format(str(i).zfill(6)), round(loss.item(), 6), round(optimizer.state_dict()['param_groups'][0]['lr'], 8))

            if (i + 1) % opt_parser.ckpt_epoch_freq == 0:
                if not os.path.exists(opt_parser.ckpt_save_dir):
                    os.makedirs(opt_parser.ckpt_save_dir)
                torch.save(vae.state_dict(), os.path.join(opt_parser.ckpt_save_dir, '{}_vae.pth'.format(str(i + 1).zfill(5))))
                if best_loss > loss.item():
                    best_loss = loss.item()
                    torch.save(vae.state_dict(), os.path.join(opt_parser.ckpt_save_dir, 'best_loss_{}_vae.pth'.format(str(round(best_loss, 6)))))

if __name__ == "__main__":
    device = "cuda"
    # train_pose_cvae(device)
    train_audio2landmark(device)