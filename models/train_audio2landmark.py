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
from models.datasets.Audio2landmark import Audio2landmark_Dataset
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

# model save
parser.add_argument('--jpg_freq', type=int, default=1, help='')
parser.add_argument('--ckpt_epoch_freq', type=int, default=1000, help='')
parser.add_argument('--ckpt_save_dir', type=str, default="checkpoints/audio2landmark", help='')

opt_parser = parser.parse_args()


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




def train(device):
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

    G = Audio2landmark_speaker_aware().to(device)

    optimizer = optim.Adam(list(C.parameters()) + list(G.parameters()), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

    # optimizer = torch.optim.AdamW(list(C.parameters()) + list(G.parameters()), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

    lr_scheduler = get_scheduler(optimizer, opt_parser)
    # https://blog.csdn.net/qq_38964360/article/details/126330336
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=500, after_scheduler=lr_scheduler)

    train_data = Audio2landmark_Dataset(num_window_frames=opt_parser.num_window_frames,
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
            fls, aus, emb_aus, std = batch
            fls, aus, emb_aus, std = fls.to(device), aus.to(device), emb_aus.to(device), std.to(device)
            std = std[:, 0, :]
            fl_dis_pred, face_id = C(aus, std)

            emb = speaker_aware(emb_aus)
            emb_fl_dis_pred = G(aus, emb)

            # 加上属于这个人的残差偏移
            fl_dis_pred += emb_fl_dis_pred

            loss = calculate_loss(fls, fl_dis_pred, face_id, device)

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

if __name__ == "__main__":
    device = "cuda"
    train(device)