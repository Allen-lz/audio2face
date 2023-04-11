# -*-coding:utf-8 -*-
"""
# File   : train_image2image.py
# Time   : 2023/4/7 14:41
# Author : luzeng
# version: python 3.7
"""

import os
import random
import logging
import datetime
from collections import defaultdict
import skimage.io
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from models.networks.landmark2face_network import UNet
from models.datasets.Landmark2face import ImageTranslationDataset
from models.networks.landmark2face_network import VGGLoss, define_D, GANLoss


# 日志的格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-9s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_g', type=float, default=3e-4, help='learning rate')

parser.add_argument('--epoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--use_discriminator', type=int, default=1, help='')

parser.add_argument('--gan_g_loss_weight', type=float, default=0.01, help='')
parser.add_argument('--feature_loss_weight', type=float, default=1.0, help='')

# model save
parser.add_argument('--ckpt_epoch_freq', type=int, default=200, help='')
parser.add_argument('--ckpt_save_dir', type=str, default="checkpoints/landmark2face", help='')

# dataset save
# /data/xujiajian/dataset/face_images_eval_3
# /data/xujiajian/dataset/face_images_3
parser.add_argument('--train_data_dir', type=str, default="/data/xujiajian/dataset/face_images_eval_3", help='')
parser.add_argument('--eval_data_dir', type=str, default="/data/xujiajian/dataset/face_images_eval_3", help='')
# parser.add_argument('--train_data_dir', type=str, default="D:/datasets/audio2face/face_images_eval_3", help='')
# parser.add_argument('--eval_data_dir', type=str, default="D:/datasets/audio2face/face_images_eval_3", help='')
parser.add_argument('--device', type=str, default="cuda:0", help='')
parser.add_argument('--max_num_per_video', type=int, default=2000, help='')

opt_parser = parser.parse_args()

class ImageSaver:
    def __init__(self, save_dir, dump_epoch_interval=10):
        self.save_dir = save_dir
        self.dump_epoch_interval = dump_epoch_interval
        self.last_save_epoch = -1
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save(self, src_img, gen_img, dst_img, epoch=0, global_step=0):
        if (self.last_save_epoch == -1 or epoch >= (self.last_save_epoch + self.dump_epoch_interval)) \
                and random.random() < 0.01:
            src_img_np = src_img[0].cpu().detach().numpy().transpose((1, 2, 0))
            gen_img_np = gen_img[0].cpu().detach().numpy().transpose((1, 2, 0))  # 这个是生成的图片
            dst_img_np = dst_img[0].cpu().detach().numpy().transpose((1, 2, 0))

            save_img = np.concatenate([
                np.concatenate([src_img_np, gen_img_np], axis=1),
                np.concatenate([dst_img_np, dst_img_np], axis=1)
            ], axis=0)

            save_img[save_img < 0] = 0
            save_img[save_img > 1] = 1
            save_img = (save_img * 255.0).astype(np.uint8).copy()
            skimage.io.imsave(
                os.path.join(self.save_dir, "ckpt_ep{:05d}_step{:08d}_img.jpg".format(epoch, global_step)), save_img)
            self.last_save_epoch = epoch

def train(opt):

    train_data_dir = opt.train_data_dir
    eval_data_dir = opt.eval_data_dir
    ckpt_root_dir = opt.ckpt_save_dir

    DEVICE = opt.device
    MAX_NUM_PER_VIDEO = opt.max_num_per_video
    EPOCH = opt.epoch
    BATCH_SIZE = opt.batch_size
    USE_DISCRIMINATOR = True if opt.use_discriminator == 1 else False
    LEARNING_RATE_G = opt.lr_g
    LEARNING_RATE_D = opt.lr_d
    GAN_G_LOSS_WEIGHT = opt.gan_g_loss_weight
    FEATURE_LOSS_WEIGHT = opt.feature_loss_weight

    device = torch.device(DEVICE)

    # 创建训练集
    train_dataset = ImageTranslationDataset(train_data_dir, max_num_per_video=MAX_NUM_PER_VIDEO)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=train_dataset.collate)

    # 创建测试集
    eval_dataset = ImageTranslationDataset(eval_data_dir, max_num_per_video=MAX_NUM_PER_VIDEO)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=eval_dataset.collate)

    # UNET生成模型
    model = UNet(n_channels=3, n_classes=3)
    model.to(device)
    # 加载预训练模型
    model.load_state_dict(
        torch.load("checkpoints/landmark2face/i2i_epoch_100_20221207183511/model.pt"))
    model.to(device)

    # 判别器
    if USE_DISCRIMINATOR:
        # 创建判别器(通过对权重大小的分析, 我复现了网络结构)
        discriminator = define_D(6, 16, 6, num_D=1, use_sigmoid=True)
        discriminator.to(device)

    # 损失函数\优化器
    l1_loss_func = torch.nn.L1Loss(reduction="none")
    l2_loss_func = torch.nn.MSELoss(reduction="none")
    # 这个就是很平常的感知损失
    vgg_loss_func = VGGLoss()
    vgg_loss_func.to(device)

    # 这里没有使用学习率下降策略, 到时候可以加上看看
    optimizer_G = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    if USE_DISCRIMINATOR:
        # 这个就是一个判别器的loss, 我一般不会去改
        gan_loss_func = GANLoss(use_lsgan=False, device=device)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=LEARNING_RATE_D)

    # 时间戳，用来命名ckpt
    current_datetime_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    ckpt_dir = os.path.join(ckpt_root_dir, "i2i_epoch_" + str(EPOCH) + "_" + str(current_datetime_str))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # tensorboardX训练日志
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "log"))
    img_saver = ImageSaver(os.path.join(ckpt_dir, "images"), dump_epoch_interval=0)

    def calculate_output_and_loss(batch_data, model):
        image_in, src_fl, dst_fl, image_out, image_weight, _margin, output_mask = batch_data
        image_in, src_fl, dst_fl, image_out, image_weight, output_mask = image_in.to(device), src_fl.to(device), \
                                                                         dst_fl.to(device), image_out.to(device), \
                                                                         image_weight.to(device), output_mask.to(device)

        tmp_pred_image = model(image_in, src_fl, dst_fl)
        pred_image = tmp_pred_image.clamp(0., 1.) * output_mask.detach()  # pred
        image_out *= output_mask                                          # target 这里乘上了嘴部下方的权重

        loss_dict = {}

        if discriminator is not None:
            # GAN损失
            # 假图检测，损失
            # 这里加一个pred_image.detach()是这里不涉及到生成器G的优化, 仅仅是优化判别器
            pred_fake = discriminator(torch.cat((image_in, pred_image.detach()), dim=1))
            loss_D_fake = gan_loss_func(pred_fake, False)
            # 真图检测, 损失
            pred_real = discriminator(torch.cat((image_in, image_out), dim=1))
            loss_D_real = gan_loss_func(pred_real, True)
            # 真图的权重要更高一点, 这里写的是2
            loss_D = (loss_D_real * 2 + loss_D_fake) / 2

            # 用于优化生成器的GAN损失, 这个应该是生成器的fool loss, 不属于判别器的
            pred_fake_G = discriminator(torch.cat((image_in, pred_image), dim=1))
            loss_G_GAN = gan_loss_func(pred_fake_G, True)

            loss_dict["loss_D_fake"] = loss_D_fake
            loss_dict["loss_D_real"] = loss_D_real
            loss_dict["loss_D"] = loss_D
            loss_dict["loss_G_GAN"] = loss_G_GAN

        # 使用l1和l2来同时优化, 这里还对loss进行了归一化
        # 这里只对面部的下方进行一个loss, 并且加重了嘴部的loss, 到时候估计是想生成面部下方的让后直接贴过去
        # 头部的姿态或者眨眼就靠剪辑, 推理的时候直接使用当前帧, 换一个嘴就行
        # 非主要部分的权重只有0.1
        l1_loss = (l1_loss_func(pred_image, image_out) * image_weight).sum() / image_weight.sum()
        l2_loss = (l2_loss_func(pred_image, image_out) * image_weight).sum() / image_weight.sum()
        vgg_loss = vgg_loss_func(pred_image, image_out)

        # crop img vgg loss
        crop_vgg_loss = 0
        """
        pred_image: 预测出来的图片
        image_out: target
        _margin: 是嘴部区域的边框
        """
        # 计算每张图的嘴部loss
        for pred_img, dst_img, _m, _w in zip(pred_image, image_out, _margin, image_weight):
            pred_crop_batch = pred_img[:, _m[0]:_m[1], _m[2]: _m[3]].unsqueeze(0).repeat((2, 1, 1, 1))
            dst_crop_batch = dst_img[:, _m[0]:_m[1], _m[2]: _m[3]].unsqueeze(0).repeat((2, 1, 1, 1))
            mask = torch.where(_w[:, _m[0]:_m[1], _m[2]: _m[3]] == 10, 1, 0)  # 抠出牙齿的部分

            # 这里的1是因为要去除batch的那个维度,
            pred_crop_batch[1] *= mask
            dst_crop_batch[1] *= mask
            _single_crop_vgg_loss = vgg_loss_func(pred_crop_batch, dst_crop_batch)
            crop_vgg_loss += _single_crop_vgg_loss
        crop_vgg_loss /= image_out.size(0)

        loss_dict["l1_loss"] = l1_loss
        loss_dict["l2_loss"] = l2_loss
        loss_dict["vgg_loss"] = vgg_loss
        loss_dict["crop_vgg_loss"] = crop_vgg_loss

        if discriminator is not None:
            generator_loss = l1_loss + vgg_loss * FEATURE_LOSS_WEIGHT + crop_vgg_loss * FEATURE_LOSS_WEIGHT + \
                             loss_G_GAN * GAN_G_LOSS_WEIGHT
            loss = generator_loss + loss_D
        else:
            generator_loss = l1_loss + vgg_loss * FEATURE_LOSS_WEIGHT + crop_vgg_loss * FEATURE_LOSS_WEIGHT
            loss = generator_loss

        loss_dict["generator_loss"] = generator_loss
        loss_dict["loss"] = loss

        return pred_image, loss_dict

    global_step = 0
    for ep in range(EPOCH):
        logging.info("Start Epoch: {}".format(ep))

        # 训练集
        model.train()
        for i, batch in enumerate(train_dataloader):
            pred_image, loss_dict = calculate_output_and_loss(batch, model)

            # backward
            optimizer_G.zero_grad()
            loss_dict["generator_loss"].backward()
            optimizer_G.step()
            if USE_DISCRIMINATOR:
                optimizer_D.zero_grad()
                loss_dict["loss_D"].backward()
                optimizer_D.step()

            image_in = batch[0]
            image_out = batch[3]
            img_saver.save(image_in, pred_image, image_out, epoch=ep, global_step=global_step)

            global_step += 1

            if i % opt_parser.ckpt_epoch_freq == 0:
                # 训练日志记录
                for key, val in loss_dict.items():
                    tensorboard_writer.add_scalar(key, val.cpu().detach().numpy(), global_step=global_step)

                logging.info("L1: {:.6f}, L2: {:.6f}, VGG: {:.6f}, Crop VGG: {:.6f}, "
                             "G_Gan: {:.6f}, D_GAN: {:.6f}, Loss: {:.6f}".format(
                                    loss_dict["l1_loss"].detach().cpu().numpy(),
                                    loss_dict["l2_loss"].detach().cpu().numpy(),
                                    loss_dict["vgg_loss"].detach().cpu().numpy(),
                                    loss_dict["crop_vgg_loss"].detach().cpu().numpy(),
                                    loss_dict["loss_G_GAN"].detach().cpu().numpy() if USE_DISCRIMINATOR else 0.,
                                    loss_dict["loss_D"].detach().cpu().numpy() if USE_DISCRIMINATOR else 0.,
                                    loss_dict["loss"].detach().cpu().numpy()))

                # 保存监测点checkpoint
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                if USE_DISCRIMINATOR:
                    torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "discriminator.pt"))

        # 验证集
        eval_loss_dict = defaultdict(float)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                pred_image, loss_dict = calculate_output_and_loss(batch, model)
                for key, val in loss_dict.items():
                    eval_loss_dict[key] += val.detach().cpu().numpy()

            logging.info(
                "\t\tEval Dataset: \tL1: {:.6f}, L2: {:.6f}, VGG: {:.6f}, Crop VGG: {:.6f}, "
                "G_Gan: {:.6f}, D_GAN: {:.6f}, Loss: {:.6f}".format(
                    eval_loss_dict["l1_loss"] / (i + 1),
                    eval_loss_dict["l2_loss"] / (i + 1),
                    eval_loss_dict["vgg_loss"] / (i + 1),
                    eval_loss_dict["crop_vgg_loss"] / (i + 1),
                    eval_loss_dict["loss_G_GAN"] / (i + 1),
                    eval_loss_dict["loss_D"] / (i + 1),
                    eval_loss_dict["loss"] / (i + 1))
            )

        # 保存监测点checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        if USE_DISCRIMINATOR:
            torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "discriminator.pt"))

if __name__ == '__main__':
    train(opt_parser)
