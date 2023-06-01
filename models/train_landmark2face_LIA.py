# -*-coding:utf-8 -*-

"""
# File   : train_image2image2.py
# Time   : 2023/4/11 17:10
# Author : luzeng
# version: python 3.9

conda activate automatic
python models/train_landmark2face_LIA.py
"""

import os
import random
import logging
import datetime
import re

import torch
import skimage.io
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from models.networks.landmarks_LIA.vgg19 import VGGLoss
from models.networks.landmarks_LIA.generator import Generator
from models.networks.landmarks_LIA.discriminator import Discriminator

import matplotlib.pyplot as plt

from models.datasets.Landmark2face_LIA import ImageTranslationDatasetForLIA, vis_landmark_on_img

from collections import namedtuple
from models.networks.mobile_net_v3_small.mbv3_w_euler_score_sep import MobileNet_V3_Small_Euler_Score

from models.networks.face_recognition.arcface import Arcface

import cv2
import tqdm
import skvideo.io
import mediapipe as mp
from models._3dmm.inference_3dmm import FaceReconBFM
from models.networks.expression.face_drive import FaceDrive
import imageio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-9s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")



parser = argparse.ArgumentParser()
parser.add_argument('--lr_d', type=float, default=2e-5, help='learning rate')
parser.add_argument('--lr_g', type=float, default=2e-5, help='learning rate')

parser.add_argument('--epoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')

parser.add_argument('--gan_g_loss_weight', type=float, default=0.05, help='')
parser.add_argument('--landmark_loss_weight', type=float, default=1, help='')

# model save
parser.add_argument('--print_step_freq', type=int, default=200, help='')
parser.add_argument('--ckpt_save_dir', type=str, default="checkpoints/landmark2face", help='')

# dataset save
# /data/xujiajian/dataset/face_images_eval_3
# /data/xujiajian/dataset/face_images_3
# /data/xujiajian/dataset/face_images_5/
# D:/datasets/audio2face/face_images_5
# /home/luzeng/datasets/audio2face/face_images_5
parser.add_argument('--train_data_dir', type=str, default="D:/datasets/audio2face/face_images_5", help='')
parser.add_argument('--eval_data_dir', type=str, default="/data/xujiajian/dataset/face_images_eval_3", help='')

# /home/luzeng/datasets/expressions/
# D:/datasets/ARKit/zhf_1/img
parser.add_argument('--train_exp_data_dir', type=str, default="D:/datasets/ARKit/zhf_1/img", help='')

parser.add_argument('--pretrained_model', type=str, default="checkpoints/model.pt", help='')
parser.add_argument('--discriminator_ckpt', type=str, default="checkpoints/discriminator.pt", help='')
parser.add_argument('--landmark_ckpt', type=str, default="checkpoints/model_best.pt", help='')

parser.add_argument('--img_size', type=int, default=256, help='')
parser.add_argument('--latent_dim_style', type=int, default=512, help='')
parser.add_argument('--landmarks_dim', type=int, default=136, help='')
parser.add_argument('--latent_dim_motion', type=int, default=128, help='')
parser.add_argument('--device', type=str, default="cuda:0", help='')
parser.add_argument('--phrase', type=str, default="train", help='')

opt_parser = parser.parse_args()

def crop_face(image, detector=None):
    max_rect = np.array([0, 0, 0, 0])
    results = detector.process(image.copy())
    if results is None or results.detections is None:
        return max_rect

    max_area = 0
    img_width = image.shape[1]
    img_height = image.shape[0]

    # 看似是取得最大的人脸区域, 其实只是得到第一个face_boox
    for detection in results.detections:
        rect_obj = detection.location_data.relative_bounding_box
        area = rect_obj.width * rect_obj.height
        if area > max_area:
            max_rect = np.array([rect_obj.xmin * img_width,
                                 rect_obj.ymin * img_height,
                                 (rect_obj.xmin + rect_obj.width) * img_width,
                                 (rect_obj.ymin + rect_obj.height) * img_height])

    return max_rect



class ImageSaver:
    def __init__(self, save_dir, dump_epoch_interval=10):
        self.save_dir = save_dir
        self.dump_epoch_interval = dump_epoch_interval
        self.last_save_epoch = -1
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save(self, src_img, gen_img, dst_img, img_exp_ref, epoch=0, global_step=0):
        if (self.last_save_epoch == -1 or epoch >= (self.last_save_epoch + self.dump_epoch_interval)) \
                and random.random() < 0.01:
            src_img_np = src_img[0].cpu().detach().numpy().transpose((1, 2, 0))
            gen_img_np = gen_img[0].cpu().detach().numpy().transpose((1, 2, 0))
            dst_img_np = dst_img[0].cpu().detach().numpy().transpose((1, 2, 0))
            img_exp_ref_np = img_exp_ref[0].cpu().detach().numpy().transpose((1, 2, 0))
            save_img = np.concatenate([src_img_np, gen_img_np, dst_img_np, img_exp_ref_np], axis=1)

            save_img[save_img < -1] = -1
            save_img[save_img > 1] = 1
            save_img = ((save_img / 2 + 0.5) * 255.0).astype(np.uint8).copy()
            skimage.io.imsave(
                os.path.join(self.save_dir, "ckpt_ep{:05d}_step{:08d}_img.jpg".format(epoch, global_step)), save_img)
            self.last_save_epoch = epoch


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()


def d_nonsaturating_loss(fake_pred, real_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def cal_crop_vgg_loss(img_gen, img_real, margin, loss_func):
    """
    margin: 是嘴部的区域
    """
    crop_vgg_loss = 0
    wrong_num = 0

    # 因为每个人脸的嘴部区域大小是不一样的, 所以这里用了训练来计算嘴部的loss
    # 但是这里其实完全可以resize到一个固定的大小, 然后再一整个batch进行loss计算的
    for pred_img, dst_img, _m in zip(img_gen, img_real, margin):
        pred_crop_batch = pred_img[:, _m[0]:_m[1], _m[2]: _m[3]].unsqueeze(0)
        dst_crop_batch = dst_img[:, _m[0]:_m[1], _m[2]: _m[3]].unsqueeze(0)
        try:
            _single_crop_vgg_loss = loss_func(pred_crop_batch, dst_crop_batch, num_scales=2)
            crop_vgg_loss += _single_crop_vgg_loss
        except:
            wrong_num += 1
    # 计算loss的均值
    crop_vgg_loss /= (img_real.size(0) - wrong_num + 1e-16)
    return crop_vgg_loss

# conda activate automatic
# python models/train_landmark2face_LIA.py
def cosin_metric(x1, x2, weight=100):
    #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    # x1 = F.normalize(x1, p=2, dim=1)
    # x2 = F.normalize(x2, p=2, dim=1)

    # 计算
    loss = torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))
    loss_G_ID = (1 - loss).mean() * weight
    return loss_G_ID


class FaceLandmarksLoss(nn.Module):
    def __init__(self, device="cuda", model_ckpt=""):
        """
        这个就是简单的从从生成的图中得到lm并计算lm的损失
        """
        super(FaceLandmarksLoss, self).__init__()
        Options = namedtuple('Options', ["resume", "ckpt"])
        options = Options._make({"resume": True, "ckpt": ""})
        # options = Options._make([True, ""])

        self.model = MobileNet_V3_Small_Euler_Score(options)
        ckpt = {k: v for k, v in torch.load(model_ckpt).items() if k in self.model.state_dict()}
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.model.to(torch.device(device))
        requires_grad(self.model, False)

        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([transforms.Resize(160), normalize])
        self.features = self.model.mobilenet_v3_small.features
        self.lm_regressor = self.model.mobilenet_v3_small.lm_regressor

        self.weight = torch.ones((1, 180, 1, 1)).to(torch.device(device))
        self.weight[0, 60 * 2: 80 * 2] = 5  # 60到80点应该是嘴部的点, 所以权重是5

    def forward(self, img_gen, img_real):
        gen_fl_hm = self.lm_regressor(self.features(self.transformer((img_gen / 2) + 0.5)))
        real_fl_hm = self.lm_regressor(self.features(self.transformer((img_real / 2) + 0.5)))  # (2, 180, 1, 1)

        return ((gen_fl_hm - real_fl_hm.detach()).abs() * img_real.shape[2] * self.weight).sum() / self.weight.sum()


def train():

    train_data_dir = opt_parser.train_data_dir

    train_exp_data_dir = opt_parser.train_exp_data_dir   # <<<< 增加了一个表情参考图片数据

    pretrained_model = opt_parser.pretrained_model
    discriminator_ckpt = opt_parser.discriminator_ckpt
    landmark_ckpt = opt_parser.landmark_ckpt
    ckpt_save_dir = opt_parser.ckpt_save_dir

    IMG_SIZE = opt_parser.img_size
    LATENT_DIM_STYLE = opt_parser.latent_dim_style
    LANDMARKS_DIM = opt_parser.landmarks_dim
    LATENT_DIM_MOTION = opt_parser.latent_dim_motion

    DEVICE = opt_parser.device
    EPOCH = opt_parser.epoch
    BATCH_SIZE = opt_parser.batch_size  # 16
    LEARNING_RATE_G = opt_parser.lr_g
    LEARNING_RATE_D = opt_parser.lr_d
    LANDMARKS_LOSS_WEIGHT = opt_parser.landmark_loss_weight
    GAN_G_LOSS_WEIGHT = opt_parser.gan_g_loss_weight

    device = torch.device(DEVICE)

    # LIA生成模型
    generator = Generator(IMG_SIZE, LATENT_DIM_STYLE, LANDMARKS_DIM, LATENT_DIM_MOTION, 1, device=device)

    # 创建一个人脸重建的模块
    # face_recon = FaceReconBFM()

    # 创建一个人脸表情的提取器
    # 创建一个面部驱动器
    faceDrive = FaceDrive(device)

    # 创建一个人脸识别器
    arcFace = Arcface()

    # 预训练checkpoint
    # ckpt_dict = torch.load(pretrained_model)["gen"]
    # new_ckpt_dict = {}
    # for key, val in ckpt_dict.items():
    #     if "enc.net_app" in key:
    #         key = key.replace(".net_app", "")
    #     elif "enc.fc" in key:
    #         continue
    #     elif "direction" in key:
    #         continue
    #     new_ckpt_dict[key] = val
    new_ckpt_dict = torch.load(pretrained_model)  # 加载预训练模型
    generator.load_state_dict(new_ckpt_dict, strict=False)
    generator.to(device)

    # 判别器
    discriminator = Discriminator(IMG_SIZE, 1).to(device)
    if discriminator_ckpt:
        discriminator.load_state_dict(torch.load(discriminator_ckpt), strict=False)

    # 数据集
    # train_dataset = ImageTranslationDatasetForLIA(
    #     train_data_dir, max_num_per_video=MAX_NUM_PER_VIDEO, img_size=(IMG_SIZE, IMG_SIZE), noise_prob=0.)
    train_dataset = ImageTranslationDatasetForLIA(train_data_dir, train_exp_data_dir, img_size=(IMG_SIZE, IMG_SIZE), noise_prob=0.)
    print("Video Chunks number:", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    # eval_dataset = ImageTranslationDataset3(eval_data_dir, max_num_per_video=MAX_NUM_PER_VIDEO, shrink=shrink,
    #                                         half_size=half_size)
    # eval_dataloader = DataLoader(
    #     eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 损失函数、优化器
    l1_loss_func = torch.nn.L1Loss(reduction="none")
    mse_loss_func = torch.nn.MSELoss(reduction="none")

    # exp loss
    exp_loss_func = cosin_metric

    # id loss
    id_loss_func = cosin_metric

    vgg_loss_func = VGGLoss().to(device)
    landmarks_loss_func = FaceLandmarksLoss(DEVICE, landmark_ckpt)

    # train_parameters = [v for k, v in model.named_parameters() if "fc" in k]
    # train_parameters = [v for k, v in model.named_parameters() if "fc" in k or "direction" in k]
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0., 0.99))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0., 0.99))

    # 时间戳，用来命名ckpt
    current_datetime_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    ckpt_dir = os.path.join(ckpt_save_dir, "i2i_epoch_" + str(EPOCH) + "_" + str(current_datetime_str))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # tensorboard、训练日志
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "log"))
    img_saver = ImageSaver(os.path.join(ckpt_dir, "images"), dump_epoch_interval=0)

    global_step = 0
    for ep in range(EPOCH):
        # logging.info("Start Epoch: {}".format(ep))

        # 训练集
        generator.train()
        for i, batch in enumerate(train_dataloader):
            img_source, fl_target, img_target, image_weight, _margin, _exp_ref = batch
            img_source, fl_target, img_target, image_weight, img_exp_ref = img_source.to(device), fl_target.float().to(device), img_target.to(device), image_weight.to(device), _exp_ref.to(device)

            # ==========================================================================================================
            # 训练生成器
            generator.zero_grad()
            optimizer_G.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            # 使用面部驱动器得到exp_latents, 然后将latent送入到encoder中
            # 这里的输入暂时使用source代替
            # ============================================================================
            face_drive_ref = img_exp_ref / 2 + 0.5
            exp_ref_latents = faceDrive.run_batch(face_drive_ref)  # (b, 52)

            face_drive_src = img_source / 2 + 0.5
            exp_src_latents = faceDrive.run_batch(face_drive_src)  # (b, 52)
            # ============================================================================


            img_target_recon = generator(img_source, fl_target, exp_ref_latents, exp_src_latents)


            # ============================================================================
            face_drive_recon = img_target_recon / 2 + 0.5
            exp_recon_latents = faceDrive.run_batch(face_drive_recon)  # (b, 52)
            # ============================================================================
            exp_loss = exp_loss_func(exp_ref_latents, exp_recon_latents)

            # =============================== id loss的计算 ================================

            id_source_img = img_source / 2 + 0.5
            id_recon_img = img_target_recon / 2 + 0.5
            source_id = arcFace.run(id_source_img)
            target_id = arcFace.run(id_recon_img)
            id_loss = id_loss_func(source_id, target_id)

            # ==============================================================================

            img_recon_pred, img_recon_feats = discriminator(img_target_recon, True)
            _, real_img_feats = discriminator(img_target, True)

            fm_loss = 0
            for img_recon_feat, real_img_feat in zip(img_recon_feats, real_img_feats):
                assert img_recon_feat.shape == real_img_feat.shape
                fm_loss += l1_loss_func(img_recon_feat, real_img_feat).sum(-1).sum(-1).mean() / 10

            if global_step % 2 == 0:
                # 这里我亲一色的乘上一个0.1
                l1_loss = (l1_loss_func(img_target_recon, img_target) * image_weight).sum() / image_weight.sum() * 0.1
                fl_loss = landmarks_loss_func(img_target_recon, img_target) * LANDMARKS_LOSS_WEIGHT * 0.1
                vgg_loss = vgg_loss_func(img_target_recon, img_target).mean() * 0.1
                crop_vgg_loss = cal_crop_vgg_loss(img_target_recon, img_target, _margin, vgg_loss_func) * 0.1
            else:
                l1_loss = 0
                fl_loss = 0
                vgg_loss = 0
                crop_vgg_loss = 0

            gan_g_loss = g_nonsaturating_loss(img_recon_pred) * GAN_G_LOSS_WEIGHT  # 这是一个愚弄loss

            # ========================= 增加一个loss, 使用重建出来的几千个点来做loss, 我认为这个loss应该放在g_loss上 ========================

            # face_proj_recon = (img_target_recon / 2 + 0.5) * 255
            # face_proj_target = (img_target / 2 + 0.5) * 255
            # # tensor rgb ---> tensor bgr
            # face_proj_recon = face_proj_recon[:, [2, 1, 0], :, ]
            # face_proj_target = face_proj_target[:, [2, 1, 0], :, ]
            #
            # recon_proj_vex = face_recon.run_tensor(face_proj_recon.permute(0, 2, 3, 1)) / 224  # 因为人脸重建的输入的尺度就是224, 是固定的
            # target_proj_vex = face_recon.run_tensor(face_proj_target.permute(0, 2, 3, 1)) / 224
            # vex_l1_loss = l1_loss_func(recon_proj_vex, target_proj_vex)
            # vex_mse_loss = mse_loss_func(recon_proj_vex, target_proj_vex)
            #
            # vex_l1_loss = vex_l1_loss.mean(-1).sum(-1).mean(0)
            # vex_mse_loss = vex_mse_loss.mean(-1).sum(-1).mean(0)
            #
            # recon_vex_loss = (vex_l1_loss + vex_mse_loss) / 2


            # =========================================================================================================
            # vis_face_proj_recon = np.array(face_proj_recon[0].permute(1, 2, 0).detach().cpu(), dtype=np.uint8)
            # vis_face_proj_target = np.array(face_proj_target[0].permute(1, 2, 0).detach().cpu(), dtype=np.uint8)
            # plt.subplot(121), plt.imshow(vis_face_proj_target)
            # plt.subplot(122), plt.imshow(vis_face_proj_recon)
            # plt.show()
            # =========================================================================================================

            g_loss = vgg_loss + crop_vgg_loss + l1_loss + fl_loss + gan_g_loss + exp_loss + id_loss + fm_loss

            g_loss.backward()
            optimizer_G.step()
            # ==========================================================================================================

            # ==========================================================================================================
            # 训练判别器
            discriminator.zero_grad()
            optimizer_D.zero_grad()

            requires_grad(generator, False)
            requires_grad(discriminator, True)

            real_img_pred = discriminator(img_target)
            recon_img_pred = discriminator(img_target_recon.detach())

            d_loss = d_nonsaturating_loss(recon_img_pred, real_img_pred)
            d_loss.backward()
            optimizer_D.step()
            # ==========================================================================================================
            # conda activate automatic
            # python models/train_landmark2face_LIA.py
            # ==========================================================================================================


            # 保存训练中间结果, 用于快速验证
            img_saver.save(img_source, img_target_recon, img_target, img_exp_ref, epoch=ep, global_step=global_step)

            if global_step % opt_parser.print_step_freq == 0:
                # 训练日志记录
                tensorboard_writer.add_scalar("L1", l1_loss.cpu().detach().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("VGG", vgg_loss.cpu().detach().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("CROP_VGG", crop_vgg_loss, global_step=global_step)
                tensorboard_writer.add_scalar("FL", fl_loss.detach().cpu().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("G_Gan", gan_g_loss.cpu().detach().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("D_GAN", d_loss.cpu().detach().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("Loss", (g_loss + d_loss).cpu().detach().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("Exp_Loss", (exp_loss).cpu().detach().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("Id_Loss", (id_loss).cpu().detach().numpy(), global_step=global_step)
                tensorboard_writer.add_scalar("FM_Loss", (fm_loss).cpu().detach().numpy(), global_step=global_step)

                for key, val in generator.state_dict().items():
                    if re.search(r"(to_flow\.5)|(to_rgbs\.5)|(convs\.10)|(direction)|(fc)|(enc\.convs\.4)", key):
                        tensorboard_writer.add_histogram(key, val, global_step=global_step)

                logging.info(
                    "Step: {:}, L1: {:.6f}, VGG: {:.6f}, Crop VGG: {:.6f}, FL: {:.6f}, G_Gan: {:.6f}, D_GAN: {:.6f}, Loss: {:.6f}, Exp_Loss: {:.6f} Id_Loss: {:.6f}, FM_Loss: {:.6f}".format(
                        global_step,
                        l1_loss.detach().cpu().numpy(),
                        vgg_loss.detach().cpu().numpy(),
                        crop_vgg_loss,
                        fl_loss.detach().cpu().numpy(),
                        gan_g_loss.detach().cpu().numpy(),
                        d_loss.detach().cpu().numpy(),
                        (g_loss + d_loss).detach().cpu().numpy(),
                        (exp_loss).cpu().detach().numpy(),
                        (id_loss).cpu().detach().numpy(),
                        (fm_loss).cpu().detach().numpy()
                    )
                )

                # 保存监测点checkpoint
                torch.save(generator.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "discriminator.pt"))

            global_step += 1

def make_gif(images):
    imageio.mimsave("examples/landmark2face.gif", images[:250], fps=25)   # 转化为gif动画

def test():
    ckpt = r"checkpoints/model.pt"
    test_video = r"E:/datasets/audio2face/cctv_short_video_bilibili_small/Av416828403-P1.mp4"
    num_frames = 1500
    device = torch.device("cuda")

    # 创建一个面部驱动器, 得到表情参考的exp_latents
    faceDrive = FaceDrive(device)

    # i2i模型
    model = Generator(256, 512, 136, 128, 1, device=device)

    model.load_state_dict(torch.load(ckpt), strict=False)
    model.to(device)
    model.eval()

    images = []

    # mediapipe关键点检测
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh_predict_model = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    face_68_landmarks_ind = [
        162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107,
        336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387,
        263, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87
    ]

    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    # 只计算嘴部landmark的loss
    weight = np.zeros((68, 2), dtype=np.float32)
    weight[48: 68] = 1

    def crop_face_image(img, smooth_rect=None):
        """
        Args:
            img:
            smooth_rect: 这是一个边框平滑项, 是按照视频的顺序积累下来的平滑边框, smooth_rect会不断地更新
        Returns:

        """
        # 得到人脸框
        rect = crop_face(img, face_detection)

        _smooth_rect = rect if smooth_rect is None else smooth_rect * 0.8 + rect * 0.2
        rect = _smooth_rect

        height = rect[3] - rect[1]
        width = rect[2] - rect[0]

        # 对边框进行1.3倍的外扩
        height = width = max(height, width) * 1.3

        mid_vertical = (rect[3] + rect[1]) / 2
        mid_horizontal = (rect[2] + rect[0]) / 2

        # 这里刻意减top上移了
        top = max(mid_vertical - height * 4 / 7, 0)
        left = max(mid_horizontal - width / 2, 0)

        face_img = img[int(top): int(top + height), int(left): int(left + width)]
        return face_img, _smooth_rect

    def landmarks_detect(img):
        predict_result = face_mesh_predict_model.process(img)
        if predict_result is None or predict_result.multi_face_landmarks is None:
            return None

        face_mesh_pts = np.array([
            [landmark.x, landmark.y, landmark.z]
            for idx, landmark in enumerate(predict_result.multi_face_landmarks[0].landmark)
        ])
        face_landmarks = face_mesh_pts[face_68_landmarks_ind]
        return face_landmarks

    def landmarks_detect_fa(img):
        kp = fa.get_landmarks(img)
        if kp is None:
            return None
        return kp[0]

    video_data = skvideo.io.vreader(test_video)
    src_img = cv2.resize(crop_face_image(video_data.__next__())[0], (256, 256))  # 取得视频的第一帧作为参考帧
    writer = skvideo.io.FFmpegWriter("test.mp4")  # 创建一个新的mp4文件
    with torch.no_grad():
        img_source = torch.from_numpy(
            (src_img.astype(np.float32).transpose((2, 0, 1))[np.newaxis] / 255.0 - 0.5) * 2.).to(device)
        # img_source = torch.from_numpy(src_img.astype(np.float32).transpose((2, 0, 1))[np.newaxis] / 255.0).to(device)

        # ========================= 对source img进行反解码 =======================
        # 在测试的时候最好传入一系列连贯的表情, 这样在表情方面的表现会更加自然
        face_drive_ref = img_source / 2 + 0.5
        exp_ref_latents = faceDrive.run_batch(face_drive_ref)  # (b ,52)
        # =======================================================================

        wa, feats = model.enc(img_source, exp_ref_latents)

        # wa是最后一层的特征
        # feats: 是前面几层的特征

        smooth_rect = None
        metrics = []

        for i, img in tqdm.tqdm(enumerate(video_data)):
            if -1 < num_frames <= i:
                break

            try:
                # 裁人脸框
                face_img, smooth_rect = crop_face_image(img, smooth_rect)
                face_img = cv2.resize(face_img, (256, 256))

                # 关键点检测
                face_landmarks = landmarks_detect(face_img)
                if face_landmarks is None:
                    continue

                fl_target = torch.from_numpy(face_landmarks[:, :2].reshape(-1)).unsqueeze(0).float().to(device)


                model_out = model.decode(wa, feats, fl_target)[0]
                gen_img = ((model_out.clamp(-1, 1).permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255).astype(
                    np.uint8).copy()
                # gen_img = (model_out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                # gen_landmarks = landmarks_detect(gen_img)
                # akd = np.abs(face_landmarks[..., :2] * 256 - gen_landmarks[..., :2] * 256).mean()
                # metrics.append(akd)

                # 检测gt的face landmark, 再检测得到生成的图片的face landmark
                ori_landmarks = landmarks_detect_fa(face_img)
                gen_landmarks = landmarks_detect_fa(gen_img)
                # akd = np.abs(ori_landmarks[..., :2] - gen_landmarks[..., :2]).mean()

                # 计算前两个维度的点的diff

                # ori_landmarks.shape = (68, 2)
                # gen_landmarks.shape = (68, 2)

                akd = (np.abs(ori_landmarks[..., :2] - gen_landmarks[..., :2]) * weight).sum() / weight.sum()
                metrics.append(akd)

                white = np.full_like(src_img, 255)
                vis_landmark_on_img(white, (face_landmarks * 256).astype(int))

                # vis_landmark_on_img2(gen_img, (face_landmarks * 256).astype(int), linewidth=1)
                _ = [cv2.circle(gen_img, (int(pt[0]), int(pt[1])), 1, (238, 130, 238), -1)
                     for pt in (face_landmarks * 256).astype(int)]

            except:
                pass

            show_img = np.concatenate([src_img, white, gen_img, face_img], axis=1)

            images.append(show_img)

            cv2.imshow("test", cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(5)
            writer.writeFrame(show_img)
    print(np.mean(metrics))
    writer.close()

    make_gif(images)


if __name__ == '__main__':
    print("opt_parser.phrase", opt_parser.phrase)

    if opt_parser.phrase == "test":
        test()
    else:
        train()