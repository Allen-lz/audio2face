import torch
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from models.networks.expression.models.mobilenetv3 import *
from models.networks.expression.configs import cfg
import cv2
import os
class FaceDrive():
    def __init__(self, device='cuda:0'):
        self.count = 0
        self.device = device
        self.net = mobilenetv3_small(num_classes=cfg.num_classes, width_mult=cfg.width_mult, output_c=cfg.output_c).to(self.device)
        # 加载权重
        ckpt = torch.load(cfg.eval_ckpt)
        self.net.load_state_dict(ckpt['state_dict'], strict=True)
        self.net.eval()

        self.trochResize = torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

    def run(self, img_rgb_tenor):
        """

        Args:
            img_rgb_tenor:

        Returns:

        """
        img = img_rgb_tenor.permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        pre = self.net.forward(img)
        exp = pre[0][0]
        # lm = pre[2][0]  # 人脸的5个关键点
        # pos = pre[1][0]
        return exp

    def run_batch(self, img_rgb_tenor):
        """

        Args:
            img_rgb_tenor:

        Returns:

        """
        img_rgb_tenor = self.trochResize(img_rgb_tenor)
        img = img_rgb_tenor.to(self.device)
        pre = self.net.forward(img)
        exp = pre[0]
        return exp


if __name__ == "__main__":

    faceDrive = FaceDrive()

    image_dir = "E:/audio2face/autovc/generate_face/Av416828403-P1"
    image_list = os.listdir(image_dir)
    for name in image_list:
        image_path = os.path.join(image_dir, name)
        image_RGB = torch.FloatTensor(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        faceDrive.run(image_RGB)








