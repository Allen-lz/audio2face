#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
import sys
sys.path.append(".")
sys.path.append("..")
from models.face_parsing.model import BiSeNet
import torch
import matplotlib.pylab as plt
import os
from PIL import Image
import torchvision.transforms as transforms
import cv2

class FaceParsing:
    def __init__(self, weight):
        self.net = BiSeNet(n_classes=19)
        self.device = 'cuda:0'
        # self.device = 'cpu'
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(weight))
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(self, img_cv):
        with torch.no_grad():
            # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            h, w = img_cv.shape[:2]

            # cv2.imshow("show", img_cv)
            # cv2.waitKey()

            img = Image.fromarray(img_cv)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            parsing[parsing != 0] = 255

            parsing = np.array(parsing, np.uint8)
            parsing = cv2.resize(parsing, (w, h))

            # plt.subplot(131), plt.imshow(img_cv)
            # plt.subplot(132), plt.imshow(parsing)
            parsing = parsing / 255
            # fusion = np.array(parsing[:, :, np.newaxis] * img_cv, dtype=np.uint8)
            # plt.subplot(133), plt.imshow(fusion)
            # plt.show()

            # return parsing, eyes_mask, mouth_mask
            return parsing

if __name__ == "__main__":
    # 使用face_parse进行人脸解析
    faceparse_model_path = "checkpoints/face_parse.pth"
    faceparse = FaceParsing(weight=faceparse_model_path)
    img_dir = r"E:\datasets\ARKit\1\img"
    images = os.listdir(img_dir)
    for name in images:
        image_path = os.path.join(img_dir, name)
        img = cv2.imread(image_path)
        faceparse.forward(img)