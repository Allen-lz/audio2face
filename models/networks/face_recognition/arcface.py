import torch
import torch.nn.functional as F
import sys
import cv2
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from models.networks.face_recognition.nets.arcface import Arcface as arcface


class Arcface(object):
    _defaults = {
        "model_path": "checkpoints/arcface_mobilefacenet.pth",
        # -------------------------------------------#
        #   所使用到的主干特征提取网络，与训练的相同
        #   mobilefacenet
        #   mobilenetv1
        #   iresnet18
        #   iresnet34
        #   iresnet50
        #   iresnet100
        #   iresnet200
        # -------------------------------------------#
        "backbone": "mobilefacenet",
        # -------------------------------------------#
        #   是否进行不失真的resize
        # -------------------------------------------#
    }

    # ---------------------------------------------------#
    #   初始化Arcface
    # ---------------------------------------------------#
    def __init__(self, device="cuda", **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            # 类中的变量
            setattr(self, name, value)
        self.weight_load(device)
        self.device = device

    def weight_load(self, device):
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        self.net = arcface(backbone=self.backbone, mode="predict").eval().to(device)
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

    def transforms(self, image):
        image -= 0.5
        image /= 0.5
        return F.interpolate(image, size=(112, 112), mode='bicubic')

    def run(self, images):
        """
        Args:
            image: image是RGB的

        Returns:

        """

        images = self.transforms(images).to(self.device)
        id_latents = self.net(images)

        return id_latents

if __name__ == "__main__":
    arc_face = Arcface("cuda")
    image_path = "D:/audio2face/SadTalker/examples/source_image/sad1.png"
    image_RGB = torch.FloatTensor(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)).unsqueeze(0).permute(0, 3, 1, 2)
    id_latents = arc_face.run(image_RGB)

    print(id_latents)