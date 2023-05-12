from torch import nn
from .encoder import EncoderApp, EqualLinear
from .styledecoder import Synthesis

# class Generator(nn.Module):
#     def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
#         super(Generator, self).__init__()
#
#         # encoder
#         self.enc = Encoder(size, style_dim, motion_dim)
#         self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)
#
#     def get_direction(self):
#         return self.dec.direction(None)
#
#     def synthesis(self, wa, alpha, feat):
#         img = self.dec(wa, alpha, feat)
#
#         return img
#
#     def forward(self, img_source, img_drive, h_start=None):
#         wa, alpha, feats = self.enc(img_source, img_drive, h_start)
#         img_recon = self.dec(wa, alpha, feats)
#
#         return img_recon


class Generator(nn.Module):
    """
    其实这个是LIA总网络，包含了encoder和decoder
    需要将Exp加在encoder中
    """
    def __init__(self, size, style_dim=512, landmarks_dim=204, motion_dim=20, channel_multiplier=1,
                 blur_kernel=[1, 3, 3, 1], device=None):
        super(Generator, self).__init__()

        # encoder
        # self.enc = Encoder(size, style_dim, motion_dim)
        """
        这个部分对应着网络中E部分, 会生成一些enc供之后使用
        """
        self.enc = EncoderApp(size, style_dim)

        # motion network
        fc = [EqualLinear(landmarks_dim, landmarks_dim)]

        for i in range(3):
            fc.append(EqualLinear(landmarks_dim, landmarks_dim))

        fc.append(EqualLinear(landmarks_dim, motion_dim))
        self.fc = nn.Sequential(*fc)

        # decoder
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier, device=device)

    def decode(self, wa, feats, landmarks_target):
        h_motion_target = self.fc(landmarks_target)
        img_recon = self.dec(wa, [h_motion_target], feats)
        return img_recon

    def forward(self, img_source, landmarks_target, exp_latents=None):
        """
        img_source: 被用于驱动的原图
        landmarks_target: driving face landmark  (batch, 136)
        """
        # wa, alpha, feats = self.enc(img_source, img_drive)
        # image encoding

        wa, feats = self.enc(img_source, exp_latents)  # wa.shape = (batch, 512)
        """
        feats_i(shape):
            torch.Size([2, 512, 8, 8])
            torch.Size([2, 512, 16, 16])
            torch.Size([2, 512, 32, 32])
            torch.Size([2, 256, 64, 64])
            torch.Size([2, 128, 128, 128])
            torch.Size([2, 64, 256, 256])
        """

        # mapping landmarks into latent space (Ar->d)
        # 这里得到一个运动向量
        h_motion_target = self.fc(landmarks_target)  # (batch, 128)

        h_motion = [h_motion_target]

        # decode, image generator
        img_recon = self.dec(wa, h_motion, feats)

        return img_recon
