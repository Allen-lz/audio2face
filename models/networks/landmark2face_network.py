# -*-coding:utf-8 -*-

"""
# File   : landmark2face_network.py
# Time   : 2023/4/7 15:34
# Author : luzeng
# version: python 3.7
"""

import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.norm_layer = nn.InstanceNorm2d
        # 这个深度运动网络
        self.dense_motion_network = DenseMotionNetwork(
            num_kp=68, num_channels=3, estimate_occlusion_map=True, block_expansion=32, max_features=1024,
            num_blocks=3, scale_factor=1)
        # 输入层
        self.inc = DoubleConv(n_channels, 64, norm_layer=self.norm_layer)
        # 下采样
        self.down1 = Down(64, 128, norm_layer=self.norm_layer)
        self.down2 = Down(128, 256, norm_layer=self.norm_layer)
        self.down3 = Down(256, 512, norm_layer=self.norm_layer)
        self.down4 = Down(512, 1024, norm_layer=self.norm_layer)
        # 上采样
        self.up1 = Up(1024, 512, False, norm_layer=self.norm_layer)
        self.up2 = Up(512, 256, False, norm_layer=self.norm_layer)
        self.up3 = Up(256, 128, False, norm_layer=self.norm_layer)
        self.up4 = Up(128, 64, False, norm_layer=self.norm_layer)
        # 输出层
        self.outc = OutConv(64, n_classes)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        # 将deformation缩小到inp的大小, 并利用缩小后的deformation到inp中进行特征的采样,
        # ...... 这个正好映衬了我之前的那个思路
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, x, fl_src, fl_dst):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        dense_motion = self.dense_motion_network(source_image=x, kp_driving=fl_dst, kp_source=fl_src)
        occlusion_map = dense_motion['occlusion_map'] if 'occlusion_map' in dense_motion else None
        # deformation是比较重要的, 以下是deformation的计算方式
        # deformation = (sparse_motion * mask).sum(dim=1)
        deformation = dense_motion['deformation']
        out = self.deform_input(x5, deformation)

        # 有遮挡的话就再乘上遮挡区域的权重, 就是让out不考虑被遮挡区域
        if occlusion_map is not None:
            if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
            out = out * occlusion_map
        # 上采样
        # up_x1 = self.up1(x5, x4)
        up_x1 = self.up1(out, x4)
        up_x2 = self.up2(up_x1, x3)
        up_x3 = self.up3(up_x2, x2)
        up_x4 = self.up4(up_x3, x1)
        logits = self.outc(up_x4)
        out = logits
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False, groups=mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, same_channels_num=False, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_layer=norm_layer)
        else:
            if same_channels_num:
                self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels * 2, out_channels, norm_layer=norm_layer)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels, norm_layer=norm_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        Gx = gram_matrix(x)
        Gy = gram_matrix(y)
        return F.mse_loss(Gx, Gy) * 30000000


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        # 沙漏网络
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            # 下采样差值
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        将关键点转kpts的heatmap，并基于热图来得到driving和src热图的差值
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        # 为什么这里cat一个zeros就是加上背景的特征
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source.type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)

        # torch.Size([1, 1, 512, 512, 2]) - torch.Size([2, 68, 1, 1, 2])
        # (512, 512)上的点有和这68个点对应上的点就会被减成0, 这就表示了kp_driving在整张图上的空间运动
        coordinate_grid = identity_grid - kp_driving.view(bs, self.num_kp, 1, 1, 2)
        # 最后再加上kp_source的作用, 将前后的运动结合起来,
        # 所以这里得到的是0 --> s, 除了landmark其余的部分是禁止的, lm上的点会运动到s上
        """
        意思就是将s对应点的特征拉到d的对应点位上!!!!!
        """
        driving_to_source = coordinate_grid + kp_source.view(bs, self.num_kp, 1, 1, 2)
        # 不过结合上面两个代码, 看起来就像d ---> s, 并不是s ---> d

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)

        source_image: 原图
        spare_motions: 空间运动map
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))  # 这里扩充的+1应该是背景
        # 利用sparse_motions到sparse_motions中去取点(其实就是基于lm在特征图上取得对应的特征)
        # print(source_repeat.shape, sparse_motions.shape)  # torch.Size([138, 3, 512, 512]) torch.Size([138, 512, 512, 2])
        # 一个点对应一个map, 每个map只有src上的点位上对应的特征才会动, 其余的保持原来的位置
        """
        意思就是将s对应点的特征拉到d的对应点位上!!!!!
        68点是固定的, 也就是对应的, 只有与68个点相关的区域才会动, (512 * 512)上的其余点是不动的

        但是我这里有个疑问, 就是这里用的还是原来的像素尺度, 容错很小, 应该要是一个区域, 这样的容错才大
        所以这里可以做一个优化就是将点周围的区域进行迁移, 而不是只迁移一个点
        """
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)  # torch.Size([138, 3, 512, 512])
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    # 这个不知道谁对谁错
    # def forward(self, source_image, kp_driving, kp_source):
    def forward(self, source_image, kp_source, kp_driving):
        """
        image_in, src_fl, dst_fl
        但是根据训练脚本中的输输入, 这个传参是不是有问题src_fl, dst_fl传反了
        """

        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        # 得到src --> drv的差值热图
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)

        # 得到drv --> src的空间运动
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        # 基于上面的空间运动点来从src_feature上得到对应的特征
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)

        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation,
                           deformed_source], dim=2)
        input = input.view(bs, -1, h, w)
        # 这里输出的prediction的意义是什么
        # torch.Size([2, 276, 512, 512])
        prediction = self.hourglass(input)
        # torch.Size([2, 308, 512, 512])

        mask = self.mask(prediction)  # torch.Size([2, 69, 512, 512])

        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        # print(sparse_motion.shape, mask.shape)  # torch.Size([2, 69, 2, 512, 512]) torch.Size([2, 69, 1, 512, 512])
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        # from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        # from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)  # (512,)
    y = torch.arange(h).type(type)  # (512,)

    # 0 ~ 512 --> -1 ~ 1
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    # 在对方的维度上进行维度的扩充
    yy = y.view(-1, 1).repeat(1, w)  # (512, w) -> (h, w)
    xx = x.view(1, -1).repeat(h, 1)  # (h, 512) -> (h, w)

    # 其实就是将每个空间位置上的位置(x, y)显示的放出来作为一个特征
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class VGGLoss(nn.Module):
    def __init__(self, model=None):
        super(VGGLoss, self).__init__()
        if model is None:
            self.vgg = Vgg19()
        else:
            self.vgg = model

        self.vgg.cuda()
        # self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.style_criterion = StyleLoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # self.weights = [5.0, 1.0, 0.5, 0.4, 0.8]
        # self.style_weights = [10e4, 1000, 50, 15, 50]

    def forward(self, x, y, style=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if style:
            # return both perceptual loss and style loss.
            style_loss = 0
            for i in range(len(x_vgg)):
                this_loss = self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
                this_style_loss = self.style_weights[i] * self.style_criterion(x_vgg[i], y_vgg[i].detach())
                loss += this_loss
                style_loss += this_style_loss
            return loss, style_loss

        for i in range(len(x_vgg)):
            this_loss = self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            loss += this_loss
        return loss


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            # 创建单个判别器
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            # 是否要得到中间的特征
            if getIntermFeat:
                for j in range(n_layers + 2):
                    # 取得属性, 并设置属性
                    setattr(self, 'scale' + str(i) + '_layer' + str(j),
                            getattr(netD, 'model' + str(j))  # 从上面创建的netD中来取得属性'model' + str(j)
                            )
            else:
                # 设置一个属性
                setattr(self, 'layer' + str(i), netD.model)
        # 均值pool的下采样
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            # 这里是将每一层的模型取出来, 再forward的
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            # 前向传播
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    # 这个是pix2pixHD上的那个判别器
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)

    # netD = netD.to(str_G_device)
    netD.apply(weights_init)
    return netD


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, device=torch.device("cpu")):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)  # 基于input的shape, 使用self.real_label的值进行填充
                self.real_label_var = torch.autograd.Variable(real_tensor, requires_grad=False).to(
                    self.device)  # 这个造出来的值是没有梯度的
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = torch.autograd.Variable(fake_tensor, requires_grad=False).to(self.device)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
