import torch.nn as nn
class ApplyExp(nn.Module):
    """
    这个style可以是任何有意义的东西, id, expersion, 都是可以的


        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        """
        Args:
            latent_size:
            channels: 是输入的x的channel
        """

        super(ApplyExp, self).__init__()
        """
        这里的权重是否需要共享, 可以后面做实验
        """
        self.linear = nn.Linear(latent_size, channels * 2)  # 将latent转换成两倍
        self.linear_src = nn.Linear(latent_size, channels * 2)  # 将latent转换成两倍

    def forward(self, x, latent, src_latent):

        # 通过一个fc的两倍长度映射得到 人脸五官的方差和均值
        style = self.linear(latent)  # style => [batch_size, n_channels*2], [b, 128]
        shape = [style.shape[0], 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        # torch.Size([1, 512, 28, 28]) torch.Size([1, 2, 512, 1, 1])

        src_style = self.linear_src(src_latent)  # style => [batch_size, n_channels*2], [b, 128]
        src_style = src_style.view(shape)  # [batch_size, 2, n_channels, ...]

        # 再将得到的方差和均值加到x(x已经被除去了)上去

        x = (x - src_style[:, 1] * 1) / (src_style[:, 0] * 1 + 1.)
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x