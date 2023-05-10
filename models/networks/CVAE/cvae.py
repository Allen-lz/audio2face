import torch
import torch.nn as nn
from collections import OrderedDict

class VAE(nn.Module):

    def __init__(self, latent_size=4, audio_feat_size=80, c_enc_hidden_size=256, num_layers=2, proj_dim=32, drop_out=0):
        super().__init__()
        # 先对audio的序列进编码
        self.audio_content_encoder = nn.LSTM(input_size=audio_feat_size,
                                             hidden_size=c_enc_hidden_size,
                                             num_layers=num_layers,
                                             dropout=drop_out,
                                             bidirectional=False,
                                             batch_first=True)

        self.linear_fusion_proj = nn.Linear(c_enc_hidden_size, proj_dim)

        self.encoder = Encoder(latent_size, proj_dim)
        self.decoder = Decoder(proj_dim, latent_size)

    def forward(self, x, aus=None, emb=None):
        audio_encode, (_, _) = self.audio_content_encoder(aus)
        audio_encode = audio_encode[:, -1, :]  # [batchszie, 256], 这里是取了最后一个算是综合的结果
        c = audio_encode + emb
        c = self.linear_fusion_proj(c)

        # ==============================================
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, aus, emb):
        """
        Args:
            z: 是随机采样得到的
            aus: 音频数据
            emb: 人声中id的emb
        Returns:
        """

        audio_encode, (_, _) = self.audio_content_encoder(aus)
        audio_encode = audio_encode[:, -1, :]  # [batchszie, 256], 这里是取了最后一个算是综合的结果
        c = audio_encode + emb
        c = self.linear_fusion_proj(c)

        recon_x = self.decoder(z, c)

        return recon_x

    def load_weight(self, checkpoint_path='checkpoints/audio2landmark/best_vae.pth'):
        state_dict = torch.load(checkpoint_path)

        self.load_state_dict(state_dict)


class Encoder(nn.Module):

    def __init__(self, latent_size=4, proj_dim=32):

        super().__init__()

        self.x_proj = nn.Linear(4, proj_dim)  # pose只有三个参数
        self.MLP = nn.Sequential(nn.Linear(2 * proj_dim, proj_dim),
                                 nn.Linear(proj_dim, 16),
                                 nn.Linear(16, proj_dim)
                                 )

        self.linear_means = nn.Linear(proj_dim, latent_size)
        self.linear_log_var = nn.Linear(proj_dim, latent_size)

    def forward(self, x, c):
        # 在对c进行一次映射
        x = self.x_proj(x)
        x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, proj_dim=32, latent_size=4):
        super().__init__()
        self.MLP = nn.Sequential(nn.Linear(proj_dim + latent_size, proj_dim),
                                 nn.Linear(proj_dim, proj_dim // 2),
                                 nn.Linear(proj_dim // 2, 4)
                                 )

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x
