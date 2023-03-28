import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")



class Embedder(nn.Module):
    def __init__(self, feat_size, d_model):
        super().__init__()
        self.embed = nn.Linear(feat_size, d_model)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].clone().detach().to(x.device)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

    # build a decoder layer with two multi-head attention layers and
    # one feed-forward layer

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, in_size):
        super().__init__()
        self.N = N
        self.embed = Embedder(in_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)

        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, in_size):
        super().__init__()
        self.N = N
        self.embed = Embedder(in_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Audio2landmark_speaker_aware(nn.Module):
    def __init__(self, audio_feat_size=80, c_enc_hidden_size=256, num_layers=3, drop_out=0,
                 spk_feat_size=256, spk_emb_enc_size=128, lstm_g_win_size=64, add_info_size=6,
                 transformer_d_model=32, N=2, heads=2, z_size=128, audio_dim=256):
        super(Audio2landmark_speaker_aware, self).__init__()

        self.lstm_g_win_size = lstm_g_win_size
        self.add_info_size = add_info_size
        self.audio_content_encoder = nn.LSTM(input_size=audio_feat_size,
                                             hidden_size=c_enc_hidden_size,
                                             num_layers=num_layers,
                                             dropout=drop_out,
                                             bidirectional=False,
                                             batch_first=True)

        self.use_audio_projection = not (audio_dim == c_enc_hidden_size)
        if(self.use_audio_projection):
            self.audio_projection = nn.Sequential(
                nn.Linear(in_features=c_enc_hidden_size, out_features=256),
                nn.LeakyReLU(0.02),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.02),
                nn.Linear(128, audio_dim),
            )


        ''' original version '''
        self.spk_emb_encoder = nn.Sequential(
            nn.Linear(in_features=spk_feat_size, out_features=256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, spk_emb_enc_size),
        )

        d_model = transformer_d_model * heads
        N = N
        heads = heads

        self.encoder = Encoder(d_model, N, heads, in_size=audio_dim + spk_emb_enc_size + z_size)
        self.decoder = Decoder(d_model, N, heads, in_size=204)
        self.out = nn.Sequential(
            nn.Linear(in_features=d_model + z_size, out_features=512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 204),
        )


    def forward(self,
                au,         # 说话人的音频
                emb,        # 用于预测说话人emb的au
                add_z_spk=True):
        """
        Args:
            au: 18个音频帧(18个视频帧对应的音频mel)
            emb: 在inference的阶段是可以一直累计并求平均, 这样慢慢的emb的结果就会越来越稳定
            add_z_spk: 是否要在emb上加噪
        Returns:

        """
        # audio
        audio_encode, (_, _) = self.audio_content_encoder(au)
        audio_encode = audio_encode[:, -1, :]

        if self.use_audio_projection:
            audio_encode = self.audio_projection(audio_encode)

        # spk
        spk_encode = self.spk_emb_encoder(emb)
        # 增加噪声
        if add_z_spk:
            z_spk = torch.tensor(torch.randn(spk_encode.shape)*0.01, requires_grad=False, dtype=torch.float).to(spk_encode.device)
            spk_encode = spk_encode + z_spk

        # comb
        # cat + 加噪
        z = torch.tensor(torch.zeros(au.shape[0], 128), requires_grad=False, dtype=torch.float).to(audio_encode.device)
        comb_encode = torch.cat((audio_encode, spk_encode, z), dim=1)
        src_feat = comb_encode.unsqueeze(0)

        e_outputs = self.encoder(src_feat)[0]

        # 再加噪
        e_outputs = torch.cat((e_outputs, z), dim=1)

        fl_pred = self.out(e_outputs)

        return fl_pred