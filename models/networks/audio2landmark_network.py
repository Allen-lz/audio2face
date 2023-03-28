import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import math
import torch.nn.functional as F
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_FEAT_SIZE = 161
FACE_ID_FEAT_SIZE = 204
Z_SIZE = 16
EPSILON = 1e-40


class Audio2landmark_content(nn.Module):

    def __init__(self, num_window_frames=18, in_size=80, lstm_size=AUDIO_FEAT_SIZE, use_prior_net=False, hidden_size=256, num_layers=3, drop_out=0, bidirectional=False):
        super(Audio2landmark_content, self).__init__()

        self.fc_prior = self.fc = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, lstm_size),
        )

        self.use_prior_net = use_prior_net
        if(use_prior_net):
            self.bilstm = nn.LSTM(input_size=lstm_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=drop_out,
                                  bidirectional=bidirectional,
                                  batch_first=True, )
        else:
            self.bilstm = nn.LSTM(input_size=in_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=drop_out,
                                  bidirectional=bidirectional,
                                  batch_first=True, )

        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_window_frames = num_window_frames

        self.fc_in_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.fc_in_features + FACE_ID_FEAT_SIZE, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 204),
        )

    def forward(self, au, face_id):
        # aus.shape =              torch.Size([287, 18, 80])
        # residual_face_id.shape = torch.Size([287, 204])
        inputs = au
        if(self.use_prior_net):
            inputs = self.fc_prior(inputs.contiguous().view(-1, self.in_size))
            inputs = inputs.view(-1, self.num_window_frames, self.lstm_size)

        # torch.Size([287, 18, 256])
        output, (hn, cn) = self.bilstm(inputs)
        output = output[:, -1, :]  # 这里只要最后的结果 output.shape = torch.Size([287, 256])

        if(face_id.shape[0] == 1):
            face_id = face_id.repeat(output.shape[0], 1)
        output2 = torch.cat((output, face_id), dim=1)

        output2 = self.fc(output2)
        # output += face_id

        # print(output2.shape, face_id.shape)
        # 可以发现这里的18, 没有了
        # torch.Size([287, 204]) torch.Size([287, 204])
        return output2, face_id