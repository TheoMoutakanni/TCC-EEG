import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class SimpleSleepStagerChambon2018(nn.Module):
    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.25, apply_batch_norm=False):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        max_pool_size = int(max_pool_size_s * sfreq)
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        self.n_classes = n_classes
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size),
                padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size))
        )
        self.gru1 = nn.GRU(len_last_layer, len_last_layer // 2, batch_first=True, bidirectional=True)
        self.skip_gru1 = nn.Linear(len_last_layer, len_last_layer)
        self.gru2 = nn.GRU(len_last_layer, len_last_layer // 2, batch_first=True, bidirectional=True)
        self.skip_gru2 = nn.Linear(len_last_layer, len_last_layer)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last_layer, n_classes)
        )

    @staticmethod
    def _len_last_layer(n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs

    def forward(self, data):
        """Forward pass.
        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        n_windows = data[1]
        x = data[0]

        n_batch, n_channels, n_times = x.shape
        x = x.unsqueeze(1)

        if self.n_channels > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)

        x = x.view(n_batch // n_windows, n_windows, -1)
        
        x_gru, _ = self.gru1(x)
        x_skip = self.skip_gru1(x)
        x = x_gru + x_skip
        
        x_gru, _ = self.gru2(x)
        x_skip = self.skip_gru2(x)
        x = x_gru + x_skip

        x = self.fc(x)
        
        if False: #not self.training:
            center_pos = data[2]
            x = x[torch.arange(x.size(0)), center_pos, :]
        
        return x.view(-1, self.n_classes) #.transpose(1, 2)