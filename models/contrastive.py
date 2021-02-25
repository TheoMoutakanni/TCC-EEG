import torch
from torch import nn
import torch.nn.functional as F

class EncoderNet(torch.nn.Module):
    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, input_size_s=30,
                 dropout=0.25, apply_batch_norm=False):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        max_pool_size = int(max_pool_size_s * sfreq)
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        self.num_features = n_conv_chs

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
            nn.MaxPool2d((1, max_pool_size)),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
    def forward(self,x):
        """
        Input shape: (batch_size, 2, 1, nb_channels, 3000)
        Output shape: (batch_size, 1)
        """
        if self.n_channels > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        return x

    
class ContrastiveModule(nn.Module):
    def __init__(self, encoder):
        super(ContrastiveModule, self).__init__()
        self.encoder = encoder
        self.num_features = self.encoder.num_features
        self.fc = nn.Linear(in_features=self.num_features, out_features=2)
    
    def forward(self,x):
        """
        Input shape: (batch_size, 2, 1, nb_channels, 3000)
        Output shape: (batch_size, 1)
        """
        batch_size, _, nb_channels, window_size = x.shape
        x = x.view(batch_size*2, 1, nb_channels, window_size)
        x = self.encoder(x)
        features = x
        x = x.view(batch_size, 2, self.num_features)
        x1, x2 = x[:,0], x[:,1]
        x = torch.abs(x1 - x2)
        x = self.fc(x)
        return x, features