import numpy as np
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring, ProgressBar
import torch
from torch import nn
import torch.nn.functional as F

from utils.skorch import EEGTransformer


class EncoderNet(nn.Module):
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

    def train_(self, train_set, valid_set, lr=5e-4, batch_size=16, nb_epochs=20):
        # Train using a GPU if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Callbacks
        train_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
            lower_is_better=False)
        valid_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
            lower_is_better=False)
        callbacks = [
            ('train_bal_acc', train_bal_acc),
            ('valid_bal_acc', valid_bal_acc),
            ('progress_bar', ProgressBar()),
        ]

        # Skorch model creation
        self.skorch_net = EEGTransformer(
            self.to(device),
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            optimizer__lr=lr,
            train_split=predefined_split(valid_set),
            batch_size=batch_size,
            callbacks=callbacks,
            device=device
        )

        # Training: `y` is None since it is already supplied in the dataset.
        self.skorch_net.fit(train_set, y=None, epochs=nb_epochs)


class ClassifierNet(nn.Module):
    def __init__(self, encoder):
        super(ClassifierNet, self).__init__()
        #Auto-encoder
        self.encoder = encoder
        #Dense classifier
        self.dense1 = nn.Linear(encoder.num_features, encoder.num_features)
        self.dense2 = nn.Linear(encoder.num_features, 5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x
    

class Classifier:
    def __init__(self, encoder):
        self.encoder = encoder


    def train_and_test(self, train_set, valid_set, test_set=None, lr=5e-4, batch_size=16, nb_epochs=20):
        """
        """
        self.classifier_net = ClassifierNet(self.encoder)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
            lower_is_better=False)
        valid_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
            lower_is_better=False)
        callbacks = [('train_bal_acc', train_bal_acc),
                     ('valid_bal_acc', valid_bal_acc)]

        self.skorch_classifier = NeuralNetClassifier(
            self.classifier_net,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=predefined_split(valid_set),  # using valid_set for validation
            optimizer__lr=lr,
            batch_size=batch_size,
            callbacks=callbacks,
            device=device,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )
        # Model training for a specified number of epochs. `y` is None as it is already
        # supplied in the dataset.
        self.skorch_classifier.fit(train_set, y=None, epochs=nb_epochs)
        
        if test_set is not None:
            X, y = zip(*list(iter(test_set)))
            X = torch.stack(X)
            acc = self.skorch_classifier.score(X, y)
            return acc