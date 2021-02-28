import numpy as np
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring, ProgressBar, EarlyStopping
import torch
from torch import nn
import torch.nn.functional as F

from utils.skorch import EEGTransformer


class EncoderNet(nn.Module):
    def __init__(self, feat_per_layer, n_channels, p_dropout=0.25, apply_batch_norm=False):
        super().__init__()

        self.num_features = feat_per_layer[-1]

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        feature_extractor = []
        for i in range(len(feat_per_layer)):
            feature_extractor.append(nn.Conv2d(in_channels=feat_per_layer[i - 1] if i > 0 else 1,
                                               out_channels=feat_per_layer[i],
                                               kernel_size=(1 if i > 0 else n_channels, 50)))
            feature_extractor.append(nn.ReLU())
            feature_extractor.append(batch_norm(feat_per_layer[i]))
            feature_extractor.append(nn.Dropout(p_dropout))
            feature_extractor.append(nn.MaxPool2d(
                kernel_size=(1, 6), stride=(1, 6)))

        self.feature_extractor = nn.Sequential(
            *feature_extractor,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        """
        Input shape: (batch_size, 2, 1, n_channels, 3000)
        Output shape: (batch_size, 1)
        """
        x = self.feature_extractor(x)
        return x


class ContrastiveModule(nn.Module):
    def __init__(self, encoder):
        super(ContrastiveModule, self).__init__()
        self.encoder = encoder
        self.num_features = self.encoder.num_features
        self.fc = nn.Linear(in_features=self.num_features, out_features=2)

    def forward(self, x):
        """
        Input shape: (batch_size, 2, 1, nb_channels, 3000)
        Output shape: (batch_size, 1)
        """
        batch_size, _, nb_channels, window_size = x.shape
        x = x.view(batch_size * 2, 1, nb_channels, window_size)
        x = self.encoder(x)
        features = x
        x = x.view(batch_size, 2, self.num_features)
        x1, x2 = x[:, 0], x[:, 1]
        x = torch.abs(x1 - x2)
        x = self.fc(x)
        return x, features

    def train_(self, train_set, valid_set, lr=5e-4, batch_size=16, max_nb_epochs=20,
               early_stopping_patience=5):
        # Train using a GPU if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Callbacks
        train_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
            lower_is_better=False)
        valid_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
            lower_is_better=False)
        early_stopping = EarlyStopping(
            monitor='valid_loss', patience=early_stopping_patience)
        callbacks = [
            ('train_bal_acc', train_bal_acc),
            ('valid_bal_acc', valid_bal_acc),
            ('progress_bar', ProgressBar()),
            ('early_stopping', early_stopping),
        ]

        # Skorch model creation
        skorch_net = EEGTransformer(
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
        skorch_net.fit(train_set, y=None, epochs=max_nb_epochs)

        return skorch_net


class ClassifierNet(nn.Module):
    def __init__(self, encoder, p_dropout=0.4):
        super(ClassifierNet, self).__init__()
        # Auto-encoder
        self.encoder = encoder
        # Dense classifier
        self.dropout = nn.Dropout(p_dropout)
        self.dense1 = nn.Linear(encoder.num_features, encoder.num_features)
        self.dense2 = nn.Linear(encoder.num_features, 5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


def train_and_test(
        classifier_net, train_set, valid_set, test_set=None, lr=5e-4,
        batch_size=16, max_nb_epochs=20, early_stopping_patience=5,
        train_what="last"):
    """
    """
    if train_what == "last":
        for param in classifier_net.encoder.parameters():
            param.requires_grad = False
    # else train all

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Callbacks
    train_bal_acc = EpochScoring(
        scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
        lower_is_better=False)
    valid_bal_acc = EpochScoring(
        scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
        lower_is_better=False)
    early_stopping = EarlyStopping(
        monitor='valid_bal_acc', patience=early_stopping_patience, lower_is_better=False)
    callbacks = [
        ('train_bal_acc', train_bal_acc),
        ('valid_bal_acc', valid_bal_acc),
        ('progress_bar', ProgressBar()),
        ('early_stopping', early_stopping),
    ]

    skorch_classifier = NeuralNetClassifier(
        classifier_net,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        # using valid_set for validation
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    # Model training for a specified number of epochs. `y` is None as it is already
    # supplied in the dataset.
    skorch_classifier.fit(train_set, y=None, epochs=max_nb_epochs)

    if test_set is not None:
        X, y = zip(*list(iter(test_set)))
        X = torch.stack(X)
        acc = skorch_classifier.score(X, y)
        return skorch_classifier, acc

    return skorch_classifier
