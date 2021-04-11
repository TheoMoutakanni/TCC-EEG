import bisect

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess, zscore
from braindecode.datautil.windowers import create_windows_from_events
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


def get_sleep_physionet(subject_ids=range(83), recording_ids=[1, 2]):
    bug_subjects = [13, 36, 39, 48, 52, 59, 65, 68, 69, 73, 75, 76, 78, 79]
    subject_ids = [id_ for id_ in subject_ids if id_ not in bug_subjects]

    dataset = SleepPhysionet(
        subject_ids=subject_ids, recording_ids=recording_ids, crop_wake_mins=30)
    high_cut_hz = 30

    preprocessors = [
        # convert from volt to microvolt, directly modifying the numpy array
        NumpyPreproc(fn=lambda x: x * 1e6),
        # bandpass filter
        MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz),
    ]
    # Transform the data
    preprocess(dataset, preprocessors)

    mapping = {  # We merge stages 3 and 4 following AASM standards.
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }

    window_size_s = 30
    sfreq = 100
    window_size_samples = window_size_s * sfreq

    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True, mapping=mapping)

    preprocess(windows_dataset, [MNEPreproc(fn=zscore)])

    info = pd.read_excel('SC-subjects.xls')

    return windows_dataset, info


def split_dataset(
        windows_dataset, self_train_ratio=0.7, self_valid_ratio=0.1,
        valid_ratio=0.1, test_ratio=0.1):
    """
    70% for self_train_set
    10% for self_valid_set
    10% for supervised valid_set
    10% for supervised test_set
    The supervised train_set is identical to self_train_set + self_valid_set
    """
    assert (self_train_ratio + self_valid_ratio + valid_ratio +
            test_ratio != 1), "wrong dataset ratios"

    splitted = windows_dataset.split('subject')
    subjects = list(splitted.keys())
    np.random.shuffle(subjects)
    self_train_index = round(len(subjects) * self_train_ratio)
    self_valid_index = self_train_index + \
        round(len(subjects) * self_valid_ratio)
    valid_index = self_valid_index + round(len(subjects) * valid_ratio)

    self_train_subjects = subjects[:self_train_index]
    self_valid_subjects = subjects[self_train_index:self_valid_index]
    full_train_subjects = subjects[:self_valid_index]
    valid_subjects = subjects[self_valid_index:valid_index]
    test_subjects = subjects[valid_index:]

    self_train_set = TimeContrastiveDataset([splitted[s] for s in self_train_subjects],
                                            delta_index_positive=5, delta_index_negative=20)
    self_valid_set = TimeContrastiveDataset([splitted[s] for s in self_valid_subjects],
                                            delta_index_positive=5, delta_index_negative=20)
    full_train_set = SkorchDataset([splitted[s] for s in full_train_subjects])
    valid_set = SkorchDataset([splitted[s] for s in valid_subjects])
    test_set = SkorchDataset([splitted[s] for s in test_subjects])

    subjects_dic = {
        "self_train_subjects": self_train_subjects,
        "self_valid_subjects": self_valid_subjects,
        "full_train_subjects": full_train_subjects,
        "valid_subjects": valid_subjects,
        "test_subjects": test_subjects
    }

    return self_train_set, self_valid_set, full_train_set, valid_set, test_set, subjects_dic


class SkorchDataset(BaseConcatDataset):
    def __init__(self, list_of_ds):
        super(SkorchDataset, self).__init__(list_of_ds)

    def __getitem__(self, index):
        x, y, _ = super().__getitem__(index)
        return torch.from_numpy(x).unsqueeze(0), y


def list_of_train_sets(train_subjects, windows_dataset, dataset_cls=SkorchDataset, dataset_args={}):
    splitted = windows_dataset.split('subject')
    train_sets = []
    for i, _ in enumerate(train_subjects):
        train_set = dataset_cls([splitted[s] for s in train_subjects[:i + 1]],
                                **dataset_args)
        train_sets.append(train_set)
    return train_sets


class TimeContrastiveDataset(BaseConcatDataset):
    """
    ----------
    list_of_ds: list
        list of BaseDataset, BaseConcatDataset or WindowsDataset
    """

    def __init__(self, list_of_ds, delta_index_positive=10, delta_index_negative=100):
        super(TimeContrastiveDataset, self).__init__(list_of_ds)
        self.delta_index_positive = delta_index_positive
        self.delta_index_negative = delta_index_negative

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            anchor_sample_idx = idx
        else:
            anchor_sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        dataset_size = len(self.datasets[dataset_idx])
        contrastive_sample_idx = anchor_sample_idx
        while contrastive_sample_idx == anchor_sample_idx:
            if np.random.random() <= 0.5:
                y = 1
                contrastive_sample_idx = np.random.choice(np.arange(max(0, anchor_sample_idx - self.delta_index_positive),
                                                                    min(dataset_size, anchor_sample_idx + self.delta_index_positive + 1)))
            else:
                y = 0
                contrastive_sample_idx = np.random.choice(np.concatenate([np.arange(0, max(0, anchor_sample_idx - self.delta_index_negative)),
                                                                          np.arange(min(dataset_size, anchor_sample_idx + self.delta_index_negative + 1), dataset_size)]))

        anchor_sample = self.datasets[dataset_idx][anchor_sample_idx]
        contastive_sample = self.datasets[dataset_idx][contrastive_sample_idx]

        X = list(zip(*[anchor_sample, contastive_sample]))[0]
        X = np.array(X)

        return X, y


class MultipleWindowsDataset(BaseConcatDataset):
    """
    ----------
    list_of_ds: list
        list of BaseDataset, BaseConcatDataset or WindowsDataset
    """

    def __init__(self, list_of_ds, number_of_windows=10, training=True):
        super(MultipleWindowsDataset, self).__init__(list_of_ds)
        self.number_of_windows = number_of_windows
        self.training = training

    def __len__(self):
        if self.training:
            return super().__len__() // self.number_of_windows
        else:
            return super().__len__()

    def __getitem__(self, idx):
        if self.training:
            idx = idx * self.number_of_windows + \
                np.random.randint(0, self.number_of_windows)

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            center_idx = idx
        else:
            center_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        center_crop = center_idx % self.number_of_windows

        dataset_size = len(self.datasets[dataset_idx])

        if center_idx < self.number_of_windows // 2:
            center_idx = self.number_of_windows // 2
        elif center_idx > dataset_size - self.number_of_windows // 2 - 1:
            center_idx = dataset_size - self.number_of_windows // 2 - 1

        start_idx = center_idx - self.number_of_windows // 2
        end_idx = center_idx + self.number_of_windows // 2
        if end_idx == start_idx:
            end_idx += 1

        windows = []
        labels = []
        #pos = []
        for i in range(start_idx, end_idx):
            windows.append(self.datasets[dataset_idx][i][0])
            labels.append(self.datasets[dataset_idx][i][1])
            # pos.append(i)

        X = np.array(windows)
        y = np.array(labels)
        if self.training:
            return X, y
        else:
            return X, y[center_crop], [center_crop]
