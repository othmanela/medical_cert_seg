import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.append('..')
import helpers as h


class PolypDataset(Dataset):
    in_channels = 3
    out_channels = 1

    width = 384
    height = 288

    def __init__(self, directory, manual_centers=None, center_augmentation=False):
        self.directory = p.join('datasets/polyp', directory)
        self.polar = polar
        self.manual_centers = manual_centers
        self.center_augmentation = center_augmentation

        self.file_names = h.listdir(p.join(self.directory, 'label'))
        self.file_names.sort()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label_file = p.join(self.directory, 'label', file_name)
        input_file = p.join(self.directory, 'input', file_name)

        label = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
        label = label.astype(np.float32)
        label /= 255.0

        input = cv.imread(input_file)
        input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
        input = input.astype(np.float32)
        input /= 255.0

        input = input.transpose(2, 0, 1)
        label = np.expand_dims(label, axis=-1)
        label = label.transpose(2, 0, 1)

        input_tensor = torch.from_numpy(input)
        label_tensor = torch.from_numpy(label)

        return input_tensor, label_tensor
