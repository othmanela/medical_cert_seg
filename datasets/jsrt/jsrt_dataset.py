import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import albumentations.augmentations.functional as F

sys.path.append('..')
import helpers as h


class JsrtDataset(Dataset):
    width = 512
    height = 512

    in_channels = 3
    out_channels = 4

    def __init__(self, directory, manual_centers=None, center_augmentation=False, percent=None):
        self.directory = p.join('datasets/jsrt', directory)
        self.polar = polar
        self.manual_centers = manual_centers
        self.center_augmentation = center_augmentation
        self.percent = percent

        self.file_names = h.listdir(p.join(self.directory, 'input'))
        self.file_names.sort()

    def __len__(self):
        length = len(self.file_names)
        if self.percent is not None:
            length = int(length * self.percent)
        return length

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label_file = p.join(self.directory, 'label', file_name.replace('.png', '.npy'))
        input_file = p.join(self.directory, 'input', file_name)

        label = np.load(label_file)
        label = label.astype(np.float32)

        input = cv.imread(input_file)
        input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
        input = input.astype(np.float32)
        input /= 255.0

        input = input.transpose(2, 0, 1)
        label = np.expand_dims(label, axis=-1)
        label = label.transpose(2, 0, 1)

        input_tensor = torch.from_numpy(input)
        label_tensor = torch.from_numpy(label).to(torch.int64)

        return input_tensor, label_tensor
