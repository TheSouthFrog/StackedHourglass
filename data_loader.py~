# DataLoader

from __future__ import print_function, division

import os
import torch
import numpy as np
import pandas as pd
from skimage import io.transform
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils




class FaceLandmarksDataset(Dataset):

    def __init__(self, txt_file, img_dir, transform=None):
        self.landmarks_frame = pd.read_table(txt_file,sep = ' ',header = None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
            self.landmarks_frame.iloc[idx, -1])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 0:-1].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample
