import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import *
from data_loader import *
import numpy as np

if __name__ == '__main__':

    datasets = FaceLandmarksDataset(img_dir='./afw/Image/',
        txt_file='./afw/train_68pt.txt')

    dataloader = DataLoader

    Hourglass = hg(*kwargs)


    criterion = nn.MSELoss()
    optimizer = optim.Adam()
