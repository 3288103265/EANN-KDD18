import pickle

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils_multi_dataset import FusionFolder
import numpy as np


# Prepare data
F = open('train_set.pkl', 'rb')
t1 = pickle.load(F)
F.close()

F2 = open('test_set.pkl', 'rb')
t2 = pickle.load(F2)
F2.close()

print(type(t2))