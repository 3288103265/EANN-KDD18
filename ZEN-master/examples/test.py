import pickle

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils_multi_dataset import FusionFolder
import numpy as np


# Prepare data
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

F = open('all_dict.pkl', 'rb')
all_dict = pickle.load(F)
F.close()
all_sets = FusionFolder("/home/wangpenghui/Datasets/WeiboRumorSet/", all_dict, transform=data_transforms)

# split dataset into trainset and test set.
train_size = int(0.6 * len(all_sets))
test_size = len(all_sets) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(all_sets, [train_size, test_size])
F = open('train_set.pkl', 'wb')
pickle.dump(train_dataset, F)
F.close()

F = open('train_set.pkl', 'rb')
t1 = pickle.load(F)
F.close()

print(t1  train_dataset)