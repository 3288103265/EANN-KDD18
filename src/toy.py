import torchvision.models as models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch import optim
import pretrainedmodels

# Pretrained model
# modify last layer
model = models.vgg19_bn()
fc_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(fc_features, 2)

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Randomly split dataset into train and test parts.
batch_size = 16
dataset = ImageFolder(root='/home/wangpenghui/Datasets/New folder/',transform=data_transforms)

