import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = ImageFolder(root='/home/wangpenghui/Datasets/New folder/', transform=transforms)
dataloader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True)

for idx, (inputs, labels) in enumerate(dataloader):
    print(labels)