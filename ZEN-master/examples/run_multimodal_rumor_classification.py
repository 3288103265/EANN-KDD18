import torch.nn as nn
from torchvision import models


# define models.
class FusionModel(nn.Module):
    """Fuse vgg and ZEN"""

    def __init__(self):
        self.vgg = models.vgg19_bn()
        self.
