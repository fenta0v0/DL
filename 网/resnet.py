"""
ResNet
"""
import torch
from tensorboard import summary
from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    # residual function
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_fuction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False)
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_fuction(x) + self.shortcut(x))
