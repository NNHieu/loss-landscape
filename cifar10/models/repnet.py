
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import matplotlib.pyplot as plt 

import torch.autograd as autograd 

from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

import torch.optim as optim

import tqdm as tqdm
import os
import argparse

torch.manual_seed(0)

class ResNetLayer(nn.Module): 
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8): 
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels) 
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        if z is None:
            y = self.norm1(F.relu(self.conv1(x)))
            return self.norm3(F.relu(x + self.norm2(self.conv2(y))))
        else:
            y = self.norm1(F.relu(self.conv1(z)))
            return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))


class RepeatConvLayer(nn.Module):
    def __init__(self, f, num_repeat): 
        super().__init__() 
        self.f = f
        self.num_repeat = num_repeat

    def forward(self, x): 
        z = self.f(None, x)
        for i in range(self.num_repeat):
            z = self.f(z, x)
        return z

def repeatNet(num_repeat):
    chan = 48
    f = ResNetLayer(chan, 64, kernel_size=3) 
    model = nn.Sequential(nn.Conv2d(3,chan, kernel_size=3, bias=True, padding=1), 
                            nn.BatchNorm2d(chan), 
                            RepeatConvLayer(f, num_repeat), 
                            nn.BatchNorm2d(chan), 
                            nn.AvgPool2d(8,8), 
                            nn.Flatten(), 
                            nn.Linear(chan*4*4,10)).to(device)
    return model

def repeatNet5():
    return repeatNet(5)
