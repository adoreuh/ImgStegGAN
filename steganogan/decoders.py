# -*- coding: utf-8 -*-
"""
SteganoGAN 解码器 v2.2
完全兼容旧版模型格式
"""

import torch
from torch import nn


class BasicDecoder(nn.Module):
    """基础解码器
    
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def __init__(self, data_depth: int, hidden_size: int):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size, data_depth, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DenseDecoder(nn.Module):
    """密集连接解码器
    
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """
    
    def __init__(self, data_depth: int, hidden_size: int):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3, data_depth, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3], dim=1))
        return x4
