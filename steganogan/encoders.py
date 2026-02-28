# -*- coding: utf-8 -*-
"""
SteganoGAN 编码器 v2.2
完全兼容旧版模型格式
"""

import torch
from torch import nn


class BasicEncoder(nn.Module):
    """基础编码器
    
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = False

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
            nn.Conv2d(hidden_size + data_depth, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size, 3, 3, padding=1),
        )

    def forward(self, image: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        x = self.conv1(image)
        x = self.conv2(torch.cat([x, data], dim=1))
        x = self.conv3(x)
        x = self.conv4(x)
        
        if self.add_image:
            x = image + x
        
        return x


class ResidualEncoder(BasicEncoder):
    """残差编码器
    
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = True

    def __init__(self, data_depth: int, hidden_size: int):
        super().__init__(data_depth, hidden_size)


class DenseEncoder(nn.Module):
    """密集连接编码器
    
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = True

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
            nn.Conv2d(hidden_size + data_depth, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2 + data_depth, hidden_size, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3 + data_depth, 3, 3, padding=1),
        )

    def forward(self, image: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(image)
        x2 = self.conv2(torch.cat([x1, data], dim=1))
        x3 = self.conv3(torch.cat([x1, x2, data], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3, data], dim=1))
        
        if self.add_image:
            x4 = image + x4
        
        return x4
