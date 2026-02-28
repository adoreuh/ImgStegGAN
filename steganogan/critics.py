# -*- coding: utf-8 -*-
"""
SteganoGAN 评论器 v2.2
完全兼容旧版模型格式
"""

import torch
from torch import nn


class BasicCritic(nn.Module):
    """基础评论器
    
    Input: (N, 3, H, W)
    Output: (N, 1)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self._models = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, 1, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._models(x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)
        return x
