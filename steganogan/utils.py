# -*- coding: utf-8 -*-
"""
SteganoGAN 工具函数 v2.2
仅支持 PyTorch 2.x
"""

import zlib
from math import exp
from typing import List

import numpy as np
import torch
from reedsolo import RSCodec
from torch.nn.functional import conv2d

rs = RSCodec(32)
rs_strong = RSCodec(128)


def text_to_bits(text: str) -> List[int]:
    """将文本转换为比特列表"""
    return bytearray_to_bits(text_to_bytearray(text))


def bits_to_text(bits: List[int]) -> str:
    """将比特列表转换为文本"""
    return bytearray_to_text(bits_to_bytearray(bits))


def text_to_bits_rs(text: str) -> List[int]:
    """将文本转换为比特列表 - 使用RS纠错码"""
    return bytearray_to_bits(text_to_bytearray_rs(text))


def bits_to_text_rs(bits: List[int]) -> str:
    """将比特列表转换为文本 - 使用RS纠错码"""
    return bytearray_to_text_rs(bits_to_bytearray(bits))


def bytearray_to_bits(x: bytearray) -> List[int]:
    """将字节数组转换为比特列表"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def bits_to_bytearray(bits: List[int]) -> bytearray:
    """将比特列表转换为字节数组"""
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)


def text_to_bytearray(text: str) -> bytearray:
    """压缩并添加纠错码"""
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"), level=9)
    x = rs.encode(bytearray(x))
    return x


def bytearray_to_text_raw(x: bytearray) -> str:
    """直接解码，不使用纠错码"""
    try:
        text = zlib.decompress(x)
        return text.decode("utf-8")
    except BaseException:
        return None


def bytearray_to_text(x: bytearray) -> str:
    """纠错并解压 - 多策略尝试"""
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except BaseException:
        pass
    
    try:
        text = zlib.decompress(x)
        return text.decode("utf-8")
    except BaseException:
        pass
    
    for offset in range(0, min(len(x), 20)):
        try:
            decoded = rs.decode(x[offset:])
            text = zlib.decompress(decoded)
            return text.decode("utf-8")
        except:
            continue
    
    for offset in range(0, min(len(x), 20)):
        try:
            text = zlib.decompress(x[offset:])
            return text.decode("utf-8")
        except:
            continue
    
    return None


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """高斯窗口"""
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.tensor(_exp, dtype=torch.float32)
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """创建卷积窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
          window_size: int, channel: int, size_average: bool = True) -> torch.Tensor:
    """计算SSIM"""
    padding_size = window_size // 2

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, 
         size_average: bool = True) -> torch.Tensor:
    """计算两张图像的SSIM"""
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    window = window.to(device=img1.device, dtype=img1.dtype)
    return _ssim(img1, img2, window, window_size, channel, size_average)
