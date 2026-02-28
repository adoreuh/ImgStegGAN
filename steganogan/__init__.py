# -*- coding: utf-8 -*-
"""
ImgStegGAN V1.0.0
GAN-Driven Image Steganography with Qwen Enhancement

This project is a modified and extended version of SteganoGAN
by MIT Data To AI Lab (https://github.com/DAI-Lab/SteganoGAN)

Original paper:
    Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan.
    SteganoGAN: High Capacity Image Steganography with GANs.
    MIT EECS, January 2019. (arXiv:1901.03892)
"""

__version__ = '1.0.0'
__author__ = 'ImgStegGAN Team'
__title__ = 'ImgStegGAN: GAN-Driven Image Steganography with Qwen Enhancement'
__original_paper__ = 'https://arxiv.org/abs/1901.03892'
__original_repo__ = 'https://github.com/DAI-Lab/SteganoGAN'

from steganogan.models import SteganoGAN
from steganogan.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan.critics import BasicCritic
from steganogan.compat import (
    autocast,
    get_grad_scaler,
    torch_load,
    get_device,
    clear_cuda_cache,
    compile_model
)

__all__ = [
    'SteganoGAN',
    'BasicEncoder', 'ResidualEncoder', 'DenseEncoder',
    'BasicDecoder', 'DenseDecoder',
    'BasicCritic',
    'autocast',
    'get_grad_scaler',
    'torch_load',
    'get_device',
    'clear_cuda_cache',
    'compile_model',
]
