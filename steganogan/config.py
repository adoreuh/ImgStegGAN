# -*- coding: utf-8 -*-
"""
SteganoGAN 配置文件 V3.0
"""

VERSION = '3.0.0'
VERSION_NAME = 'SteganoGAN V3.0.0'
VERSION_DATE = '2026-02-16'

MIN_PYTORCH_VERSION = '2.0.0'

DEFAULT_CONFIG = {
    'data_depth': 1,
    'hidden_size': 64,
    'encoder': 'dense',
    'decoder': 'dense',
    'critic': 'basic',
    'cuda': True,
    'verbose': True
}

TRAINING_CONFIG = {
    'max_epochs': 20,
    'early_stop_patience': 5,
    'overfit_threshold': 0.08,
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    'batch_size': 4,
    'image_size': 128,
    'grad_clip': 1.0,
    'encoder_weight': 100.0
}

MODEL_CONFIGS = {
    'basic': {
        'encoder': 'BasicEncoder',
        'decoder': 'BasicDecoder',
        'hidden_size': 64
    },
    'dense': {
        'encoder': 'DenseEncoder',
        'decoder': 'DenseDecoder',
        'hidden_size': 64
    },
    'residual': {
        'encoder': 'ResidualEncoder',
        'decoder': 'DenseDecoder',
        'hidden_size': 64
    }
}

SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp']
MAX_IMAGE_SIZE = 8192
MIN_IMAGE_SIZE = 256
