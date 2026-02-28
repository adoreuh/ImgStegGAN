# -*- coding: utf-8 -*-
"""
PyTorch 2.x 优化模块
仅支持 PyTorch 2.0+
"""

import torch
from torch.amp import GradScaler, autocast as _autocast


def autocast(enabled: bool = True, device_type: str = 'cuda'):
    return _autocast(device_type=device_type, enabled=enabled)


def get_grad_scaler():
    return GradScaler()


def torch_load(path, map_location='cpu', **kwargs):
    return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_cuda_info() -> dict:
    info = {
        'available': torch.cuda.is_available(),
        'version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'memory_allocated': 0,
        'memory_cached': 0,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**2
        info['memory_cached'] = torch.cuda.memory_reserved(0) / 1024**2
    
    return info


def compile_model(model, mode: str = 'reduce-overhead'):
    try:
        return torch.compile(model, mode=mode, fullgraph=False)
    except Exception as e:
        print(f"[警告] torch.compile 失败: {e}")
        return model


__all__ = [
    'autocast',
    'get_grad_scaler',
    'torch_load',
    'get_device',
    'clear_cuda_cache',
    'get_cuda_info',
    'compile_model',
]
