# -*- coding: utf-8 -*-
"""
ImgStegGAN V1.0.0 Training Script
Supports large datasets (20GB+), checkpoint resumption, dynamic learning rate

This project is a modified and extended version of SteganoGAN
by MIT Data To AI Lab (https://github.com/DAI-Lab/SteganoGAN)

Original paper:
    Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan.
    SteganoGAN: High Capacity Image Steganography with GANs.
    MIT EECS, January 2019. (arXiv:1901.03892)

Features:
1. Batch data loading with checkpoint resumption
2. Dynamic learning rate adjustment
3. Training progress monitoring and logging
4. Model saving (intermediate and best models)
5. GPU memory optimization
6. Command line configuration
7. Post-training evaluation and performance report

Usage:
    python train.py --data_root ./research/data --epochs 20 --batch_size 4
    python train.py --resume ./checkpoints/checkpoint_epoch_10.pt
    python train.py --help
"""

import os
import sys
import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings('ignore')

from steganogan.models import SteganoGAN
from steganogan.encoders import BasicEncoder, DenseEncoder, ResidualEncoder
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan.critics import BasicCritic
from steganogan.compat import autocast, get_grad_scaler, clear_cuda_cache


# ============================================================================
# 配置常量
# ============================================================================

VERSION = "3.0.0"
DEFAULT_CONFIG = {
    'data_depth': 1,
    'hidden_size': 64,
    'architecture': 'dense',
    'max_epochs': 20,
    'batch_size': 4,
    'image_size': 128,
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'encoder_weight': 100.0,
    'target_accuracy': 0.90,
    'early_stop_patience': 5,
    'overfit_threshold': 0.08,
    'num_workers': 4,
    'val_ratio': 0.1,
    'seed': 42,
    'use_amp': True,
    'checkpoint_interval': 5,
    'log_interval': 10,
}

ARCHITECTURES = {
    'basic': {'encoder': BasicEncoder, 'decoder': BasicDecoder},
    'dense': {'encoder': DenseEncoder, 'decoder': DenseDecoder},
    'residual': {'encoder': ResidualEncoder, 'decoder': DenseDecoder},
}


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(log_dir: str, verbose: bool = True) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    logger = logging.getLogger('SteganoGAN')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# 数据集类
# ============================================================================

class LargeImageDataset(torch.utils.data.Dataset):
    """大型图像数据集 - 支持惰性加载和内存优化"""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 128,
        transform: Optional[transforms.Compose] = None,
        cache_size: int = 1000
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform or self._get_transform(image_size)
        self.cache_size = cache_size
        self._cache: Dict[int, torch.Tensor] = {}
        
        self.image_paths = self._collect_paths()
        print(f"  [数据集] 加载完成: {len(self.image_paths)} 张图像")
    
    def _collect_paths(self) -> List[str]:
        valid_paths = []
        
        if self.root_dir.is_file():
            valid_paths.append(str(self.root_dir))
        elif self.root_dir.is_dir():
            for root, _, files in os.walk(self.root_dir):
                for f in sorted(files):
                    if Path(f).suffix.lower() in self.VALID_EXTENSIONS:
                        valid_paths.append(str(Path(root) / f))
        
        return valid_paths
    
    @staticmethod
    def _get_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self._cache:
            return self._cache[idx], 0
        
        try:
            path = self.image_paths[idx]
            with Image.open(path).convert('RGB') as img:
                tensor = self.transform(img)
                
                if len(self._cache) < self.cache_size:
                    self._cache[idx] = tensor
                
                return tensor, 0
        except Exception as e:
            print(f"  [警告] 加载图像失败: {path}, 错误: {e}")
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank, 0
    
    def clear_cache(self):
        self._cache.clear()


class DataManager:
    """数据管理器 - 处理大型数据集的分批加载"""
    
    def __init__(
        self,
        data_root: str,
        image_size: int = 128,
        batch_size: int = 4,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        
        self.train_loader = None
        self.val_loader = None
        self.train_size = 0
        self.val_size = 0
    
    def _scan_datasets(self) -> List[Path]:
        dataset_paths = []
        
        div2k_path = self.data_root / 'DIV2K-main' / 'DIV2K_train_HR'
        if div2k_path.exists():
            dataset_paths.append(div2k_path)
            print(f"  [DIV2K] 找到: {div2k_path}")
        
        coco_path = self.data_root / 'val2017'
        if coco_path.exists():
            dataset_paths.append(coco_path)
            print(f"  [COCO] 找到: {coco_path}")
        
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name not in ['DIV2K-main', 'val2017']:
                has_images = any(
                    f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
                    for f in item.rglob('*') if f.is_file()
                )
                if has_images:
                    dataset_paths.append(item)
                    print(f"  [自定义] 找到: {item}")
        
        return dataset_paths
    
    def setup(self) -> Tuple[DataLoader, DataLoader]:
        print("\n[数据加载] 扫描数据集...")
        
        dataset_paths = self._scan_datasets()
        
        if not dataset_paths:
            raise ValueError(f"未找到有效数据集: {self.data_root}")
        
        datasets = []
        for path in dataset_paths:
            ds = LargeImageDataset(
                str(path),
                image_size=self.image_size,
                cache_size=500
            )
            if len(ds) > 0:
                datasets.append(ds)
        
        if not datasets:
            raise ValueError("所有数据集为空")
        
        combined = ConcatDataset(datasets)
        total = len(combined)
        
        val_size = int(total * self.val_ratio)
        train_size = total - val_size
        
        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds = random_split(
            combined,
            [train_size, val_size],
            generator=generator
        )
        
        self.train_size = train_size
        self.val_size = val_size
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0
        )
        
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        print(f"\n[数据统计]")
        print(f"  总计: {total} 张")
        print(f"  训练: {train_size} 张 ({(1-self.val_ratio)*100:.0f}%)")
        print(f"  验证: {val_size} 张 ({self.val_ratio*100:.0f}%)")
        print(f"  批次大小: {self.batch_size}")
        print(f"  训练批次: {len(self.train_loader)}")
        print(f"  验证批次: {len(self.val_loader)}")
        
        return self.train_loader, self.val_loader


# ============================================================================
# 检查点管理
# ============================================================================

class CheckpointManager:
    """检查点管理器 - 支持断点续训"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_checkpoint = None
        self.best_accuracy = 0.0
    
    def save(
        self,
        epoch: int,
        model: SteganoGAN,
        optimizer_state: Dict,
        scheduler_state: Dict,
        metrics: Dict,
        is_best: bool = False
    ) -> str:
        checkpoint = {
            'epoch': epoch,
            'model_state': {
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'critic': model.critic.state_dict(),
            },
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            'metrics': metrics,
            'config': {
                'data_depth': model.data_depth,
                'hidden_size': model.encoder.hidden_size,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.best_checkpoint = checkpoint_path
            self.best_accuracy = metrics.get('val_acc', 0)
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  [保存] 最佳模型已保存 (准确率: {self.best_accuracy*100:.2f}%)")
        
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load(self, path: str, model: SteganoGAN, device: torch.device) -> Dict:
        print(f"\n[断点续训] 加载检查点: {path}")
        
        checkpoint = torch.load(path, map_location=device)
        
        model.encoder.load_state_dict(checkpoint['model_state']['encoder'])
        model.decoder.load_state_dict(checkpoint['model_state']['decoder'])
        model.critic.load_state_dict(checkpoint['model_state']['critic'])
        
        print(f"  已恢复到 Epoch {checkpoint['epoch']}")
        print(f"  之前最佳准确率: {checkpoint['metrics'].get('val_acc', 0)*100:.2f}%")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoints[self.max_checkpoints:]:
            old_checkpoint.unlink()
    
    def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: int(x.stem.split('_')[-1]),
            reverse=True
        )
        return str(checkpoints[0]) if checkpoints else None


# ============================================================================
# 训练器
# ============================================================================

class AdvancedTrainer:
    """高级训练器 - 支持动态学习率、监控、断点续训"""
    
    def __init__(
        self,
        model: SteganoGAN,
        device: torch.device,
        config: Dict,
        logger: logging.Logger,
        checkpoint_manager: CheckpointManager
    ):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        
        self.use_amp = config.get('use_amp', True) and device.type == 'cuda'
        self.scaler = get_grad_scaler() if self.use_amp else None
        
        enc_dec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        self.critic_opt = AdamW(
            model.critic.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.enc_dec_opt = AdamW(
            enc_dec_params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.critic_scheduler = CosineAnnealingWarmRestarts(
            self.critic_opt,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        self.enc_dec_scheduler = CosineAnnealingWarmRestarts(
            self.enc_dec_opt,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.plateau_scheduler = ReduceLROnPlateau(
            self.enc_dec_opt,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        self.best_acc = 0.0
        self.best_state = None
        self.patience_counter = 0
        self.history: List[Dict] = []
        self.start_epoch = 0
        self.accuracy_above_threshold_count = 0
        
        self.metrics_history = {
            'train_acc': [],
            'val_acc': [],
            'val_psnr': [],
            'val_bpp': [],
            'learning_rate': [],
            'epoch_time': [],
        }
    
    def _random_payload(self, cover: torch.Tensor) -> torch.Tensor:
        N, _, H, W = cover.size()
        return torch.zeros(
            (N, self.model.data_depth, H, W),
            device=self.device
        ).random_(0, 2)
    
    def _critic_score(self, image: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.model.critic(image))
    
    def _encode_decode(self, cover: torch.Tensor, quantize: bool = False):
        payload = self._random_payload(cover)
        generated = self.model.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0
        decoded = self.model.decoder(generated)
        return generated, payload, decoded
    
    def train_critic(self, loader: DataLoader) -> Dict:
        self.model.critic.train()
        cover_scores, gen_scores = [], []
        
        pbar = tqdm(loader, desc="  Critic", leave=False, ncols=100)
        for cover, _ in pbar:
            cover = cover.to(self.device, non_blocking=True)
            payload = self._random_payload(cover)
            
            with autocast(enabled=self.use_amp, device_type='cuda'):
                generated = self.model.encoder(cover, payload)
                loss = self._critic_score(generated) - self._critic_score(cover)
            
            self.critic_opt.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.critic_opt)
                self.scaler.update()
            else:
                loss.backward()
                self.critic_opt.step()
            
            with torch.no_grad():
                for p in self.model.critic.parameters():
                    p.clamp_(-0.1, 0.1)
            
            cover_scores.append(self._critic_score(cover).item())
            gen_scores.append(self._critic_score(generated).item())
        
        return {'cover_score': np.mean(cover_scores), 'gen_score': np.mean(gen_scores)}
    
    def train_enc_dec(self, loader: DataLoader) -> Dict:
        self.model.encoder.train()
        self.model.decoder.train()
        
        metrics = {'mse': 0.0, 'dec_loss': 0.0, 'acc': 0.0}
        n = 0
        
        pbar = tqdm(loader, desc="  Enc/Dec", leave=False, ncols=100)
        for cover, _ in pbar:
            cover = cover.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp, device_type='cuda'):
                generated, payload, decoded = self._encode_decode(cover)
                gen_score = self._critic_score(generated)
                
                mse = nn.functional.mse_loss(generated, cover)
                dec_loss = nn.functional.binary_cross_entropy_with_logits(decoded, payload)
                loss = self.config['encoder_weight'] * mse + dec_loss + gen_score
            
            self.enc_dec_opt.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.enc_dec_opt)
                nn.utils.clip_grad_norm_(
                    list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                    self.config['grad_clip']
                )
                self.scaler.step(self.enc_dec_opt)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                    self.config['grad_clip']
                )
                self.enc_dec_opt.step()
            
            with torch.no_grad():
                acc = ((decoded >= 0.0) == (payload >= 0.5)).float().mean().item()
            
            metrics['mse'] += mse.item()
            metrics['dec_loss'] += dec_loss.item()
            metrics['acc'] += acc
            n += 1
            
            pbar.set_postfix({'acc': f'{acc*100:.1f}%'})
        
        return {k: v / n for k, v in metrics.items()}
    
    def validate(self, loader: DataLoader) -> Dict:
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.critic.eval()
        
        metrics = {'mse': 0.0, 'dec_loss': 0.0, 'acc': 0.0}
        n = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc="  Valid", leave=False, ncols=100)
            for cover, _ in pbar:
                cover = cover.to(self.device, non_blocking=True)
                generated, payload, decoded = self._encode_decode(cover, quantize=True)
                
                metrics['mse'] += nn.functional.mse_loss(generated, cover).item()
                metrics['dec_loss'] += nn.functional.binary_cross_entropy_with_logits(
                    decoded, payload
                ).item()
                metrics['acc'] += ((decoded >= 0.0) == (payload >= 0.5)).float().mean().item()
                n += 1
        
        metrics = {k: v / n for k, v in metrics.items()}
        metrics['psnr'] = 10 * np.log10(4 / metrics['mse']) if metrics['mse'] > 0 else 100
        metrics['bpp'] = self.model.data_depth * (2 * metrics['acc'] - 1)
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("  SteganoGAN V3.0 训练器")
        self.logger.info("=" * 60)
        self.logger.info(f"  训练轮数: {epochs}")
        self.logger.info(f"  批次大小: {self.config['batch_size']}")
        self.logger.info(f"  学习率: {self.config['learning_rate']}")
        self.logger.info(f"  目标准确率: {self.config['target_accuracy']*100:.0f}%")
        self.logger.info(f"  混合精度: {'启用' if self.use_amp else '禁用'}")
        self.logger.info("=" * 60)
        
        total_start_time = time.time()
        stop_reason = None
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            epoch_start_time = time.time()
            
            self.logger.info(f"\nEpoch {epoch}/{epochs}")
            self.logger.info("-" * 40)
            
            critic_metrics = self.train_critic(train_loader)
            enc_dec_metrics = self.train_enc_dec(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.critic_scheduler.step()
            self.enc_dec_scheduler.step()
            self.plateau_scheduler.step(val_metrics['acc'])
            
            overfit_gap = enc_dec_metrics['acc'] - val_metrics['acc']
            current_lr = self.enc_dec_opt.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
            epoch_metrics = {
                'epoch': epoch,
                'train_acc': enc_dec_metrics['acc'],
                'val_acc': val_metrics['acc'],
                'val_psnr': val_metrics['psnr'],
                'val_bpp': val_metrics['bpp'],
                'overfit_gap': overfit_gap,
                'lr': current_lr,
                'epoch_time': epoch_time,
            }
            self.history.append(epoch_metrics)
            
            self.metrics_history['train_acc'].append(enc_dec_metrics['acc'])
            self.metrics_history['val_acc'].append(val_metrics['acc'])
            self.metrics_history['val_psnr'].append(val_metrics['psnr'])
            self.metrics_history['val_bpp'].append(val_metrics['bpp'])
            self.metrics_history['learning_rate'].append(current_lr)
            self.metrics_history['epoch_time'].append(epoch_time)
            
            self.logger.info(f"  训练准确率: {enc_dec_metrics['acc']*100:.2f}%")
            self.logger.info(f"  验证准确率: {val_metrics['acc']*100:.2f}%")
            self.logger.info(f"  验证 PSNR:  {val_metrics['psnr']:.2f} dB")
            self.logger.info(f"  验证 BPP:   {val_metrics['bpp']:.4f}")
            self.logger.info(f"  学习率:     {current_lr:.2e}")
            self.logger.info(f"  耗时:       {epoch_time:.1f}s")
            
            if val_metrics['acc'] >= self.config['target_accuracy']:
                self.accuracy_above_threshold_count += 1
                self.logger.info(
                    f"  [达标] 验证准确率 >= {self.config['target_accuracy']*100:.0f}% "
                    f"(连续 {self.accuracy_above_threshold_count}/3 轮)"
                )
                
                if self.accuracy_above_threshold_count >= 3:
                    stop_reason = f'验证准确率连续3轮超过{self.config["target_accuracy"]*100:.0f}%'
                    self.logger.info(f"  [停止] {stop_reason}")
            else:
                self.accuracy_above_threshold_count = 0
            
            if overfit_gap > self.config['overfit_threshold']:
                self.logger.warning(f"  [警告] 过拟合检测: Train-Val差距 {overfit_gap*100:.1f}%")
            
            is_best = val_metrics['acc'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['acc']
                self.best_state = {
                    'encoder': self.model.encoder.state_dict(),
                    'decoder': self.model.decoder.state_dict(),
                    'critic': self.model.critic.state_dict(),
                }
                self.patience_counter = 0
                self.logger.info(f"  [最佳] 新的最佳准确率: {self.best_acc*100:.2f}%")
            else:
                self.patience_counter += 1
            
            if epoch % self.config.get('checkpoint_interval', 5) == 0 or is_best:
                self.checkpoint_manager.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer_state={
                        'critic': self.critic_opt.state_dict(),
                        'enc_dec': self.enc_dec_opt.state_dict(),
                    },
                    scheduler_state={
                        'critic': self.critic_scheduler.state_dict(),
                        'enc_dec': self.enc_dec_scheduler.state_dict(),
                        'plateau': self.plateau_scheduler.state_dict(),
                    },
                    metrics=epoch_metrics,
                    is_best=is_best
                )
            
            if self.patience_counter >= self.config['early_stop_patience']:
                stop_reason = f'{self.config["early_stop_patience"]}轮无改善'
                self.logger.info(f"  [早停] {stop_reason}")
            
            if self.device.type == 'cuda':
                clear_cuda_cache()
            
            if stop_reason:
                break
        
        total_time = time.time() - total_start_time
        
        if self.best_state:
            self.model.encoder.load_state_dict(self.best_state['encoder'])
            self.model.decoder.load_state_dict(self.best_state['decoder'])
            self.model.critic.load_state_dict(self.best_state['critic'])
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("  训练完成")
        self.logger.info("=" * 60)
        self.logger.info(f"  总训练轮数: {len(self.history)}")
        self.logger.info(f"  最佳准确率: {self.best_acc*100:.2f}%")
        self.logger.info(f"  总耗时:     {total_time/60:.1f} 分钟")
        self.logger.info(f"  停止原因:   {stop_reason or '完成所有轮次'}")
        self.logger.info("=" * 60)
        
        return {
            'history': self.history,
            'best_accuracy': self.best_acc,
            'stop_reason': stop_reason,
            'total_time': total_time,
            'metrics_history': self.metrics_history,
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = self.checkpoint_manager.load(
            checkpoint_path, self.model, self.device
        )
        
        self.critic_opt.load_state_dict(checkpoint['optimizer_state']['critic'])
        self.enc_dec_opt.load_state_dict(checkpoint['optimizer_state']['enc_dec'])
        self.critic_scheduler.load_state_dict(checkpoint['scheduler_state']['critic'])
        self.enc_dec_scheduler.load_state_dict(checkpoint['scheduler_state']['enc_dec'])
        self.plateau_scheduler.load_state_dict(checkpoint['scheduler_state']['plateau'])
        
        self.start_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['metrics'].get('val_acc', 0)
        
        return checkpoint


# ============================================================================
# 模型评估
# ============================================================================

class ModelEvaluator:
    """模型评估器 - 生成性能报告"""
    
    def __init__(self, model: SteganoGAN, device: torch.device, logger: logging.Logger):
        self.model = model
        self.device = device
        self.logger = logger
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        self.logger.info("\n[模型评估] 开始评估...")
        
        self.model.encoder.eval()
        self.model.decoder.eval()
        
        all_metrics = {
            'accuracy': [],
            'psnr': [],
            'ssim': [],
            'bpp': [],
        }
        
        with torch.no_grad():
            for cover, _ in tqdm(val_loader, desc="评估中", ncols=100):
                cover = cover.to(self.device)
                
                N, _, H, W = cover.size()
                payload = torch.zeros(
                    (N, self.model.data_depth, H, W),
                    device=self.device
                ).random_(0, 2)
                
                generated = self.model.encoder(cover, payload)
                generated_q = (255.0 * (generated + 1.0) / 2.0).long()
                generated_q = 2.0 * generated_q.float() / 255.0 - 1.0
                
                decoded = self.model.decoder(generated_q)
                
                acc = ((decoded >= 0.0) == (payload >= 0.5)).float().mean().item()
                mse = nn.functional.mse_loss(generated_q, cover).item()
                psnr = 10 * np.log10(4 / mse) if mse > 0 else 100
                bpp = self.model.data_depth * (2 * acc - 1)
                
                all_metrics['accuracy'].append(acc)
                all_metrics['psnr'].append(psnr)
                all_metrics['bpp'].append(bpp)
        
        results = {
            'accuracy_mean': np.mean(all_metrics['accuracy']),
            'accuracy_std': np.std(all_metrics['accuracy']),
            'psnr_mean': np.mean(all_metrics['psnr']),
            'psnr_std': np.std(all_metrics['psnr']),
            'bpp_mean': np.mean(all_metrics['bpp']),
            'bpp_std': np.std(all_metrics['bpp']),
        }
        
        return results
    
    def generate_report(self, training_results: Dict, eval_results: Dict, output_path: str):
        report = []
        report.append("=" * 70)
        report.append("                    SteganoGAN V3.0 训练报告")
        report.append("=" * 70)
        report.append("")
        report.append("【训练概况】")
        report.append("-" * 50)
        report.append(f"  训练轮数:     {len(training_results['history'])}")
        report.append(f"  最佳准确率:   {training_results['best_accuracy']*100:.2f}%")
        report.append(f"  总训练时间:   {training_results['total_time']/60:.1f} 分钟")
        report.append(f"  停止原因:     {training_results['stop_reason'] or '完成所有轮次'}")
        report.append("")
        
        report.append("【模型性能】")
        report.append("-" * 50)
        report.append(f"  解码准确率:   {eval_results['accuracy_mean']*100:.2f}% ± {eval_results['accuracy_std']*100:.2f}%")
        report.append(f"  PSNR:         {eval_results['psnr_mean']:.2f} ± {eval_results['psnr_std']:.2f} dB")
        report.append(f"  BPP:          {eval_results['bpp_mean']:.4f} ± {eval_results['bpp_std']:.4f}")
        report.append("")
        
        report.append("【训练历史】")
        report.append("-" * 50)
        for record in training_results['history'][-5:]:
            report.append(
                f"  Epoch {record['epoch']:2d}: "
                f"Train={record['train_acc']*100:5.2f}% "
                f"Val={record['val_acc']*100:5.2f}% "
                f"PSNR={record['val_psnr']:5.2f}dB"
            )
        report.append("")
        
        report.append("【硬件信息】")
        report.append("-" * 50)
        report.append(f"  设备:         {self.device}")
        if self.device.type == 'cuda':
            report.append(f"  GPU:          {torch.cuda.get_device_name(0)}")
            report.append(f"  显存使用:     {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        report.append("")
        
        report.append("=" * 70)
        report.append(f"  报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"\n[报告] 已保存至: {output_path}")
        print("\n" + report_text)
        
        return report_text


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='SteganoGAN V3.0 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本训练
  python train.py --data_root ./research/data --epochs 20

  # 自定义参数
  python train.py --data_root ./data --epochs 30 --batch_size 8 --learning_rate 1e-4

  # 断点续训
  python train.py --resume ./checkpoints/checkpoint_epoch_10.pt

  # 使用特定GPU
  python train.py --data_root ./data --gpu 0
        """
    )
    
    parser.add_argument('--data_root', type=str, default='./research/data',
                        help='数据集根目录 (默认: ./research/data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数 (默认: 20)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小 (默认: 4)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='图像尺寸 (默认: 128)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='学习率 (默认: 2e-4)')
    parser.add_argument('--architecture', type=str, default='dense',
                        choices=['basic', 'dense', 'residual'],
                        help='模型架构 (默认: dense)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='隐藏层大小 (默认: 64)')
    parser.add_argument('--data_depth', type=int, default=1,
                        help='数据深度 (默认: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')
    parser.add_argument('--target_accuracy', type=float, default=0.90,
                        help='目标准确率 (默认: 0.90)')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='早停耐心值 (默认: 5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续训检查点路径')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU编号 (默认: 0)')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU训练')
    parser.add_argument('--no_amp', action='store_true',
                        help='禁用混合精度训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出')
    
    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("  SteganoGAN V3.0 训练脚本")
    print("=" * 60)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    logger = setup_logging(str(log_dir), args.verbose)
    logger.info("初始化训练环境...")
    
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.info("使用 CPU 设备")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(args.gpu)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    
    config = {
        'data_depth': args.data_depth,
        'hidden_size': args.hidden_size,
        'architecture': args.architecture,
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'encoder_weight': 100.0,
        'target_accuracy': args.target_accuracy,
        'early_stop_patience': args.early_stop_patience,
        'overfit_threshold': 0.08,
        'num_workers': args.num_workers,
        'val_ratio': 0.1,
        'seed': args.seed,
        'use_amp': not args.no_amp and device.type == 'cuda',
        'checkpoint_interval': 5,
    }
    
    logger.info("配置参数:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    logger.info("\n加载数据...")
    data_manager = DataManager(
        data_root=args.data_root,
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    train_loader, val_loader = data_manager.setup()
    
    logger.info("\n构建模型...")
    arch = ARCHITECTURES[config['architecture']]
    model = SteganoGAN(
        data_depth=config['data_depth'],
        encoder=arch['encoder'],
        decoder=arch['decoder'],
        critic=BasicCritic,
        cuda=device.type == 'cuda',
        verbose=args.verbose,
        hidden_size=config['hidden_size']
    )
    
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    
    trainer = AdvancedTrainer(
        model=model,
        device=device,
        config=config,
        logger=logger,
        checkpoint_manager=checkpoint_manager
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    logger.info("\n开始训练...")
    results = trainer.fit(train_loader, val_loader, config['max_epochs'])
    
    logger.info("\n评估模型...")
    evaluator = ModelEvaluator(model, device, logger)
    eval_results = evaluator.evaluate(val_loader)
    
    final_model_path = output_dir / 'final_model.steg'
    model.save(str(final_model_path))
    logger.info(f"\n[保存] 最终模型: {final_model_path}")
    
    report_path = output_dir / 'training_report.txt'
    evaluator.generate_report(results, eval_results, str(report_path))
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'training_history': results['history'],
            'evaluation': eval_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"[保存] 训练指标: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("  训练完成!")
    print("=" * 60)
    print(f"  最佳准确率: {results['best_accuracy']*100:.2f}%")
    print(f"  模型路径:   {final_model_path}")
    print(f"  报告路径:   {report_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
