# -*- coding: utf-8 -*-
"""
Qwen3-SteganoGAN 训练脚本
针对Flickr2K数据集进行训练

目标:
1. 100%数据可提取
2. 50%+速度提升
3. <10%大小增长
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
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings('ignore')

from steganogan.qwen_integration import QwenSteganoGAN, QwenSteganoConfig


VERSION = "1.0.0"


def setup_logging(log_dir: str, verbose: bool = True) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'qwen_training_{timestamp}.log'
    
    logger = logging.getLogger('QwenSteganoGAN')
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


class Flickr2KDataset(Dataset):
    """Flickr2K数据集加载器"""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(self, root_dir: str, image_size: int = 128, transform=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform or self._get_transform(image_size)
        
        self.image_paths = self._collect_paths()
        print(f"  [数据集] 加载完成: {len(self.image_paths)} 张图像")
    
    def _collect_paths(self) -> List[str]:
        valid_paths = []
        
        sample_dir = self.root_dir / 'sample' / 'image'
        if sample_dir.exists():
            for f in sorted(sample_dir.iterdir()):
                if f.suffix.lower() in self.VALID_EXTENSIONS:
                    valid_paths.append(str(f))
            if valid_paths:
                return valid_paths
        
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
        try:
            path = self.image_paths[idx]
            with Image.open(path).convert('RGB') as img:
                tensor = self.transform(img)
                return tensor, 0
        except Exception as e:
            print(f"  [警告] 加载图像失败: {path}, 错误: {e}")
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank, 0


class QwenTrainer:
    """QwenSteganoGAN训练器"""
    
    def __init__(
        self,
        model: QwenSteganoGAN,
        device: torch.device,
        config: Dict,
        logger: logging.Logger
    ):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger
        
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
            self.critic_opt, T_0=10, T_mult=2, eta_min=1e-6
        )
        self.enc_dec_scheduler = CosineAnnealingWarmRestarts(
            self.enc_dec_opt, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.best_acc = 0.0
        self.best_state = None
        self.history: List[Dict] = []
    
    def _random_payload(self, cover: torch.Tensor) -> torch.Tensor:
        N, _, H, W = cover.size()
        return torch.zeros(
            (N, self.model.data_depth, H, W),
            device=self.device
        ).random_(0, 2)
    
    def train_critic(self, loader: DataLoader) -> Dict:
        self.model.critic.train()
        cover_scores, gen_scores = [], []
        
        pbar = tqdm(loader, desc="  Critic", leave=False, ncols=100)
        for cover, _ in pbar:
            cover = cover.to(self.device, non_blocking=True)
            payload = self._random_payload(cover)
            
            generated = self.model.encoder(cover, payload)
            cover_score = torch.mean(self.model.critic(cover))
            gen_score = torch.mean(self.model.critic(generated))
            
            loss = cover_score - gen_score
            
            self.critic_opt.zero_grad()
            loss.backward()
            self.critic_opt.step()
            
            with torch.no_grad():
                for p in self.model.critic.parameters():
                    p.clamp_(-0.1, 0.1)
            
            cover_scores.append(cover_score.item())
            gen_scores.append(gen_score.item())
        
        return {'cover_score': np.mean(cover_scores), 'gen_score': np.mean(gen_scores)}
    
    def train_enc_dec(self, loader: DataLoader) -> Dict:
        self.model.encoder.train()
        self.model.decoder.train()
        
        metrics = {'mse': 0.0, 'dec_loss': 0.0, 'acc': 0.0}
        n = 0
        
        pbar = tqdm(loader, desc="  Enc/Dec", leave=False, ncols=100)
        for cover, _ in pbar:
            cover = cover.to(self.device, non_blocking=True)
            
            payload = self._random_payload(cover)
            generated = self.model.encoder(cover, payload)
            decoded = self.model.decoder(generated)
            
            mse = nn.functional.mse_loss(generated, cover)
            dec_loss = nn.functional.binary_cross_entropy_with_logits(decoded, payload)
            gen_score = torch.mean(self.model.critic(generated))
            
            loss = self.config['encoder_weight'] * mse + dec_loss + gen_score
            
            self.enc_dec_opt.zero_grad()
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
                
                payload = self._random_payload(cover)
                generated = self.model.encoder(cover, payload)
                generated_q = (255.0 * (generated + 1.0) / 2.0).long()
                generated_q = 2.0 * generated_q.float() / 255.0 - 1.0
                
                decoded = self.model.decoder(generated_q)
                
                metrics['mse'] += nn.functional.mse_loss(generated_q, cover).item()
                metrics['dec_loss'] += nn.functional.binary_cross_entropy_with_logits(
                    decoded, payload
                ).item()
                metrics['acc'] += ((decoded >= 0.0) == (payload >= 0.5)).float().mean().item()
                n += 1
        
        metrics = {k: v / n for k, v in metrics.items()}
        metrics['psnr'] = 10 * np.log10(4 / metrics['mse']) if metrics['mse'] > 0 else 100
        metrics['bpp'] = self.model.data_depth * (2 * metrics['acc'] - 1)
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("  QwenSteganoGAN 训练器")
        self.logger.info("=" * 60)
        self.logger.info(f"  训练轮数: {epochs}")
        self.logger.info(f"  批次大小: {self.config['batch_size']}")
        self.logger.info(f"  学习率: {self.config['learning_rate']}")
        self.logger.info(f"  目标准确率: {self.config['target_accuracy']*100:.0f}%")
        self.logger.info("=" * 60)
        
        total_start_time = time.time()
        stop_reason = None
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            self.logger.info(f"\nEpoch {epoch}/{epochs}")
            self.logger.info("-" * 40)
            
            critic_metrics = self.train_critic(train_loader)
            enc_dec_metrics = self.train_enc_dec(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.critic_scheduler.step()
            self.enc_dec_scheduler.step()
            
            current_lr = self.enc_dec_opt.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
            epoch_metrics = {
                'epoch': epoch,
                'train_acc': enc_dec_metrics['acc'],
                'val_acc': val_metrics['acc'],
                'val_psnr': val_metrics['psnr'],
                'val_bpp': val_metrics['bpp'],
                'lr': current_lr,
                'epoch_time': epoch_time,
            }
            self.history.append(epoch_metrics)
            
            self.logger.info(f"  训练准确率: {enc_dec_metrics['acc']*100:.2f}%")
            self.logger.info(f"  验证准确率: {val_metrics['acc']*100:.2f}%")
            self.logger.info(f"  验证 PSNR:  {val_metrics['psnr']:.2f} dB")
            self.logger.info(f"  验证 BPP:   {val_metrics['bpp']:.4f}")
            self.logger.info(f"  学习率:     {current_lr:.2e}")
            self.logger.info(f"  耗时:       {epoch_time:.1f}s")
            
            if val_metrics['acc'] >= self.config['target_accuracy']:
                self.logger.info(f"  [达标] 验证准确率 >= {self.config['target_accuracy']*100:.0f}%")
            
            is_best = val_metrics['acc'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['acc']
                self.best_state = {
                    'encoder': self.model.encoder.state_dict(),
                    'decoder': self.model.decoder.state_dict(),
                    'critic': self.model.critic.state_dict(),
                }
                self.logger.info(f"  [最佳] 新的最佳准确率: {self.best_acc*100:.2f}%")
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
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
        self.logger.info("=" * 60)
        
        return {
            'history': self.history,
            'best_accuracy': self.best_acc,
            'total_time': total_time,
        }


def test_model(model: QwenSteganoGAN, test_images: List[str], output_dir: str, logger: logging.Logger):
    """测试模型性能"""
    logger.info("\n" + "=" * 60)
    logger.info("  性能测试")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_messages = [
        "Hello, QwenSteganoGAN!",
        "这是一条中文测试消息，用于验证模型的编码和解码能力。",
        "Test message with numbers: 1234567890",
        "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
    ]
    
    results = {
        'total_tests': 0,
        'successful': 0,
        'failed': 0,
        'encode_times': [],
        'decode_times': [],
        'size_increases': [],
    }
    
    for img_path in test_images[:5]:
        if not os.path.exists(img_path):
            continue
        
        for msg in test_messages:
            results['total_tests'] += 1
            
            try:
                output_path = str(output_dir / f"test_{results['total_tests']}.png")
                
                start_time = time.time()
                model.encode(img_path, output_path, msg)
                encode_time = time.time() - start_time
                results['encode_times'].append(encode_time)
                
                start_time = time.time()
                decoded_msg = model.decode(output_path)
                decode_time = time.time() - start_time
                results['decode_times'].append(decode_time)
                
                orig_size = os.path.getsize(img_path)
                new_size = os.path.getsize(output_path)
                size_increase = (new_size - orig_size) / orig_size * 100
                results['size_increases'].append(size_increase)
                
                if decoded_msg == msg:
                    results['successful'] += 1
                    logger.info(f"  [成功] {Path(img_path).name}: '{msg[:20]}...' "
                              f"编码={encode_time:.3f}s 解码={decode_time:.3f}s "
                              f"大小+{size_increase:.1f}%")
                else:
                    results['failed'] += 1
                    logger.warning(f"  [失败] 消息不匹配: 期望 '{msg[:20]}...' 实际 '{decoded_msg[:20] if decoded_msg else 'None'}...'")
            
            except Exception as e:
                results['failed'] += 1
                logger.error(f"  [错误] {Path(img_path).name}: {e}")
    
    logger.info("\n" + "-" * 40)
    logger.info(f"  总测试数: {results['total_tests']}")
    logger.info(f"  成功数: {results['successful']}")
    logger.info(f"  失败数: {results['failed']}")
    logger.info(f"  成功率: {results['successful']/results['total_tests']*100:.1f}%")
    
    if results['encode_times']:
        logger.info(f"  平均编码时间: {np.mean(results['encode_times']):.3f}s")
        logger.info(f"  平均解码时间: {np.mean(results['decode_times']):.3f}s")
        logger.info(f"  平均大小增长: {np.mean(results['size_increases']):.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='QwenSteganoGAN 训练脚本')
    parser.add_argument('--data_root', type=str, 
                       default='C:/Users/shg/Desktop/SteganoGAN-master/research/data',
                       help='数据集根目录')
    parser.add_argument('--output_dir', type=str, 
                       default='C:/Users/shg/Desktop/SteganoGAN-master/output_qwen',
                       help='输出目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--image_size', type=int, default=128, help='图像尺寸')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--data_depth', type=int, default=1, help='数据深度')
    parser.add_argument('--target_accuracy', type=float, default=0.95, help='目标准确率')
    parser.add_argument('--cpu', action='store_true', help='使用CPU训练')
    parser.add_argument('--test_only', action='store_true', help='仅运行测试')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--verbose', action='store_true', default=True, help='详细输出')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  QwenSteganoGAN 训练脚本 v" + VERSION)
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    logger = setup_logging(str(log_dir), args.verbose)
    
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.info("使用 CPU 设备")
    else:
        device = torch.device('cuda:0')
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    
    config = {
        'data_depth': args.data_depth,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'encoder_weight': 100.0,
        'target_accuracy': args.target_accuracy,
    }
    
    qwen_config = QwenSteganoConfig(
        data_depth=args.data_depth,
        hidden_size=args.hidden_size,
    )
    
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"\n加载预训练模型: {args.model_path}")
        model = QwenSteganoGAN.load(args.model_path, cuda=device.type=='cuda', verbose=args.verbose)
    else:
        model = QwenSteganoGAN(
            data_depth=args.data_depth,
            hidden_size=args.hidden_size,
            cuda=device.type == 'cuda',
            verbose=args.verbose,
            config=qwen_config
        )
    
    if not args.test_only:
        logger.info("\n加载数据集...")
        dataset = Flickr2KDataset(
            root_dir=args.data_root,
            image_size=args.image_size
        )
        
        if len(dataset) == 0:
            logger.error(f"未找到图像数据: {args.data_root}")
            return
        
        val_size = max(1, int(len(dataset) * 0.1))
        train_size = len(dataset) - val_size
        
        train_ds, val_ds = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == 'cuda',
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == 'cuda',
        )
        
        logger.info(f"训练集: {train_size} 张, 验证集: {val_size} 张")
        
        trainer = QwenTrainer(model, device, config, logger)
        
        logger.info("\n开始训练...")
        results = trainer.fit(train_loader, val_loader, args.epochs)
        
        model_path = output_dir / 'qwen_steganogan.steg'
        model.save(str(model_path))
        logger.info(f"\n模型已保存: {model_path}")
        
        metrics_path = output_dir / 'training_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    sample_dir = Path(args.data_root) / 'Flickr2K' / 'sample' / 'image'
    test_images = []
    if sample_dir.exists():
        for f in sorted(sample_dir.iterdir())[:10]:
            if f.suffix.lower() in {'.jpg', '.png'}:
                test_images.append(str(f))
    
    if test_images:
        test_results = test_model(model, test_images, str(output_dir / 'test_output'), logger)
        
        test_report_path = output_dir / 'test_report.json'
        with open(test_report_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("  训练/测试完成!")
    print("=" * 60)
    print(f"  输出目录: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
