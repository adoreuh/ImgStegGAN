# -*- coding: utf-8 -*-
"""
SteganoGAN 训练器 V3.0
优化训练策略，限制最大20轮，增强过拟合监控，准确率达标自动停止
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, List

from steganogan.compat import autocast, get_grad_scaler, clear_cuda_cache


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器 - PyTorch 2.x
    
    V3.0 改进:
    - 最大训练轮次限制为20轮
    - 早停策略防止过拟合
    - 权重衰减正则化
    - 余弦退火学习率调度
    - 训练/验证准确率差距监控
    - 验证集准确率超过90%自动停止
    - 连续N个epoch稳定达标触发停止
    - 完整日志记录
    """
    
    MAX_EPOCHS = 20
    EARLY_STOP_PATIENCE = 5
    OVERFIT_THRESHOLD = 0.08
    TARGET_ACCURACY = 0.90
    ACCURACY_STABILITY_EPOCHS = 3
    
    def __init__(
        self,
        model,
        device: torch.device,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        encoder_weight: float = 100.0,
        target_accuracy: float = 0.90,
        accuracy_stability_epochs: int = 3
    ):
        self.model = model
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.grad_clip = grad_clip
        self.encoder_weight = encoder_weight
        
        self.target_accuracy = target_accuracy
        self.accuracy_stability_epochs = accuracy_stability_epochs
        
        enc_dec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        self.critic_opt = AdamW(model.critic.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.enc_dec_opt = AdamW(enc_dec_params, lr=learning_rate, weight_decay=weight_decay)
        
        self.scaler = get_grad_scaler() if self.use_amp else None
        
        self.best_acc = 0.0
        self.best_state = None
        self.patience_counter = 0
        
        self.history: List[Dict] = []
        self.stop_reason: Optional[str] = None
        self.accuracy_above_threshold_count = 0
        
        self._init_logger()
    
    def _init_logger(self):
        self.training_log = {
            'start_time': None,
            'end_time': None,
            'total_epochs': 0,
            'stop_reason': None,
            'best_accuracy': 0.0,
            'accuracy_history': [],
            'target_accuracy': self.target_accuracy,
            'stability_epochs': self.accuracy_stability_epochs,
            'events': []
        }
    
    def _log_event(self, event_type: str, message: str, details: Optional[Dict] = None):
        event = {
            'timestamp': datetime.now().isoformat(),
            'epoch': len(self.history) + 1,
            'type': event_type,
            'message': message,
            'details': details or {}
        }
        self.training_log['events'].append(event)
        logger.info(f"[{event_type}] {message}")
    
    def _check_accuracy_stopping(self, val_acc: float) -> bool:
        if val_acc >= self.target_accuracy:
            self.accuracy_above_threshold_count += 1
            self._log_event(
                'ACCURACY_CHECK',
                f'验证准确率 {val_acc*100:.2f}% >= 目标 {self.target_accuracy*100:.0f}%',
                {
                    'current_acc': val_acc,
                    'target_acc': self.target_accuracy,
                    'consecutive_epochs': self.accuracy_above_threshold_count,
                    'required_epochs': self.accuracy_stability_epochs
                }
            )
            
            if self.accuracy_above_threshold_count >= self.accuracy_stability_epochs:
                self._log_event(
                    'STOP_TRIGGERED',
                    f'连续 {self.accuracy_stability_epochs} 轮验证准确率超过 {self.target_accuracy*100:.0f}%，触发停止',
                    {
                        'final_accuracy': val_acc,
                        'consecutive_epochs': self.accuracy_above_threshold_count,
                        'stop_condition': 'accuracy_threshold_reached'
                    }
                )
                return True
        else:
            if self.accuracy_above_threshold_count > 0:
                self._log_event(
                    'ACCURACY_RESET',
                    f'验证准确率 {val_acc*100:.2f}% < 目标 {self.target_accuracy*100:.0f}%，重置计数器',
                    {
                        'previous_count': self.accuracy_above_threshold_count,
                        'current_acc': val_acc
                    }
                )
            self.accuracy_above_threshold_count = 0
        
        return False
    
    def _random_payload(self, cover: torch.Tensor) -> torch.Tensor:
        N, _, H, W = cover.size()
        return torch.zeros((N, self.model.data_depth, H, W), device=self.device).random_(0, 2)
    
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
    
    def train_critic(self, loader) -> dict:
        self.model.critic.train()
        cover_scores, gen_scores = [], []
        
        for cover, _ in tqdm(loader, desc="Critic", leave=False):
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
    
    def train_enc_dec(self, loader) -> dict:
        self.model.encoder.train()
        self.model.decoder.train()
        
        metrics = {'mse': 0.0, 'dec_loss': 0.0, 'acc': 0.0}
        n = 0
        
        for cover, _ in tqdm(loader, desc="Enc/Dec", leave=False):
            cover = cover.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp, device_type='cuda'):
                generated, payload, decoded = self._encode_decode(cover)
                gen_score = self._critic_score(generated)
                
                mse = nn.functional.mse_loss(generated, cover)
                dec_loss = nn.functional.binary_cross_entropy_with_logits(decoded, payload)
                loss = self.encoder_weight * mse + dec_loss + gen_score
            
            self.enc_dec_opt.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.enc_dec_opt)
                nn.utils.clip_grad_norm_(
                    list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                    self.grad_clip
                )
                self.scaler.step(self.enc_dec_opt)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                    self.grad_clip
                )
                self.enc_dec_opt.step()
            
            with torch.no_grad():
                acc = ((decoded >= 0.0) == (payload >= 0.5)).float().mean().item()
            
            metrics['mse'] += mse.item()
            metrics['dec_loss'] += dec_loss.item()
            metrics['acc'] += acc
            n += 1
        
        return {k: v / n for k, v in metrics.items()}
    
    def validate(self, loader) -> dict:
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.critic.eval()
        
        metrics = {'mse': 0.0, 'dec_loss': 0.0, 'acc': 0.0}
        n = 0
        
        with torch.no_grad():
            for cover, _ in tqdm(loader, desc="Valid", leave=False):
                cover = cover.to(self.device, non_blocking=True)
                generated, payload, decoded = self._encode_decode(cover, quantize=True)
                
                metrics['mse'] += nn.functional.mse_loss(generated, cover).item()
                metrics['dec_loss'] += nn.functional.binary_cross_entropy_with_logits(decoded, payload).item()
                metrics['acc'] += ((decoded >= 0.0) == (payload >= 0.5)).float().mean().item()
                n += 1
        
        metrics = {k: v / n for k, v in metrics.items()}
        metrics['psnr'] = 10 * np.log10(4 / metrics['mse']) if metrics['mse'] > 0 else 100
        metrics['bpp'] = self.model.data_depth * (2 * metrics['acc'] - 1)
        return metrics
    
    def fit(self, train_loader, val_loader, epochs: int = None) -> dict:
        self.training_log['start_time'] = datetime.now().isoformat()
        epochs = min(epochs or self.MAX_EPOCHS, self.MAX_EPOCHS)
        
        critic_scheduler = CosineAnnealingLR(self.critic_opt, T_max=epochs, eta_min=1e-6)
        enc_dec_scheduler = CosineAnnealingLR(self.enc_dec_opt, T_max=epochs, eta_min=1e-6)
        
        print(f"\n{'='*60}")
        print(f"  SteganoGAN V3.0 训练器")
        print(f"{'='*60}")
        print(f"  最大训练轮数: {epochs}")
        print(f"  目标准确率: {self.target_accuracy*100:.0f}%")
        print(f"  稳定达标轮数: {self.accuracy_stability_epochs}")
        print(f"  早停耐心值: {self.EARLY_STOP_PATIENCE}")
        print(f"  过拟合阈值: {self.OVERFIT_THRESHOLD*100:.0f}%")
        print(f"{'='*60}\n")
        
        self._log_event('TRAINING_START', f'开始训练，最大 {epochs} 轮', {
            'max_epochs': epochs,
            'target_accuracy': self.target_accuracy,
            'stability_epochs': self.accuracy_stability_epochs
        })
        
        stop_training = False
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
            
            critic_metrics = self.train_critic(train_loader)
            enc_dec_metrics = self.train_enc_dec(train_loader)
            val_metrics = self.validate(val_loader)
            
            critic_scheduler.step()
            enc_dec_scheduler.step()
            
            overfit_gap = enc_dec_metrics['acc'] - val_metrics['acc']
            
            epoch_metrics = {
                'epoch': epoch,
                'train_acc': enc_dec_metrics['acc'],
                'val_acc': val_metrics['acc'],
                'val_psnr': val_metrics['psnr'],
                'val_bpp': val_metrics['bpp'],
                'overfit_gap': overfit_gap,
                'lr': self.enc_dec_opt.param_groups[0]['lr'],
                'above_target': val_metrics['acc'] >= self.target_accuracy
            }
            self.history.append(epoch_metrics)
            self.training_log['accuracy_history'].append({
                'epoch': epoch,
                'train_acc': enc_dec_metrics['acc'],
                'val_acc': val_metrics['acc']
            })
            
            print(f"  训练准确率: {enc_dec_metrics['acc']*100:.2f}%")
            print(f"  验证准确率: {val_metrics['acc']*100:.2f}%")
            print(f"  验证 PSNR:  {val_metrics['psnr']:.2f} dB")
            print(f"  学习率:     {self.enc_dec_opt.param_groups[0]['lr']:.2e}")
            
            if val_metrics['acc'] >= self.target_accuracy:
                print(f"  [达标] 验证准确率 >= {self.target_accuracy*100:.0f}% (连续 {self.accuracy_above_threshold_count + 1}/{self.accuracy_stability_epochs} 轮)")
            
            if overfit_gap > self.OVERFIT_THRESHOLD:
                print(f"  [警告] 过拟合检测: Train-Val差距 {overfit_gap*100:.1f}%")
                self._log_event('OVERFIT_WARNING', f'检测到过拟合，差距 {overfit_gap*100:.1f}%', {
                    'overfit_gap': overfit_gap,
                    'train_acc': enc_dec_metrics['acc'],
                    'val_acc': val_metrics['acc']
                })
            
            if val_metrics['acc'] > self.best_acc:
                self.best_acc = val_metrics['acc']
                self.best_state = {
                    'encoder': self.model.encoder.state_dict(),
                    'decoder': self.model.decoder.state_dict(),
                    'critic': self.model.critic.state_dict()
                }
                self.patience_counter = 0
                print(f"  [最佳] 新的最佳准确率: {self.best_acc*100:.2f}%")
                self._log_event('BEST_MODEL', f'新的最佳准确率 {self.best_acc*100:.2f}%', {
                    'best_acc': self.best_acc,
                    'epoch': epoch
                })
            else:
                self.patience_counter += 1
            
            if self._check_accuracy_stopping(val_metrics['acc']):
                self.stop_reason = f'验证准确率连续 {self.accuracy_stability_epochs} 轮超过 {self.target_accuracy*100:.0f}%'
                stop_training = True
            
            if self.patience_counter >= self.EARLY_STOP_PATIENCE:
                self.stop_reason = f'{self.EARLY_STOP_PATIENCE} 轮无改善'
                self._log_event('EARLY_STOP', f'早停触发: {self.EARLY_STOP_PATIENCE} 轮无改善', {
                    'patience_counter': self.patience_counter
                })
                stop_training = True
            
            if stop_training:
                break
            
            if self.device.type == 'cuda':
                clear_cuda_cache()
        
        self.training_log['end_time'] = datetime.now().isoformat()
        self.training_log['total_epochs'] = len(self.history)
        self.training_log['stop_reason'] = self.stop_reason or '达到最大训练轮数'
        self.training_log['best_accuracy'] = self.best_acc
        
        if self.best_state:
            self.model.encoder.load_state_dict(self.best_state['encoder'])
            self.model.decoder.load_state_dict(self.best_state['decoder'])
            self.model.critic.load_state_dict(self.best_state['critic'])
        
        print(f"\n{'='*60}")
        print(f"  训练完成")
        print(f"{'='*60}")
        print(f"  总训练轮数: {len(self.history)}")
        print(f"  最佳准确率: {self.best_acc*100:.2f}%")
        print(f"  停止原因: {self.training_log['stop_reason']}")
        print(f"{'='*60}\n")
        
        self._log_event('TRAINING_END', f'训练完成: {self.training_log["stop_reason"]}', {
            'total_epochs': len(self.history),
            'best_accuracy': self.best_acc,
            'stop_reason': self.training_log['stop_reason']
        })
        
        return {
            'history': self.history,
            'log': self.training_log,
            'best_accuracy': self.best_acc,
            'stop_reason': self.training_log['stop_reason']
        }
    
    def get_training_summary(self) -> str:
        lines = [
            "=" * 50,
            "训练摘要报告",
            "=" * 50,
            f"开始时间: {self.training_log['start_time']}",
            f"结束时间: {self.training_log['end_time']}",
            f"总训练轮数: {self.training_log['total_epochs']}",
            f"最佳准确率: {self.training_log['best_accuracy']*100:.2f}%",
            f"目标准确率: {self.training_log['target_accuracy']*100:.0f}%",
            f"停止原因: {self.training_log['stop_reason']}",
            "",
            "准确率历史:",
        ]
        
        for record in self.training_log['accuracy_history']:
            lines.append(f"  Epoch {record['epoch']}: Train={record['train_acc']*100:.2f}%, Val={record['val_acc']*100:.2f}%")
        
        lines.extend([
            "",
            "关键事件:",
        ])
        
        for event in self.training_log['events']:
            lines.append(f"  [{event['type']}] {event['message']}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
