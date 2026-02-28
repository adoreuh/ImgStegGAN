# -*- coding: utf-8 -*-
"""
SteganoGAN 核心模型 V3.0
仅支持 PyTorch 2.x
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gc
import inspect
import warnings
from typing import Optional, List

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from steganogan.utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits
from steganogan.compat import torch_load, clear_cuda_cache, autocast, compile_model


class SteganoGAN:
    """SteganoGAN 隐写模型 - PyTorch 2.x
    
    V3.0 优化版本:
    - 智能图像缩放：大图自动缩放到 max_process_size
    - 性能优化：使用 inference_mode 加速推理
    - 内存优化：减少不必要的内存拷贝
    """
    
    MAX_PROCESS_SIZE = 256
    
    def __init__(self, data_depth, encoder, decoder, critic,
                 cuda=False, verbose=False, log_dir=None, 
                 max_process_size=None, auto_scale=True, **kwargs):
        self.verbose = verbose
        self.data_depth = data_depth
        self.max_process_size = max_process_size or self.MAX_PROCESS_SIZE
        self.auto_scale = auto_scale
        
        kwargs['data_depth'] = data_depth
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.set_device(cuda)
        
        self.critic_optimizer = None
        self.decoder_optimizer = None
        self.fit_metrics = None
        self.history = []
        self.log_dir = log_dir
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def _get_instance(self, class_or_instance, kwargs):
        if not inspect.isclass(class_or_instance):
            return class_or_instance
        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        return class_or_instance(**{arg: kwargs[arg] for arg in argspec})
    
    def set_device(self, cuda=True, compile_models=False):
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')
        
        if self.verbose:
            print(f'使用 {"CUDA" if self.cuda else "CPU"} 设备')
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)
    
    def _random_data(self, cover):
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)
    
    def _encode_decode(self, cover, quantize=False):
        payload = self._random_data(cover)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0
        return generated, payload, self.decoder(generated)
    
    def _critic(self, image):
        return torch.mean(self.critic(image))
    
    def _get_optimizers(self):
        params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        return Adam(self.critic.parameters(), lr=1e-4), Adam(params, lr=1e-4)
    
    def _coding_scores(self, cover, generated, payload, decoded):
        mse = mse_loss(generated, cover)
        bce = binary_cross_entropy_with_logits(decoded, payload)
        acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
        return mse, bce, acc
    
    def fit(self, train, validate, epochs=5):
        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0
        
        METRIC_FIELDS = [
            'val.encoder_mse', 'val.decoder_loss', 'val.decoder_acc',
            'val.cover_score', 'val.generated_score', 'val.ssim', 'val.psnr', 'val.bpp',
            'train.encoder_mse', 'train.decoder_loss', 'train.decoder_acc',
            'train.cover_score', 'train.generated_score'
        ]
        
        for epoch in range(1, epochs + 1):
            self.epochs += 1
            metrics = {field: [] for field in METRIC_FIELDS}
            
            if self.verbose:
                print(f'Epoch {self.epochs}/{self.epochs + epochs - epoch}')
            
            for cover, _ in tqdm(train, disable=not self.verbose, desc='Critic'):
                cover = cover.to(self.device)
                payload = self._random_data(cover)
                generated = self.encoder(cover, payload)
                cover_score = self._critic(cover)
                gen_score = self._critic(generated)
                
                self.critic_optimizer.zero_grad()
                (cover_score - gen_score).backward()
                self.critic_optimizer.step()
                
                for p in self.critic.parameters():
                    p.data.clamp_(-0.1, 0.1)
                
                metrics['train.cover_score'].append(cover_score.item())
                metrics['train.generated_score'].append(gen_score.item())
            
            for cover, _ in tqdm(train, disable=not self.verbose, desc='Enc/Dec'):
                cover = cover.to(self.device)
                generated, payload, decoded = self._encode_decode(cover)
                mse, bce, acc = self._coding_scores(cover, generated, payload, decoded)
                gen_score = self._critic(generated)
                
                self.decoder_optimizer.zero_grad()
                (100.0 * mse + bce + gen_score).backward()
                self.decoder_optimizer.step()
                
                metrics['train.encoder_mse'].append(mse.item())
                metrics['train.decoder_loss'].append(bce.item())
                metrics['train.decoder_acc'].append(acc.item())
            
            for cover, _ in tqdm(validate, disable=not self.verbose, desc='Valid'):
                cover = cover.to(self.device)
                generated, payload, decoded = self._encode_decode(cover, quantize=True)
                mse, bce, acc = self._coding_scores(cover, generated, payload, decoded)
                
                metrics['val.encoder_mse'].append(mse.item())
                metrics['val.decoder_loss'].append(bce.item())
                metrics['val.decoder_acc'].append(acc.item())
                metrics['val.cover_score'].append(self._critic(cover).item())
                metrics['val.generated_score'].append(self._critic(generated).item())
                metrics['val.ssim'].append(ssim(cover, generated).item())
                metrics['val.psnr'].append(10 * torch.log10(4 / mse).item())
                metrics['val.bpp'].append(self.data_depth * (2 * acc.item() - 1))
            
            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch
            
            if self.log_dir:
                self.history.append(self.fit_metrics)
                self.save(os.path.join(self.log_dir, f'{self.epochs}.steg'))
            
            if self.cuda:
                clear_cuda_cache()
            gc.collect()
    
    def _calculate_target_size(self, width, height):
        if not self.auto_scale:
            return width, height, False
        
        max_dim = max(width, height)
        if max_dim <= self.max_process_size:
            return width, height, False
        
        scale = self.max_process_size / max_dim
        new_size = int(min(width, height) * scale)
        new_size = max(32, (new_size // 32) * 32)
        
        return new_size, new_size, True
    
    def _make_payload(self, width, height, depth, text):
        message = text_to_bits(text) + [0] * 32
        payload = message
        while len(payload) < width * height * depth:
            payload += message
        return torch.FloatTensor(payload[:width * height * depth]).view(1, depth, height, width)
    
    @torch.inference_mode()
    def encode(self, cover, output, text):
        pil_image = Image.open(cover).convert('RGB')
        orig_width, orig_height = pil_image.size
        
        target_width, target_height, was_scaled = self._calculate_target_size(orig_width, orig_height)
        
        if was_scaled:
            pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            if self.verbose:
                print(f"[编码] 缩放: {orig_width}x{orig_height} -> {target_width}x{target_height}")
        
        raw_image = np.array(pil_image, dtype=np.float32)
        
        if self.verbose:
            print(f"[编码] 图像: {raw_image.shape[1]}x{raw_image.shape[0]}")
        
        cover_tensor = torch.from_numpy(raw_image / 127.5 - 1.0).permute(2, 1, 0).unsqueeze(0)
        _, _, height, width = cover_tensor.size()
        payload = self._make_payload(width, height, self.data_depth, text)
        
        cover_tensor = cover_tensor.to(self.device)
        payload = payload.to(self.device)
        
        generated = self.encoder(cover_tensor, payload)[0].clamp(-1.0, 1.0)
        generated = (255.0 * (generated + 1.0) / 2.0).long()
        generated = 2.0 * generated.float() / 255.0 - 1.0
        
        output_image = (generated.permute(2, 1, 0).cpu().numpy() + 1.0) * 127.5
        output_pil = Image.fromarray(output_image.astype('uint8'))
        
        if was_scaled:
            output_pil = output_pil.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
            if self.verbose:
                print(f"[编码] 恢复: {target_width}x{target_height} -> {orig_width}x{orig_height}")
        
        output_pil.save(output, 'PNG', optimize=True)
        
        if self.verbose:
            print(f'[编码] 完成: {output}')
        
        if self.cuda:
            clear_cuda_cache()
    
    @torch.inference_mode()
    def decode(self, image):
        if not os.path.exists(image):
            raise ValueError(f'无法读取: {image}')
        
        pil_image = Image.open(image).convert('RGB')
        orig_width, orig_height = pil_image.size
        
        target_width, target_height, was_scaled = self._calculate_target_size(orig_width, orig_height)
        
        if was_scaled:
            pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            if self.verbose:
                print(f"[解码] 缩放: {orig_width}x{orig_height} -> {target_width}x{target_height}")
        
        image_data = np.array(pil_image, dtype=np.float32)
        
        normalized = image_data / 127.5 - 1.0
        image_tensor = torch.from_numpy(normalized).permute(2, 1, 0).unsqueeze(0).to(self.device)
        
        decoded = self.decoder(image_tensor).view(-1) > 0
        bits = decoded.int().cpu().numpy().tolist()
        
        message = self._extract_message(bits)
        if message:
            return message
        
        if image_data.mean() > 230 or image_data.mean() < 25:
            inverted = (255 - image_data) / 127.5 - 1.0
            image_tensor = torch.from_numpy(inverted).permute(2, 1, 0).unsqueeze(0).to(self.device)
            
            decoded = self.decoder(image_tensor).view(-1) > 0
            bits = decoded.int().cpu().numpy().tolist()
            
            message = self._extract_message(bits)
            if message:
                return message
        
        if self.cuda:
            clear_cuda_cache()
        
        raise ValueError(f"解码失败 - 图像可能未包含有效消息")
    
    def _extract_message(self, bits):
        raw_bytes = bits_to_bytearray(bits)
        
        text = bytearray_to_text(raw_bytes)
        if text:
            text = text.lstrip('\x00').strip()
            if text:
                return text
        
        for sep in [b'\x00\x00\x00\x00', b'\x00\x00\x00', b'\x00\x00', b'\x00']:
            for candidate in raw_bytes.split(sep):
                if candidate and len(candidate) > 0:
                    try:
                        text = bytearray_to_text(bytearray(candidate))
                        if text:
                            text = text.lstrip('\x00').strip()
                            if text:
                                return text
                    except:
                        continue
        
        for offset in range(min(8, len(bits) // 8)):
            try:
                partial = bits_to_bytearray(bits[offset * 8:])
                text = bytearray_to_text(partial)
                if text:
                    text = text.lstrip('\x00').strip()
                    if text:
                        return text
            except:
                continue
        
        return None
    
    @torch.no_grad()
    def encode_batch(self, items, output_dir=None):
        self.encoder.eval()
        results = []
        
        for item in items:
            try:
                cover_path, message = item[:2]
                output_path = item[2] if len(item) > 2 else os.path.join(
                    output_dir, f"{os.path.splitext(os.path.basename(cover_path))[0]}_encoded.png"
                )
                
                img = np.array(Image.open(cover_path).convert('RGB'), dtype=np.float32) / 127.5 - 1.0
                cover = torch.from_numpy(img).permute(2, 1, 0).unsqueeze(0).to(self.device)
                
                _, _, h, w = cover.size()
                payload = self._make_payload(w, h, self.data_depth, message).to(self.device)
                
                generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)
                generated = (255.0 * (generated + 1.0) / 2.0).long()
                generated = 2.0 * generated.float() / 255.0 - 1.0
                
                output_arr = (generated.permute(2, 1, 0).cpu().numpy() + 1.0) * 127.5
                Image.fromarray(output_arr.astype('uint8')).save(output_path, 'PNG')
                results.append(output_path)
            except Exception as e:
                if self.verbose:
                    print(f'编码失败: {e}')
                results.append(None)
        
        return results
    
    @torch.no_grad()
    def decode_batch(self, images):
        self.decoder.eval()
        results = []
        
        for path in images:
            try:
                img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 127.5 - 1.0
                tensor = torch.from_numpy(img).permute(2, 1, 0).unsqueeze(0).to(self.device)
                
                decoded = self.decoder(tensor).view(-1) > 0
                bits = decoded.int().cpu().numpy().tolist()
                message = self._extract_message(bits)
                results.append(message)
            except Exception as e:
                if self.verbose:
                    print(f'解码失败: {e}')
                results.append(None)
        
        return results
    
    def save(self, path):
        torch.save(self, path)
    
    @classmethod
    def load(cls, architecture=None, path=None, cuda=True, verbose=False):
        if architecture and not path:
            path = os.path.join(os.path.dirname(__file__), 'pretrained', f'{architecture}.steg')
        elif not architecture and not path:
            raise ValueError('请提供 architecture 或 path')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = torch_load(path, map_location='cpu')
        
        model.critic_optimizer = None
        model.decoder_optimizer = None
        model.verbose = verbose
        
        if not hasattr(model, 'auto_scale'):
            model.auto_scale = True
        if not hasattr(model, 'max_process_size'):
            model.max_process_size = cls.MAX_PROCESS_SIZE if hasattr(cls, 'MAX_PROCESS_SIZE') else 256
        
        model.set_device(cuda)
        model.encoder.eval()
        model.decoder.eval()
        model.critic.eval()
        
        if verbose:
            print(f"[OK] 模型加载成功: {architecture or path}")
        
        return model
