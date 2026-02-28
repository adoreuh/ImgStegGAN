# -*- coding: utf-8 -*-
"""
SteganoGAN 数据加载模块 V3.0
支持 DIV2K 和 COCO 数据集
"""

import os
import warnings
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
from typing import Optional, List, Tuple

warnings.filterwarnings('ignore', category=UserWarning, module='PIL')


class ImageDataset(Dataset):
    """通用图像数据集"""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(self, root_dir: str, image_size: int = 128, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform or self._get_transform(image_size)
        self.image_paths = self._collect_paths()
        print(f"  有效图像: {len(self.image_paths)} 张")
    
    def _collect_paths(self) -> List[str]:
        valid_paths = []
        
        if os.path.isfile(self.root_dir):
            valid_paths.append(self.root_dir)
        elif os.path.isdir(self.root_dir):
            for root, _, files in os.walk(self.root_dir):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in self.VALID_EXTENSIONS:
                        valid_paths.append(os.path.join(root, f))
        
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
                return self.transform(img), 0
        except Exception:
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank, 0


class SteganoGANDataLoader:
    """数据加载器工厂"""
    
    DEFAULT_CONFIG = {
        'div2k_train': 'DIV2K-main/DIV2K_train_HR',
        'coco_val': 'val2017'
    }
    
    def __init__(self, data_root: str, image_size: int = 128, batch_size: int = 4, num_workers: int = 4):
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _create_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    
    def load_div2k(self) -> Optional[Dataset]:
        path = os.path.join(self.data_root, self.DEFAULT_CONFIG['div2k_train'])
        if os.path.exists(path):
            print(f"[DIV2K] 扫描: {path}")
            return ImageDataset(path, self.image_size)
        return None
    
    def load_coco(self) -> Optional[Dataset]:
        path = os.path.join(self.data_root, self.DEFAULT_CONFIG['coco_val'])
        if os.path.exists(path):
            print(f"[COCO] 扫描: {path}")
            return ImageDataset(path, self.image_size)
        return None
    
    def get_loaders(self, val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader]:
        all_datasets = []
        
        for loader_func in [self.load_div2k, self.load_coco]:
            ds = loader_func()
            if ds and len(ds) > 0:
                all_datasets.append(ds)
        
        if not all_datasets:
            raise ValueError(f"未找到有效数据集: {self.data_root}")
        
        combined = ConcatDataset(all_datasets)
        total = len(combined)
        
        val_size = int(total * val_ratio)
        train_size = total - val_size
        
        train_ds, val_ds = random_split(
            combined, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"\n[数据统计]")
        print(f"  总计: {total} 张")
        print(f"  训练: {train_size} 张")
        print(f"  验证: {val_size} 张")
        
        return (
            self._create_dataloader(train_ds, shuffle=True),
            self._create_dataloader(val_ds, shuffle=False)
        )


def create_dataloader(
    data_root: str,
    image_size: int = 128,
    batch_size: int = 4,
    num_workers: int = 4,
    val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """便捷函数：创建数据加载器"""
    loader = SteganoGANDataLoader(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return loader.get_loaders(val_ratio)
