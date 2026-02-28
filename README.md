# ImgStegGAN: GAN-Driven Image Steganography with Qwen Enhancement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

<!-- CHINESE VERSION -->
<a name="中文文档"></a>
# 中文文档

</div>

一个高性能的图像隐写系统，结合生成对抗网络（GAN）与 Qwen 语言模型增强，实现安全高效的隐藏消息嵌入和提取。

## 功能特性

- **基于 GAN 的隐写术**：利用深度学习 GAN 架构实现鲁棒的消息隐藏
- **Qwen 增强**：集成 Qwen 语言模型，实现语义理解和改进编码
- **高容量**：支持嵌入大量消息，同时保持图像质量
- **无损提取**：100% 准确的消息提取，带纠错功能
- **维度保持**：输出图像保持与原始图像完全相同的尺寸
- **Web 界面**：现代化、专业的 Web UI，便于操作
- **批量处理**：支持同时处理多张图像
- **GPU 加速**：支持 CUDA，实现更快的处理速度

## 项目结构

```
ImgStegGAN/
├── steganogan/                 # 核心库
│   ├── __init__.py
│   ├── cli.py                  # 命令行接口
│   ├── models.py               # GAN 模型
│   ├── encoders.py             # 编码器架构
│   ├── decoders.py             # 解码器架构
│   ├── critics.py              # 判别器网络
│   ├── qwen_integration.py     # Qwen 模型集成
│   ├── trainer.py              # 训练工具
│   ├── data_loader.py          # 数据加载工具
│   ├── utils.py                # 辅助函数
│   ├── compat.py               # PyTorch 兼容性
│   └── config.py               # 配置
├── web_interface/              # Web 应用
│   ├── app.py                  # Flask 应用
│   ├── static/
│   │   ├── css/style.css       # 样式
│   │   └── js/app.js           # 前端逻辑
│   ├── templates/
│   │   └── index.html          # 主模板
│   └── uploads/                # 上传目录
├── output_qwen/                # 模型输出
│   └── qwen_steganogan.steg    # 预训练模型
├── train.py                    # 训练脚本
├── setup.py                    # 包设置
├── requirements.txt            # 依赖
└── README.md                   # 文档
```

## 安装

### 前置要求

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本
- CUDA（可选，用于 GPU 加速）

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/ImgStegGAN.git
cd ImgStegGAN

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 依赖项

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
reedsolo>=1.0.0
tqdm>=4.64.0
flask>=2.0.0
flask-cors>=3.0.0
transformers>=4.30.0
```

## Qwen3 模型配置

本项目使用 Qwen3-0.6B 进行增强隐写。请按照以下步骤配置模型。

### 步骤 1：下载 Qwen3-0.6B 模型

从以下官方来源之一下载：

| 来源 | URL |
|--------|-----|
| Hugging Face | https://huggingface.co/Qwen/Qwen3-0.6B |
| ModelScope | https://modelscope.cn/models/Qwen/Qwen3-0.6B |

**推荐版本**：Qwen3-0.6B（约 1.2GB）

### 步骤 2：创建模型目录

```bash
# 在项目根目录创建 Qwen 目录
mkdir -p Qwen/Qwen3-0.6B
```

### 步骤 3：放置模型文件

下载并将以下文件放置到 `Qwen/Qwen3-0.6B/`：

```
Qwen/
└── Qwen3-0.6B/
    ├── config.json
    ├── generation_config.json
    ├── merges.txt
    ├── model.safetensors      # 或 pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```

### 步骤 4：验证配置

运行验证脚本：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

print("Qwen3 model loaded successfully!")
```

### 备选方案：使用 Hugging Face 自动下载

如果您有稳定的网络连接，模型可以在首次运行时自动下载：

```python
# 设置环境变量（可选）
import os
os.environ['QWEN_MODEL_NAME'] = 'Qwen/Qwen3-0.6B'

# 或在代码中
from steganogan.qwen_integration import QwenSteganoGAN
model = QwenSteganoGAN(use_local_model=False)  # 将从 HuggingFace 下载
```

详细配置请参见 [MODEL_SETUP.md](MODEL_SETUP.md)。

## 训练数据集

以下数据集用于训练隐写模型。

| 数据集 | 描述 | 大小 | 来源 |
|---------|-------------|------|--------|
| Flickr2K | 高质量训练图像 | ~2GB | [HuggingFace](https://huggingface.co/datasets/laion/Flickr2K) |
| DIV2K | 多样化高分辨率图像 | ~3.5GB | [Data](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| COCO-2017 | 上下文中的常见物体 | ~18GB | [COCO](https://cocodataset.org/) |

### 训练数据准备

1. 从上述来源下载数据集
2. 将图像放置到 `research/data/[数据集名称]/images/`
3. 运行训练：`python train.py --data_root ./research/data`

## 使用方法

### Web 界面

启动 Web 服务器：

```bash
cd web_interface
python app.py
```

在 `http://localhost:5000` 访问界面

#### 功能：
- **嵌入消息**：上传图像并输入要隐藏的秘密消息
- **提取消息**：上传隐写图像以提取隐藏消息
- **批量处理**：一次处理多张图像

### 命令行

```bash
# 将消息编码到图像中
imgsteggan encode input.png output.png "Your secret message"

# 从图像中解码消息
imgsteggan decode output.png

# 训练新模型
python train.py --data_dir ./data --output_dir ./output
```

### Python API

```python
from steganogan.qwen_integration import QwenSteganoGAN

# 加载模型
model = QwenSteganoGAN.load('output_qwen/qwen_steganogan.steg')

# 编码消息
model.encode('cover_image.png', 'output_image.png', 'Secret message')

# 解码消息
message = model.decode('output_image.png')
print(message)  # 'Secret message'
```

## 模型训练

训练您自己的模型：

```bash
python train.py \
    --data_dir /path/to/training/images \
    --output_dir output_qwen \
    --epochs 5 \
    --batch_size 4 \
    --image_size 128
```

### 训练参数

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `--data_dir` | 必填 | 训练图像路径 |
| `--output_dir` | `output_qwen` | 模型输出目录 |
| `--epochs` | 5 | 训练轮数 |
| `--batch_size` | 4 | 训练批次大小 |
| `--image_size` | 128 | 训练图像尺寸 |
| `--lr` | 2e-4 | 学习率 |
| `--cuda` | 自动 | 如果可用则使用 CUDA |

## 性能指标

| 指标 | 数值 |
|--------|-------|
| 解码成功率 | 100% |
| PSNR | ≥34 dB |
| SSIM | ≥0.99 |
| 维度匹配 | 100% |

## 技术细节

### 架构

- **编码器**：具有基于注意力位置选择的密集编码器
- **解码器**：具有多尺度特征提取的残差解码器
- **判别器**：用于对抗训练的基础判别器
- **消息处理器**：Reed-Solomon 纠错编码

### 支持的格式

- **输入**：PNG、JPG、JPEG、BMP
- **输出**：PNG（推荐用于无损提取）

### 容量

消息容量取决于图像尺寸：

- 容量（字节）≈ 宽 × 高 × 3 / 8
- 最小推荐图像尺寸：256×256 像素

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

本项目基于 MIT Data To AI Lab 的原始 SteganoGAN 项目构建。我们向原作者的基础工作表示感谢。

### 原始项目

**SteganoGAN: High Capacity Image Steganography with GANs**

- Authors: Kevin Alex Zhang, Alfredo Cuesta-Infante, Kalyan Veeramachaneni
- Institution: MIT EECS
- Year: 2019
- Paper: [arXiv:1901.03892](https://arxiv.org/abs/1901.03892)

## 引用

如果您在研究中使用本项目，请同时引用原始 SteganoGAN 论文和本项目：

### 原始 SteganoGAN 论文

```bibtex
@article{zhang2019steganogan,
  title={SteganoGAN: High Capacity Image Steganography with GANs},
  author={Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  journal={arXiv preprint arXiv:1901.03892},
  year={2019},
  url={https://arxiv.org/abs/1901.03892}
}
```

**文本引用：**
> Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan. SteganoGAN: High Capacity Image Steganography with GANs. MIT EECS, January 2019.

---

<!-- ENGLISH VERSION -->
<a name="English-documentation"></a>
# English Documentation

</div>

A high-performance image steganography system that combines Generative Adversarial Networks (GAN) with Qwen language model enhancement for secure and efficient hidden message embedding and extraction.

## Features

- **GAN-Based Steganography**: Utilizes deep learning GAN architecture for robust message hiding
- **Qwen Enhancement**: Integrated Qwen language model for semantic understanding and improved encoding
- **High Capacity**: Supports embedding large messages while maintaining image quality
- **Lossless Extraction**: 100% accurate message extraction with error correction
- **Dimension Preservation**: Output images maintain exact original dimensions
- **Web Interface**: Modern, professional web UI for easy operation
- **Batch Processing**: Support for processing multiple images simultaneously
- **GPU Acceleration**: CUDA support for faster processing

## Project Structure

```
ImgStegGAN/
├── steganogan/                 # Core library
│   ├── __init__.py
│   ├── cli.py                  # Command line interface
│   ├── models.py               # GAN models
│   ├── encoders.py             # Encoder architectures
│   ├── decoders.py             # Decoder architectures
│   ├── critics.py              # Discriminator networks
│   ├── qwen_integration.py     # Qwen model integration
│   ├── trainer.py              # Training utilities
│   ├── data_loader.py          # Data loading utilities
│   ├── utils.py                # Helper functions
│   ├── compat.py               # PyTorch compatibility
│   └── config.py               # Configuration
├── web_interface/              # Web application
│   ├── app.py                  # Flask application
│   ├── static/
│   │   ├── css/style.css       # Styles
│   │   └── js/app.js           # Frontend logic
│   ├── templates/
│   │   └── index.html          # Main template
│   └── uploads/                # Upload directory
├── output_qwen/                # Model outputs
│   └── qwen_steganogan.steg    # Pre-trained model
├── train.py                    # Training script
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ImgStegGAN.git
cd ImgStegGAN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
reedsolo>=1.0.0
tqdm>=4.64.0
flask>=2.0.0
flask-cors>=3.0.0
transformers>=4.30.0
```

## Qwen3 Model Configuration

This project uses Qwen3-0.6B for enhanced steganography.Please follow the steps below to configure the model.

### Step 1: Download Qwen3-0.6B Model

Download from one of the official sources:

| Source | URL |
|--------|-----|
| Hugging Face | https://huggingface.co/Qwen/Qwen3-0.6B |
| ModelScope | https://modelscope.cn/models/Qwen/Qwen3-0.6B |

**Recommended version**: Qwen3-0.6B (approximately 1.2GB)

### Step 2: Create Model Directory

```bash
# Create the Qwen directory in project root
mkdir -p Qwen/Qwen3-0.6B
```

### Step 3: Place Model Files

Download and place the following files in `Qwen/Qwen3-0.6B/`:

```
Qwen/
└── Qwen3-0.6B/
    ├── config.json
    ├── generation_config.json
    ├── merges.txt
    ├── model.safetensors      # or pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```

### Step 4: Verify Configuration

Run the verification script:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

print("Qwen3 model loaded successfully!")
```

### Alternative: Use Hugging Face Auto-download

If you have a stable internet connection, the model can be auto-downloaded on first run:

```python
# Set environment variable (optional)
import os
os.environ['QWEN_MODEL_NAME'] = 'Qwen/Qwen3-0.6B'

# Or in code
from steganogan.qwen_integration import QwenSteganoGAN
model = QwenSteganoGAN(use_local_model=False)  # Will download from HuggingFace
```

For detailed configuration, see [MODEL_SETUP.md](MODEL_SETUP.md).

## Training Datasets

The following datasets were used for training the steganography model.

| Dataset | Description | Size | Source |
|---------|-------------|------|--------|
| Flickr2K | High-quality images for training | ~2GB | [HuggingFace](https://huggingface.co/datasets/laion/Flickr2K) |
| DIV2K | Diverse high-resolution images | ~3.5GB | [Data](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| COCO-2017 | Common objects in context | ~18GB | [COCO](https://cocodataset.org/) |

### Training Data Preparation

1. Download the datasets from the sources above
2. Place images in `research/data/[dataset_name]/images/`
3. Run training: `python train.py --data_root ./research/data`

## Usage

### Web Interface

Start the web server:

```bash
cd web_interface
python app.py
```

Access the interface at `http://localhost:5000`

#### Features:
- **Embed Message**: Upload an image and enter the secret message to hide
- **Extract Message**: Upload a steganographic image to extract hidden message
- **Batch Processing**: Process multiple images at once

### Command Line

```bash
# Encode a message into an image
imgsteggan encode input.png output.png "Your secret message"

# Decode a message from an image
imgsteggan decode output.png

# Train a new model
python train.py --data_dir ./data --output_dir ./output
```

### Python API

```python
from steganogan.qwen_integration import QwenSteganoGAN

# Load the model
model = QwenSteganoGAN.load('output_qwen/qwen_steganogan.steg')

# Encode a message
model.encode('cover_image.png', 'output_image.png', 'Secret message')

# Decode a message
message = model.decode('output_image.png')
print(message)  # 'Secret message'
```

## Model Training

Train your own model:

```bash
python train.py \
    --data_dir /path/to/training/images \
    --output_dir output_qwen \
    --epochs 5 \
    --batch_size 4 \
    --image_size 128
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Path to training images |
| `--output_dir` | `output_qwen` | Output directory for model |
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 4 | Batch size for training |
| `--image_size` | 128 | Image size for training |
| `--lr` | 2e-4 | Learning rate |
| `--cuda` | Auto | Use CUDA if available |

## Performance

| Metric | Value |
|--------|-------|
| Decode Success Rate | 100% |
| PSNR | ≥34 dB |
| SSIM | ≥0.99 |
| Dimension Match | 100% |

## Technical Details

### Architecture

- **Encoder**: Dense encoder with attention-based position selection
- **Decoder**: Residual decoder with multi-scale feature extraction
- **Critic**: Basic critic for adversarial training
- **Message Processor**: Reed-Solomon error correction encoding

### Supported Formats

- **Input**: PNG, JPG, JPEG, BMP
- **Output**: PNG (recommended for lossless extraction)

### Capacity

The message capacity depends on image dimensions:
- Capacity (bytes) ≈ Width × Height × 3 / 8
- Minimum recommended image size: 256×256 pixels

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is built upon the original SteganoGAN project by MIT Data To AI Lab. We extend our gratitude to the original authors for their foundational work.

### Original Project

**SteganoGAN: High Capacity Image Steganography with GANs**

- Authors: Kevin Alex Zhang, Alfredo Cuesta-Infante, Kalyan Veeramachaneni
- Institution: MIT EECS
- Year: 2019
- Paper: [arXiv:1901.03892](https://arxiv.org/abs/1901.03892)

## Citation

If you use this project in your research, please cite both the original SteganoGAN paper and this project:

### Original SteganoGAN Paper

```bibtex
@article{zhang2019steganogan,
  title={SteganoGAN: High Capacity Image Steganography with GANs},
  author={Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  journal={arXiv preprint arXiv:1901.03892},
  year={2019},
  url={https://arxiv.org/abs/1901.03892}
}
```

**Text Citation:**
> Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan. SteganoGAN: High Capacity Image Steganography with GANs. MIT EECS, January 2019.
