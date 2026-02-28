# ImgStegGAN: GAN-Driven Image Steganography with Qwen Enhancement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Version: V1.0.0 | 版本：V1.0.0**

A high-performance image steganography system that combines Generative Adversarial Networks (GAN) with Qwen language model enhancement for secure and efficient hidden message embedding and extraction.

一个高性能的图像隐写系统，结合生成对抗网络（GAN）与 Qwen 语言模型增强，实现安全高效的隐藏消息嵌入和提取。

## Features | 功能特性

- **GAN-Based Steganography**: Utilizes deep learning GAN architecture for robust message hiding
- **基于 GAN 的隐写术**：利用深度学习 GAN 架构实现鲁棒的消息隐藏

- **Qwen Enhancement**: Integrated Qwen language model for semantic understanding and improved encoding
- **Qwen 增强**：集成 Qwen 语言模型，实现语义理解和改进编码

- **High Capacity**: Supports embedding large messages while maintaining image quality
- **高容量**：支持嵌入大量消息，同时保持图像质量

- **Lossless Extraction**: 100% accurate message extraction with error correction
- **无损提取**：100% 准确的消息提取，带纠错功能

- **Dimension Preservation**: Output images maintain exact original dimensions
- **维度保持**：输出图像保持与原始图像完全相同的尺寸

- **Web Interface**: Modern, professional web UI for easy operation
- **Web 界面**：现代化、专业的 Web UI，便于操作

- **Batch Processing**: Support for processing multiple images simultaneously
- **批量处理**：支持同时处理多张图像

- **GPU Acceleration**: CUDA support for faster processing
- **GPU 加速**：支持 CUDA，实现更快的处理速度

## Project Structure | 项目结构

```
ImgStegGAN/
├── steganogan/                 # Core library | 核心库
│   ├── __init__.py
│   ├── cli.py                  # Command line interface | 命令行接口
│   ├── models.py               # GAN models | GAN 模型
│   ├── encoders.py             # Encoder architectures | 编码器架构
│   ├── decoders.py             # Decoder architectures | 解码器架构
│   ├── critics.py              # Discriminator networks | 判别器网络
│   ├── qwen_integration.py     # Qwen model integration | Qwen 模型集成
│   ├── trainer.py              # Training utilities | 训练工具
│   ├── data_loader.py          # Data loading utilities | 数据加载工具
│   ├── utils.py                # Helper functions | 辅助函数
│   ├── compat.py               # PyTorch compatibility | PyTorch 兼容性
│   └── config.py               # Configuration | 配置
├── web_interface/              # Web application | Web 应用
│   ├── app.py                  # Flask application | Flask 应用
│   ├── static/
│   │   ├── css/style.css       # Styles | 样式
│   │   └── js/app.js           # Frontend logic | 前端逻辑
│   ├── templates/
│   │   └── index.html          # Main template | 主模板
│   └── uploads/                # Upload directory | 上传目录
├── output_qwen/                # Model outputs | 模型输出
│   └── qwen_steganogan.steg    # Pre-trained model | 预训练模型
├── train.py                    # Training script | 训练脚本
├── setup.py                    # Package setup | 包设置
├── requirements.txt            # Dependencies | 依赖
└── README.md                   # Documentation | 文档
```

## Installation | 安装

### Prerequisites | 前置要求

- Python 3.8 or higher | Python 3.8 或更高版本
- PyTorch 2.0 or higher | PyTorch 2.0 或更高版本
- CUDA (optional, for GPU acceleration) | CUDA（可选，用于 GPU 加速）

### Install from Source | 从源码安装

```bash
# Clone the repository | 克隆仓库
git clone https://github.com/yourusername/ImgStegGAN.git
cd ImgStegGAN

# Create virtual environment | 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or | 或
.\venv\Scripts\activate  # Windows

# Install dependencies | 安装依赖
pip install -r requirements.txt

# Install the package | 安装包
pip install -e .
```

### Dependencies | 依赖项

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

## Qwen3 Model Configuration | Qwen3 模型配置

This project uses Qwen3-0.6B for enhanced steganography. **The model files are NOT included in this repository** due to their large size. Please follow the steps below to configure the model.

本项目使用 Qwen3-0.6B 进行增强隐写。请按照以下步骤配置模型。

### Step 1: Download Qwen3-0.6B Model | 步骤 1：下载 Qwen3-0.6B 模型

Download from one of the official sources:

从以下官方来源之一下载：

| Source | URL |
|--------|-----|
| Hugging Face | https://huggingface.co/Qwen/Qwen3-0.6B |
| ModelScope | https://modelscope.cn/models/Qwen/Qwen3-0.6B |

**Recommended version**: Qwen3-0.6B (approximately 1.2GB)

**推荐版本**：Qwen3-0.6B（约 1.2GB）

### Step 2: Create Model Directory | 步骤 2：创建模型目录

```bash
# Create the Qwen directory in project root | 在项目根目录创建 Qwen 目录
mkdir -p Qwen/Qwen3-0.6B
```

### Step 3: Place Model Files | 步骤 3：放置模型文件

Download and place the following files in `Qwen/Qwen3-0.6B/`:

下载并将以下文件放置到 `Qwen/Qwen3-0.6B/`：

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

### Step 4: Verify Configuration | 步骤 4：验证配置

Run the verification script:

运行验证脚本：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

print("Qwen3 model loaded successfully!")
```

### Alternative: Use Hugging Face Auto-download | 备选方案：使用 Hugging Face 自动下载

If you have a stable internet connection, the model can be auto-downloaded on first run:

如果您有稳定的网络连接，模型可以在首次运行时自动下载：

```python
# Set environment variable (optional) | 设置环境变量（可选）
import os
os.environ['QWEN_MODEL_NAME'] = 'Qwen/Qwen3-0.6B'

# Or in code | 或在代码中
from steganogan.qwen_integration import QwenSteganoGAN
model = QwenSteganoGAN(use_local_model=False)  # Will download from HuggingFace
```

For detailed configuration, see [MODEL_SETUP.md](MODEL_SETUP.md).

详细配置请参见 [MODEL_SETUP.md](MODEL_SETUP.md)。

## Training Datasets | 训练数据集

The following datasets were used for training the steganography model. **Dataset files are NOT included in this repository**.

以下数据集用于训练隐写模型。

| Dataset | Description | Size | Source |
|---------|-------------|------|--------|
| Flickr2K | High-quality images for training | ~2GB | [HuggingFace](https://huggingface.co/datasets/laion/Flickr2K) |
| DIV2K | Diverse high-resolution images | ~3.5GB | [Data](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| COCO-2017 | Common objects in context | ~18GB | [COCO](https://cocodataset.org/) |

| 数据集 | 描述 | 大小 | 来源 |
|---------|-------------|------|--------|
| Flickr2K | 高质量训练图像 | ~2GB | [HuggingFace](https://huggingface.co/datasets/laion/Flickr2K) |
| DIV2K | 多样化高分辨率图像 | ~3.5GB | [Data](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| COCO-2017 | 上下文中的常见物体 | ~18GB | [COCO](https://cocodataset.org/) |

### Training Data Preparation | 训练数据准备

1. Download the datasets from the sources above
2. Place images in `research/data/[dataset_name]/images/`
3. Run training: `python train.py --data_root ./research/data`

1. 从上述来源下载数据集
2. 将图像放置到 `research/data/[数据集名称]/images/`
3. 运行训练：`python train.py --data_root ./research/data`

## Usage | 使用方法

### Web Interface | Web 界面

Start the web server:

启动 Web 服务器：

```bash
cd web_interface
python app.py
```

Access the interface at `http://localhost:5000`

在 `http://localhost:5000` 访问界面

#### Features: | 功能：
- **Embed Message**: Upload an image and enter the secret message to hide
- **嵌入消息**：上传图像并输入要隐藏的秘密消息

- **Extract Message**: Upload a steganographic image to extract hidden message
- **提取消息**：上传隐写图像以提取隐藏消息

- **Batch Processing**: Process multiple images at once
- **批量处理**：一次处理多张图像

### Command Line | 命令行

```bash
# Encode a message into an image | 将消息编码到图像中
imgsteggan encode input.png output.png "Your secret message"

# Decode a message from an image | 从图像中解码消息
imgsteggan decode output.png

# Train a new model | 训练新模型
python train.py --data_dir ./data --output_dir ./output
```

### Python API

```python
from steganogan.qwen_integration import QwenSteganoGAN

# Load the model | 加载模型
model = QwenSteganoGAN.load('output_qwen/qwen_steganogan.steg')

# Encode a message | 编码消息
model.encode('cover_image.png', 'output_image.png', 'Secret message')

# Decode a message | 解码消息
message = model.decode('output_image.png')
print(message)  # 'Secret message'
```

## Model Training | 模型训练

Train your own model:

训练您自己的模型：

```bash
python train.py \
    --data_dir /path/to/training/images \
    --output_dir output_qwen \
    --epochs 5 \
    --batch_size 4 \
    --image_size 128
```

### Training Parameters | 训练参数

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Path to training images |
| `--data_dir` | 必填 | 训练图像路径 |
| `--output_dir` | `output_qwen` | Output directory for model |
| `--output_dir` | `output_qwen` | 模型输出目录 |
| `--epochs` | 5 | Number of training epochs |
| `--epochs` | 5 | 训练轮数 |
| `--batch_size` | 4 | Batch size for training |
| `--batch_size` | 4 | 训练批次大小 |
| `--image_size` | 128 | Image size for training |
| `--image_size` | 128 | 训练图像尺寸 |
| `--lr` | 2e-4 | Learning rate |
| `--lr` | 2e-4 | 学习率 |
| `--cuda` | Auto | Use CUDA if available |
| `--cuda` | 自动 | 如果可用则使用 CUDA |

## Performance | 性能指标

| Metric | Value |
|--------|-------|
| Decode Success Rate | 100% |
| 解码成功率 | 100% |
| PSNR | ≥34 dB |
| SSIM | ≥0.99 |
| Decode Time | ≤10s (typical image) |
| 解码时间 | ≤10 秒（典型图像） |
| Dimension Match | 100% |
| 维度匹配 | 100% |

## Technical Details | 技术细节

### Architecture | 架构

- **Encoder**: Dense encoder with attention-based position selection
- **编码器**：具有基于注意力位置选择的密集编码器

- **Decoder**: Residual decoder with multi-scale feature extraction
- **解码器**：具有多尺度特征提取的残差解码器

- **Critic**: Basic critic for adversarial training
- **判别器**：用于对抗训练的基础判别器

- **Message Processor**: Reed-Solomon error correction encoding
- **消息处理器**：Reed-Solomon 纠错编码

### Supported Formats | 支持的格式

- **Input**: PNG, JPG, JPEG, BMP
- **输入**：PNG、JPG、JPEG、BMP

- **Output**: PNG (recommended for lossless extraction)
- **输出**：PNG（推荐用于无损提取）

### Capacity | 容量

The message capacity depends on image dimensions:

消息容量取决于图像尺寸：

- Capacity (bytes) ≈ Width × Height × 3 / 8
- 容量（字节）≈ 宽 × 高 × 3 / 8

- Minimum recommended image size: 256×256 pixels
- 最小推荐图像尺寸：256×256 像素

## License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## Acknowledgments | 致谢

This project is built upon the original SteganoGAN project by MIT Data To AI Lab. We extend our gratitude to the original authors for their foundational work.

本项目基于 MIT Data To AI Lab 的原始 SteganoGAN 项目构建。我们向原作者的基础工作表示感谢感谢。

### Original Project | 原始项目

**SteganoGAN: High Capacity Image Steganography with GANs**

- Authors: Kevin Alex Zhang, Alfredo Cuesta-Infante, Kalyan Veeramachaneni
- Institution: MIT EECS
- Year: 2019
- Paper: [arXiv:1901.03892](https://arxiv.org/abs/1901.03892)

### Other Acknowledgments | 其他致谢

- Qwen language model by Alibaba Cloud
- Qwen 语言模型由阿里云提供

- PyTorch framework
- PyTorch 框架

## Citation | 引用

If you use this project in your research, please cite both the original SteganoGAN paper and this project:

如果您在研究中使用本项目，请同时引用原始 SteganoGAN 论文和本项目：

### Original SteganoGAN Paper | 原始 SteganoGAN 论文

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

**文本引用：**
> Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan. SteganoGAN: High Capacity Image Steganography with GANs. MIT EECS, January 2019.

### This Project (ImgStegGAN) | 本项目（ImgStegGAN）

```bibtex
@software{imgsteggan2026,
  title = {ImgStegGAN: GAN-Driven Image Steganography with Qwen Enhancement},
  author = {ImgStegGAN Team},
  year = {2026},
  version = {1.0.0},
  note = {Based on SteganoGAN by Zhang et al. (2019)}
}
