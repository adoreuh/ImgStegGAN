# Qwen3 Model Setup Guide

This guide provides detailed instructions for downloading, configuring, and verifying the Qwen3-0.6B model for use with ImgStegGAN.

## Table of Contents

1. [Model Requirements](#model-requirements)
2. [Download Methods](#download-methods)
3. [Directory Structure](#directory-structure)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

## Model Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 4GB | 8GB+ |
| Disk Space | 2GB | 4GB+ |
| Python | 3.8+ | 3.10+ |
| PyTorch | 2.0+ | 2.1+ |

### Model Information

| Property | Value |
|----------|-------|
| Model Name | Qwen3-0.6B |
| Parameters | ~600M |
| Model Size | ~1.2GB |
| License | Apache 2.0 |

## Download Methods

### Method 1: Hugging Face (Recommended)

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download model
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./Qwen/Qwen3-0.6B",
    local_dir_use_symlinks=False
)
```

### Method 2: Git LFS

```bash
# Install Git LFS
git lfs install

# Clone the model repository
git clone https://huggingface.co/Qwen/Qwen3-0.6B ./Qwen/Qwen3-0.6B
```

### Method 3: Manual Download

1. Visit https://huggingface.co/Qwen/Qwen3-0.6B/tree/main
2. Download each file individually
3. Place files in `Qwen/Qwen3-0.6B/` directory

### Method 4: ModelScope (China Users)

```bash
# Install modelscope
pip install modelscope

# Download model
from modelscope import snapshot_download

snapshot_download(
    'Qwen/Qwen3-0.6B',
    cache_dir='./Qwen'
)
```

## Directory Structure

After downloading, your directory should look like this:

```
ImgStegGAN/
├── Qwen/
│   └── Qwen3-0.6B/
│       ├── config.json              # Model configuration
│       ├── generation_config.json   # Generation parameters
│       ├── merges.txt               # BPE merges file
│       ├── model.safetensors        # Model weights (or pytorch_model.bin)
│       ├── tokenizer.json           # Tokenizer configuration
│       ├── tokenizer_config.json    # Tokenizer settings
│       └── vocab.json               # Vocabulary file
├── steganogan/
├── web_interface/
└── ...
```

### Required Files Checklist

- [ ] `config.json` - Essential
- [ ] `generation_config.json` - Essential
- [ ] `merges.txt` - Essential for tokenizer
- [ ] `model.safetensors` or `pytorch_model.bin` - Essential (model weights)
- [ ] `tokenizer.json` - Essential
- [ ] `tokenizer_config.json` - Essential
- [ ] `vocab.json` - Essential for tokenizer

## Configuration

### Environment Variables (Optional)

```bash
# Linux/Mac
export QWEN_MODEL_PATH="./Qwen/Qwen3-0.6B"
export QWEN_MODEL_NAME="Qwen/Qwen3-0.6B"

# Windows PowerShell
$env:QWEN_MODEL_PATH = ".\Qwen\Qwen3-0.6B"
$env:QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"

# Windows CMD
set QWEN_MODEL_PATH=.\Qwen\Qwen3-0.6B
set QWEN_MODEL_NAME=Qwen/Qwen3-0.6B
```

### Configuration File

Create `config/qwen_config.json`:

```json
{
    "model_path": "./Qwen/Qwen3-0.6B",
    "model_name": "Qwen/Qwen3-0.6B",
    "use_local": true,
    "device": "auto",
    "torch_dtype": "float16",
    "trust_remote_code": true
}
```

### Loading in Python

```python
from steganogan.qwen_integration import QwenSteganoGAN

# Method 1: Load with local model
model = QwenSteganoGAN(
    qwen_model_path="./Qwen/Qwen3-0.6B",
    use_local_model=True
)

# Method 2: Auto-download from HuggingFace
model = QwenSteganoGAN(
    qwen_model_name="Qwen/Qwen3-0.6B",
    use_local_model=False
)

# Method 3: Load pre-trained steganography model
model = QwenSteganoGAN.load("output_qwen/qwen_steganogan.steg")
```

## Verification

### Quick Test

```python
# test_qwen_setup.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_qwen_setup():
    model_path = "./Qwen/Qwen3-0.6B"
    
    print("Testing Qwen3 model setup...")
    
    # Test tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Tokenizer failed: {e}")
        return False
    
    # Test model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model failed: {e}")
        return False
    
    # Test inference
    try:
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=10)
        print("✓ Inference successful")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    print("\nAll tests passed! Qwen3 is properly configured.")
    return True

if __name__ == "__main__":
    test_qwen_setup()
```

### Run Verification

```bash
python test_qwen_setup.py
```

Expected output:
```
Testing Qwen3 model setup...
✓ Tokenizer loaded successfully
✓ Model loaded successfully
✓ Inference successful

All tests passed! Qwen3 is properly configured.
```

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `OSError: Can't load tokenizer for './Qwen/Qwen3-0.6B'`

**Solution**:
- Verify the model directory exists
- Check all required files are present
- Ensure file permissions are correct

#### 2. Out of Memory

**Error**: `torch.cuda.OutOfMemoryError`

**Solution**:
```python
# Use CPU instead
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype=torch.float32
)

# Or use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True
)
```

#### 3. Slow Download

**Solution**:
- Use ModelScope for users in China
- Use a mirror site
- Download during off-peak hours

#### 4. Trust Remote Code Warning

**Error**: `The repository contains custom code`

**Solution**:
```python
# Add trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True
)
```

### Getting Help

If you encounter issues:

1. Check the [Qwen GitHub](https://github.com/QwenLM/Qwen) for updates
2. Visit [Hugging Face Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) for documentation
3. Open an issue in this repository

## Additional Resources

- [Qwen Official Documentation](https://github.com/QwenLM/Qwen)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [ModelScope Documentation](https://modelscope.cn/docs)
