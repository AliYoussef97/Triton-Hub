<h1 style="text-align: center;">TritonHub</h1>

## 🌐 Overview
TritonHub serves as a container for various PyTorch neural network components written in Triton, offering an easy way to access and integrate Triton-based functions from one centralized source. These components include custom implementations of activation functions, normalization layers, and other neural network operations designed for efficient execution on GPUs.

## 📦 Installation

Clone the repository and install using `setup.py`:

```bash
git clone https://github.com/AliYoussef97/TritonHub.git
cd TritonHub
python setup.py install
```

For development:
```bash
python setup.py develop
```

### ✓ Prerequisites
TritonHub requires the following distinguished companions:
- Linux operating system
- CUDA
- GPU hardware
- Triton (installed via pip or from source)

## 🚀 Quick Start

```python
import torch
from TritonHub.Normalization import LayerNorm
from TritonHub.Activation import GeLU

batch, length, dim = 2, 100, 128
device = "cuda"
dtype = torch.float32 # or torch.float16

x = torch.randn(batch, length, dim, device=device, dtype=dtype).to("cuda")

layernorm = LayerNorm(128, eps=1e-6, elementwise_affine=True, bias=True, device=device, dtype=dtype)
gelu = GeLU(approximate='None') # or tanh approximation.

x = layernorm(x)
x = gelu(x)
```

## 🤝 Contributions

Contributions are welcomed! To add a new feature or improve an existing module:

1. Fork the repository and create a pull request.
2. Include a unit test under the UnitTests directory for your module.
3. Follow existing coding conventions and ensure compatibility with PyTorch + Triton.

Found a bug or have a suggestion? Feel free to [open an issue](https://github.com/ayoussf/Triton-Hub/issues) or submit a PR.



## 🗺️ Roadmap
| Exquisite Feature                | Status       |
|----------------------------------|--------------|
| Linear Layer Backward Pass       | ✅ |
| Include Triton Block Sizes in Autotune | ⏳ |
| Convolution Layer                | ❌ |
| BatchNorm                        | ❌ |
| Different Activation Functions   | ✅ |
| Distance Functions               | ⏳ |


## 📄 License
TritonHub is released under the MIT License. You're free to use, modify, and distribute it.

## 🙏 Acknowledgments
Special thanks to the authors of [Mamba](https://github.com/state-spaces/mamba). Their work has been a valuable reference for parts of this repository, particularly around Triton code patterns and performance optimization.