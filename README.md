<p align="center">
    <img src="./assets/tritonhub.png" alt="TritonHub" style="object-fit: cover;"/>
</p>
<!-- <h1 align="center">TritonHub</h1> -->

## 🌐 Overview
TritonHub is a *differentiable*, efficient, and modular open-source library of PyTorch neural network modules and operations implemented in [Triton](https://github.com/triton-lang/triton). It provides GPU-accelerated primitives that leverage Triton's low-level control and parallelism, enabling seamless integration of deep learning building blocks into workflows. The framework supports both forward and backward passes while maintaining full compatibility with PyTorch, and can be easily extended and adapted to support the needs of the deep learning research and development community.

## 📦 Installation

Clone the repository and install using `setup.py`:

```bash
git clone https://github.com/ayoussf/TritonHub.git
cd TritonHub
python setup.py install
```

For development:
```bash
python setup.py develop
```

### ⚙️ Prerequisites
TritonHub requires the following dependencies:
- Linux operating system (WSL for Windows users)
- CUDA
- GPU hardware
- Triton (installed via pip or from [source](https://github.com/triton-lang/triton))

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

## 🧩 Supported Modules

TritonHub currently supports the following modules, with <span style="color:green"><strong>forward</strong></span> and <span style="color:red"><strong>backward</strong></span> passes:

- **Activation Functions**
  - GeLU (with/without tanh approximation)
  - ReLU
  - LeakyReLU
  - ReLU6
  - Sigmoid
  - Tanh
  - Mish
  - SiLU (Swish)
  - Softmax
  - LogSoftmax
  - Softmin
  - Softplus
  - Threshold

- **Normalization Layers**
  - LayerNorm
  - RMSNorm
  - **Planned: BatchNorm**

- **Neural Network Layers**
  - Linear
  - Dropout
  - **Planned: Convolution Layers (1D/2D)**

- **Distance Functions**
  - Pairwise cosine similarity

- **Ops**
  - Batched Matmul (bmm): supports unbatched inputs
  - Normalize (L2 normalization, **Planned: L1 normalization**)
  - Norm (matrix/vector L1, L2 and p-norms)


## 🗺️ Roadmap
| Exquisite Feature                | Status       |
|----------------------------------|--------------|
| Linear Layer Backward Pass       | ✅ |
| Include Triton Block Sizes in Autotune | ✅ |
| Convolution Layer (1D/2D)               | ❌ |
| BatchNorm                        | ❌ |
| L1 Tensor Normalization               | ❌ |
| Matrix/Vector L1, L2 and p Norms         | ✅ |
| Activation Functions   | ✅ |
| Distance Functions               | ✅ |
| Batched Matmul               | ✅ |

## 🤝 Contributions

Contributions are welcomed! To add a new feature or improve an existing module:

1. Fork the repository and create a pull request.
2. Include a unit test under the UnitTests directory for your module.
3. Follow existing coding conventions and ensure compatibility with PyTorch + Triton.

Found a bug or have a suggestion? Feel free to [open an issue](https://github.com/ayoussf/Triton-Hub/issues) or submit a PR.

## 📄 License
TritonHub is released under the MIT License. You're free to use, modify, and distribute it.

## 🙏 Acknowledgments
Special thanks to the authors of [Mamba](https://github.com/state-spaces/mamba). Their work has been a valuable reference for parts of this repository.