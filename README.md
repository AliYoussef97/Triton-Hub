# TritonHub

## About
TritonHub serves as a container for various PyTorch neural network components written in Triton, offering an easy way to access and integrate Triton-based functions from one centralized source. These components include custom implementations of activation functions, normalization layers, and other neural network operations designed for efficient execution on GPUs.

I hope this repository proves helpful to the community, making Triton-based neural network operations more accessible and convenient for developers and researchers alike.

## Installation

You can install **TritonHub** directly from:

```bash
pip install TritonHub
```

Alternatively, you can install it from the GitHub repository:

```bash
pip install git+https://github.com/AliYoussef97/TritonHub.git
```

Or clone the repository and install manually:

```bash
git clone https://github.com/AliYoussef97/TritonHub.git
cd TritonHub
python setup.py install
```

TritonHub requires the following dependencies:
-   Linux
-   CUDA
-   NVIDIA GPU

## Usage Example

```python
import torch
from TritonHub.Normalization import LayerNorm
from TritonHub.Activation import GeLU

batch, length, dim = 2, 100, 128
device = "cuda"
dtype = torch.float32  # or torch.float16

x = torch.randn(batch, length, dim, device=device, dtype=dtype).to("cuda")

# Initialize LayerNorm and GeLU modules from TritonHub
layernorm = LayerNorm(128, eps=1e-6, elementwise_affine=True, bias=True, device=device, dtype=dtype)
gelu = GeLU(approximate='None') # or tanh

# Apply LayerNorm and GeLU on the input tensor
x = layernorm(x)
x = gelu(x)
```

## Contributions

Contributions are welcome! If you'd like to add new functionalities, please:
1. Create a pull request with your implementation.
2. Ensure the pull request includes a unit test for the new module, similar to the tests found in the `UnitTests` folder.

If you encounter any bugs or issues with the current version, please feel free to report them by creating an issue or submitting a pull request.

## TODO
| Feature                         | Status  |
|----------------------------------|---------|
| Linear Layer Backward Pass       | ✔️      |
| Include Triton Block Sizes in Autotune | ❌  |
| Convolution Layer                      | ❌       |
| BatchNorm                        | ❌       |
| Different Activation Functions   | ✔️       |


## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/AliYoussef97/TritonHub/blob/main/LICENSE) file for more details.

## Acknowledgments
Special thanks to the authors of [Mamba](https://github.com/state-spaces/mamba), as parts of their Triton code was influential or some of the modules in TritonHub.
