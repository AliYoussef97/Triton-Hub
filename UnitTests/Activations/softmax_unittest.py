import torch
import torch.nn as nn
from TritonHub.Activations import Softmax
from tabulate import tabulate as tb

class SoftmaxUnitTest:
    def __init__(self, B=4, N=512, M=512, D=256, dtype=torch.float32, print_tb=False):
        self.B = B
        self.N = N
        self.M = M
        self.D = D
        self.dtype = dtype
        self.print_tb = print_tb

        # Triton softmax and Torch softmax
        self.softmax = Softmax()
        self.softmax_torch = nn.Softmax(dim=-1)

    def run(self):
        torch.manual_seed(42)
        # Create the input tensor. (This is an example of an "image" tensor with B H W C layout, however it can be any tensors since
        # interally tensors get flattened to 2D tensors (-1, C) before softmax computation)
        input_data = torch.randn(self.B, self.M, self.N, self.D, device='cuda', dtype=self.dtype)

        # Create separate tensors for input and input_ref using the same data and ensure gradient computation
        input = input_data.clone().detach().requires_grad_(True)
        input_ref = input_data.clone().detach().requires_grad_(True)

        # Set the tolerance for the comparison
        rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (5e-3, 1e-2)
        if dtype == torch.float16:
            rtol, atol = 1e-2, 5e-2
        self.forward(input, input_ref, atol, rtol)

    def forward(self, input, input_ref, atol, rtol):
        output = self.softmax(input)
        output_ref = self.softmax_torch(input_ref)
        assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), 'Error in forward pass'
        if self.print_tb:
            self.diff_f = (output - output_ref).abs()
        self.backward(input, input_ref, output, output_ref, atol, rtol)

    def backward(self, input, input_ref, output, output_ref, atol, rtol):
        g = torch.randn_like(output)
        output.backward(g)
        output_ref.backward(g)
        assert torch.allclose(input.grad, input_ref.grad, atol=atol, rtol=rtol), 'Error in backward pass'
        if self.print_tb:
            self.diff_b = (input.grad - input_ref.grad).abs()
            self.table()
    
    def table(self):
        print(tb([[self.dtype, self.D, self.diff_f.mean().item(), self.diff_f.max().item(), self.diff_b.mean().item(), self.diff_b.max().item()]],
                headers=['Dype', 'Dim', 'Forward Mean Diff', 'Forward Max Diff', 'Backward Mean Diff', 'Backward Max Diff'], tablefmt='orgtbl'))

if __name__ == '__main__':
    B, N, M = 1, 256, 256
    print_tb = True
    for i in range(2):
        if i ==0: print('First iteration Slow due to Triton Autotune')
        for D in [32, 64, 128, 256, 512, 1024, 2048]:
            for dtype in [torch.float16, torch.float32, torch.float64]:
                runner = SoftmaxUnitTest(B, N, M, D, dtype, print_tb)
                runner.run()
    print('All tests passed!')