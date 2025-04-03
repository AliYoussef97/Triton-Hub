import torch
from TritonHub.Ops import bmm
from tabulate import tabulate as tb

class BMMUnitTest:
    def __init__(self, B=4, N=512, M=512, D=256, dtype=torch.float32, print_tb=False, bias=True):
        self.B = B
        self.N = N
        self.M = M
        self.D = D
        self.dtype = dtype
        self.print_tb = print_tb
        self.bbm = bmm()

    def run(self):
        torch.manual_seed(42)
        input_x = torch.randn(self.B, self.M, self.D, device='cuda', dtype=self.dtype)
        input_y = torch.randn(self.B, self.N, self.D, device='cuda', dtype=self.dtype)

        # Create separate tensors for input and input_ref using the same data and ensure gradient computation
        input_x_tri = input_x.clone().detach().requires_grad_(True)
        input_x_tor = input_x.clone().detach().requires_grad_(True)
        input_y_tri = input_y.clone().detach().requires_grad_(True)
        input_y_tor = input_y.clone().detach().requires_grad_(True)

        # Set the tolerance for the comparison
        rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (1e-2, 5e-2)
        self.forward(input_x_tri, input_y_tri, input_x_tor, input_y_tor, atol, rtol)

    def forward(self, input_x_tri, input_y_tri, input_x_tor, input_y_tor, atol, rtol):
        output_tri = self.bbm(input_x_tri, input_y_tri)
        output_tor = torch.einsum('bmd,bnd->bmn', input_x_tor, input_y_tor)
        assert torch.allclose(output_tri, output_tor, atol=atol, rtol=rtol), 'Error in forward pass'
        if self.print_tb:
            self.diff_f = (output_tri - output_tor).abs()
        self.backward(input_x_tri, input_y_tri, input_x_tor, input_y_tor, output_tri, output_tor, atol, rtol)

    def backward(self, input_x_tri, input_y_tri, input_x_tor, input_y_tor, output_tri, output_tor, atol, rtol):
        g = torch.rand_like(output_tor, device="cuda")
        output_tri.backward(g)
        output_tor.backward(g)
        assert torch.allclose(input_x_tri.grad, input_x_tor.grad, atol=atol, rtol=rtol), 'Error in backward pass'
        assert torch.allclose(input_y_tri.grad, input_y_tor.grad, atol=atol, rtol=rtol), 'Error in backward pass'
        if self.print_tb:
            self.diff_xb = (input_x_tri.grad - input_x_tor.grad).abs()
            self.diff_yb = (input_y_tri.grad - input_y_tor.grad).abs()
            self.table()
    
    def table(self):
        print(tb([[self.dtype, self.D, self.diff_f.mean().item(), self.diff_f.max().item(), 
                   self.diff_xb.mean().item(), self.diff_xb.max().item(),
                   self.diff_yb.mean().item(), self.diff_yb.max().item()]],
                headers=['Dype', 'Dim', 'Forward Mean Diff', 'Forward Max Diff', 
                         'Backward X Mean Diff', 'Backward X Max Diff',
                         'Backward Y Mean Diff', 'Backward Y Max Diff'], tablefmt='orgtbl'))

if __name__ == '__main__':
    B, N, M = 1, 5000, 5000
    print_tb = True
    for i in range(3):
        if i ==0: print('First iteration Slow due to Triton Autotune')
        for D in [32, 64, 128, 256, 512]:
            for dtype in [torch.float16, torch.float32]:
                runner = BMMUnitTest(B, N, M, D, dtype, print_tb)
                runner.run()
    print('All tests passed!')