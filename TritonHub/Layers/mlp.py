from torch import nn
from TritonHub.Layers import Linear
from TritonHub import Activations

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 activation="SiLU",
                 bias=False,
                 multiple_of=128,
                 mlp_type="gated_mlp",
                 device=None,
                 dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (hidden_features if hidden_features is not None else int(8 * in_features / 3))
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        if mlp_type == "gated_mlp":
            self.fc1 = Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        elif mlp_type == "ffn":
            self.fc1 = Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        else:
            raise NotImplementedError(f"MLP type {mlp_type} not implemented")
        assert hasattr(Activations, activation), f"Activation {activation} not found in TritonHub.Activations"
        self.activation = getattr(Activations, activation)()
        self.fc2 = Linear(hidden_features, out_features, bias=bias, **factory_kwargs)
        self._fwd = self._gated_mlp if mlp_type == "gated_mlp" else self._ffn

    def forward(self, x):
        return self._fwd(x)
    
    def _gated_mlp(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y
    
    def _ffn(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y