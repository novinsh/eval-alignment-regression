#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Callable

from einops import repeat, rearrange
from functools import partial

# from ..utils import default

from typing import Any

def exists(var : Any | None) -> bool:
    return var is not None

def default(var : Any | None, val : Any) -> Any:
    return var if exists(var) else val

def enlarge_as(a : Tensor, b : Tensor) -> Tensor:
    '''
        Add sufficient number of singleton dimensions
        to tensor a **to the right** so to match the
        shape of tensor b. NOTE that simple broadcasting
        works in the opposite direction.
    '''
    return rearrange(a, f'... -> ...{" 1" * (b.dim() - a.dim())}').contiguous()

def saturating_func(
    x : Tensor,
    conv_f : Callable[[Tensor], Tensor] = None,
    conc_f : Callable[[Tensor], Tensor] = None,
    slope : float = 1.,
    const : float = 1.,
) -> Tensor:
    conv = conv_f(+torch.ones_like(x) * const)

    return slope * torch.where(
        x <= 0,
        conv_f(x + const) - conv,
        conc_f(x - const) + conv,
    )

class MonotonicLinear(nn.Linear):
    '''
        Monotonic Linear Layer as introduced in:
        `Constrained Monotonic Neural Networks` ICML (2023).

        Code is a PyTorch implementation of the official repository:
        https://github.com/airtai/mono-dense-keras/
    '''

    def __init__(
        self,
        in_features  : int,
        out_features : int,
        bias : bool = True,
        gate_func : str = 'elu',
        indicator : int | Tensor | None = None,
        act_weight : str | Tuple[float, float, float] = (7, 7, 2),
    ) -> None:
        # Assume positive monotonicity in all input features
        indicator = default(indicator, torch.ones(in_features))

        if isinstance(indicator, int):
            indicator = torch.ones(in_features) * indicator

        assert indicator.dim() == 1, 'Indicator tensor must be 1-dimensional.'
        assert indicator.size(-1) == in_features, 'Indicator tensor must have the same number of elements as the input features.'
        assert len(act_weight) == 3, f'Relative activation weights should have len = 3. Got {len(act_weight)}'
        if isinstance(act_weight, str): assert act_weight in ('concave', 'convex')

        self.indicator = indicator

        # Compute the three activation functions: concave|convex|saturating
        match gate_func:
            case 'elu' : self.act_conv = F.elu
            case 'silu': self.act_conv = F.silu
            case 'gelu': self.act_conv = F.gelu
            case 'relu': self.act_conv = F.relu
            case 'selu': self.act_conv = F.selu
            case _: raise ValueError(f'Unknown gating function {gate_func}')

        self.act_conc = lambda t : -self.act_conv(-t)
        self.act_sat = partial(
            saturating_func,
            conv_f=self.act_conv,
            conc_f=self.act_conc,
        )

        match act_weight:
            case 'concave': self.act_weight = torch.tensor((1, 0, 0))
            case 'convex' : self.act_weight = torch.tensor((0, 1, 0))
            case _: self.act_weight = torch.tensor(act_weight) / sum(act_weight)

        # Build the layer weights and bias
        super(MonotonicLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x : Tensor) -> Tensor:
        '''
        '''

        # Get the absolute values of the weights
        abs_weights = self.weight.data.abs()

        # * Use monotonicity indicator T to adjust the layer weights
        # * T_i = +1 -> W_ij <=  || W_ij ||
        # * T_i = -1 -> W_ij <= -|| W_ij ||
        # * T_i =  0 -> do nothing
        mask_pos = self.indicator == +1
        mask_neg = self.indicator == -1

        self.weight.data[..., mask_pos] = +abs_weights[..., mask_pos]
        self.weight.data[..., mask_neg] = -abs_weights[..., mask_neg]

        # Get the output of linear layer
        out = super().forward(x)

        # Compute output by adding non-linear gating according to
        # relative importance of activations
        s_conv, s_conc, _ = (self.act_weight * self.out_features).round()
        s_conv = int(s_conv)
        s_conc = int(s_conc)
        s_sat = self.out_features - s_conv - s_conc

        i_conv, i_conc, i_sat = torch.split(
            out, (s_conv, s_conc, s_sat), dim=-1
        )

        out = torch.cat((
                self.act_conv(i_conv),
                self.act_conc(i_conc),
                self.act_sat (i_sat),
            ),
            dim=-1,
        )

        return out

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Toy example to estimate a monotonic function
# Define a simple function to approximate, e.g., y = x^3 (monotonic)
def true_function(x):
    return x**3

normalize_flag = False

# Normalize function (scaling to [-1, 1] range)
def normalize(y):
    if normalize_flag:
        min_y, max_y = y.min(), y.max()
        return 2 * (y - min_y) / (max_y - min_y) - 1  # Scale to [-1, 1]
    else:
        return y

# Inverse normalization function
def inverse_normalize(y_norm, min_y, max_y):
    if normalize_flag:
        return (y_norm + 1) * (max_y - min_y) / 2 + min_y
    else:
        return y_norm

# Create input data (x)
x = torch.linspace(-10, 10, 1000).view(-1, 1)

# True function outputs
y_true = true_function(x)

# Normalize the true function outputs to [-1, 1]
y_true_normalized = normalize(y_true)

# Define a simple network with the MonotonicLinear layer
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.layer1 = MonotonicLinear(1, 20, gate_func='selu', indicator=torch.tensor([1]))
        # self.layer2 = MonotonicLinear(10, 10, gate_func='relu', indicator=torch.tensor([1]*10))
        if normalize_flag:
            self.layer3 = MonotonicLinear(20, 1, gate_func='selu', indicator=torch.tensor([1]*20))
        else:
            self.layer3 = MonotonicLinear(20, 10, gate_func='selu', indicator=torch.tensor([1]*20))
            self.layer4 = nn.Linear(10,1, bias=False)
        # self.scaling_factor = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) if not normalize_flag else x
        # x = self.scaling_factor*x if not normalize_flag else x
        return x

# Instantiate the model and define a loss function
model = ToyModel()
criterion = nn.MSELoss()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y_true_normalized)  # Use normalized output for loss
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plot the results
model.eval()
with torch.no_grad():
    predicted_normalized = model(x)  # Predicted output in normalized range

# Inverse map the predictions to the original scale
min_y, max_y = y_true.min(), y_true.max()
predicted = inverse_normalize(predicted_normalized, min_y, max_y)

# Plot both the true and predicted functions
plt.figure(figsize=(8,4))

# Plot original scale
plt.subplot(1, 2, 1)
plt.plot(x.numpy(), y_true.numpy(), label="True Function $x^3$", color="blue")
plt.plot(x.numpy(), predicted.numpy(), label="Predicted Function", color="red", linestyle="--")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Monotonic Linear Layer Approximation\nOriginal Scale")

# Plot normalized scale
plt.subplot(1, 2, 2)
plt.plot(x.numpy(), y_true_normalized.numpy(), label="True Function $x^3$ (Normalized)", color="blue")
plt.plot(x.numpy(), predicted_normalized.numpy(), label="Predicted Function (Normalized)", color="red", linestyle="--")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Monotonic Linear Layer Approximation\nNormalized Scale")

plt.tight_layout()
plt.show()
