# %%
from typing import Any
from functools import wraps
import casadi as cs
import torch
import l4casadi as l4c
import numpy as np
import timeit
from torch import nn


class Cs_Torch(torch.nn.Module):

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self._layers = torch.nn.ModuleList([])

    def post_init(self, device="cpu"):
        self.cs_model = l4c.L4CasADi(self,
                                     has_batch=False,
                                     device=device,
                                     name="f_" + str(hash(self)))
        self.weights_shape = self._count_weights_shape()
        self.to_torch()

    def _count_weights_shape(self):
        counter = 0
        for param in self.parameters():
            shape = 1
            for dimention in param.shape:
                shape *= dimention
            counter += shape
        return counter

    def linear(self, in_features, out_features, bias: bool = True):
        torch_layer = torch.nn.Linear(in_features, out_features, bias)
        cs_layer = self._cs_linear(bias)

        self._layers.append(torch_layer)

        def wrapper(*args, **kwargs):
            if (self.type == "torch"):
                return torch_layer(*args, **kwargs)
            elif (self.type == "casadi"):
                return cs_layer(*args, **kwargs)

        return wrapper

    def _cs_linear(self, bias):

        def wrapper(input):
            A = self.weights_package.pop()
            if bias:
                return input @ A.T + self.weights_package.pop()
            else:
                return input @ A.T

        return wrapper

    def pack_up_input(self, input):
        if self.type == "casadi":
            x = input[:self.input_size]
            weights = input[self.input_size:]

            weights_package = []
            pointer = 0
            for param in self.parameters():
                if len(param.shape) == 2:
                    rows, cols = param.shape
                elif len(param.shape) == 1:
                    rows, cols = 1, param.shape[0]
                w = weights[pointer:pointer + rows * cols].reshape(rows, cols)
                weights_package.append(w)
                pointer = pointer + rows * cols
            weights_package.reverse()
            return x, weights_package
        elif (self.type == "torch"):
            return input, None

    def _sym_concat_with_weights(self, x):
        params = []
        for param in self.parameters():
            params.append(param.flatten().detach().numpy())
        params = np.hstack(params)
        weights = cs.MX(params)
        inp = cs.vcat([x, weights])
        return inp

    def to_torch(self):
        self.type = "torch"
        return self

    def to_casadi(self):
        self.type = "casadi"

        def wrapper(x, weights=None):
            if weights is None:
                return self.cs_model(self._sym_concat_with_weights(x))
            else:
                return self.cs_model(cs.vcat([x, weights]))

        return wrapper


# %%
### Example Of your NN model


class MultiLayerPerceptron(Cs_Torch):

    def __init__(self, input_size, dim_hidden, n_hidden_layers, is_bias=False):
        super().__init__(input_size)
        self.input_layer = self.linear(input_size, dim_hidden, bias=False)
        self.hidden_layers = [
            self.linear(dim_hidden, dim_hidden, bias=is_bias)
            for _ in range(n_hidden_layers)
        ]
        self.out_layer = self.linear(dim_hidden, 1, bias=False)
        self.post_init()

    def forward(self, inp):
        x, self.weights_package = self.pack_up_input(inp)
        x = self.input_layer(x.T)
        for layer in self.hidden_layers:
            x = torch.nn.functional.tanh(layer(x))
            # x = torch.nn.functional.leaky_relu(layer(x), negative_slope=0.1)
        x = self.out_layer(x)
        return x


# %%
### Test
in_size = 3
hid_size = 15
## Create model
pyTorch_model = MultiLayerPerceptron(in_size, hid_size, 3, is_bias=True)

## Casadi usage with sym weights
x_sym = cs.MX.sym('x', pyTorch_model.input_size, 1)
w_sym = cs.MX.sym('w', pyTorch_model.weights_shape, 1)
inp_sym = cs.vcat([x_sym, w_sym])

f_sym = pyTorch_model.to_casadi()(x_sym, w_sym)
cs_model_with_w = cs.Function('y', [inp_sym], [f_sym])
## Casadi usage with torch weights

f_sym = pyTorch_model.to_casadi()(x_sym)
cs_model = cs.Function('y', [x_sym], [f_sym])

## logs
x = np.random.random(in_size)
x_torch = torch.tensor(x, dtype=torch.float32)
x_cs = cs.DM(x)
random_w = cs.DM(np.random.random(pyTorch_model.weights_shape))
x_with_w = cs.vcat([x_cs, random_w])
pyTorch_model.to_torch()
print("torch ", pyTorch_model(x_torch).detach().numpy())
print("casadi ", cs_model(x_cs))
print("casadi with random weghts ", cs_model_with_w(x_with_w))

# %%
# speed test
# %timeit pyTorch_model(torch.tensor(np.random.random(in_size), dtype=torch.float32))
# %timeit cs_model_with_w(np.random.random(pyTorch_model.input_size + pyTorch_model.weights_shape))
# %timeit cs_model(np.random.random(pyTorch_model.input_size))


# %%
### Enother test
### More complex model
class ModelPerceptron(Cs_Torch):

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float = 0.15,
        force_positive_def: bool = False,
        is_force_infinitesimal: bool = False,
        is_bias: bool = True,
        weights=None,
    ):
        super().__init__(dim_input)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_hidden_layers = n_hidden_layers
        self.leaky_relu_coef = leaky_relu_coef
        self.force_positive_def = force_positive_def
        self.is_force_infinitesimal = is_force_infinitesimal
        self.is_bias = is_bias

        self.input_layer = self.linear(dim_input, dim_hidden, bias=is_bias)
        self.hidden_layers = [
            self.linear(dim_hidden, dim_hidden, bias=is_bias)
            for _ in range(n_hidden_layers)
        ]
        self.output_layer = self.linear(dim_hidden, dim_output, bias=is_bias)

        if weights is not None:
            self.load_state_dict(weights)

        self.post_init()

    def _forward(self, x):
        x = nn.functional.leaky_relu(self.input_layer(x.T),
                                     negative_slope=self.leaky_relu_coef)
        for layer in self.hidden_layers:
            x = nn.functional.leaky_relu(layer(x),
                                         negative_slope=self.leaky_relu_coef)
        x = self.output_layer(x)
        return x

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)
        input_tensor, self.weights_package = self.pack_up_input(input_tensor)

        if self.is_force_infinitesimal:
            return self._forward(input_tensor) - self._forward(
                torch.zeros_like(input_tensor))

        return self._forward(input_tensor)


# %%
### Test
pyTorch_model2 = ModelPerceptron(12, 1, 4, 3, is_bias=True)
x_sym = cs.MX.sym('x', pyTorch_model2.input_size, 1)

f_sym = pyTorch_model2.to_casadi()(x_sym)
cs_model = cs.Function('y', [x_sym], [f_sym])
x = np.random.random(pyTorch_model2.input_size)
x_torch = torch.tensor(x, dtype=torch.float32)
x_cs = cs.DM(x)
pyTorch_model2.to_torch()
print(pyTorch_model2(x_torch).detach().numpy())
print(cs_model(x_cs))
