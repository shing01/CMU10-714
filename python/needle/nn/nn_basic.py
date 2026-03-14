"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        weight_tensor = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(weight_tensor)

        if bias:
            bias_tensor = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = Parameter(bias_tensor.reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias:
            broadcast_bias = ops.broadcast_to(self.bias, out.shape)
            out = out + broadcast_bias
        
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        dim = 1
        for i in range(1, len(X.shape)):
            dim *= X.shape[i]
        
        return ops.reshape(X, (batch_size, dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        num_classes = logits.shape[1]
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        z_y = ops.summation(logits * y_one_hot, axes=(1,))
        lse = ops.logsumexp(logits, axes=(1,))
        loss = ops.summation(lse - z_y) / logits.shape[0]

        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

        # 运行时统计量，不需要梯度，不用Parameter包裹
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]

        weight_reshaped = ops.reshape(self.weight, (1, self.dim))
        bias_reshaped = ops.reshape(self.bias, (1, self.dim))
        weight_broadcasted = ops.broadcast_to(weight_reshaped, x.shape)
        bias_broadcasted = ops.broadcast_to(bias_reshaped, x.shape)

        if self.training:
            mean = ops.summation(x, axes=(0,)) / batch_size
            mean_reshaped = ops.reshape(mean, (1, self.dim))
            mean_broadcasted = ops.broadcast_to(mean_reshaped, x.shape)

            diff = x - mean_broadcasted
            var = ops.summation(diff ** 2, axes=(0,)) / batch_size
            var_reshaped = ops.reshape(var, (1, self.dim))
            var_broadcasted = ops.broadcast_to(var_reshaped, x.shape)
            x_norm = diff / (var_broadcasted + self.eps) ** 0.5

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var.data
            
            return x_norm * weight_broadcasted + bias_broadcasted
        else:
            running_mean_reshaped = ops.reshape(self.running_mean, (1, self.dim))
            running_var_reshaped = ops.reshape(self.running_var, (1, self.dim))
            running_mean_broadcasted = ops.broadcast_to(running_mean_reshaped, x.shape)
            running_var_broadcasted = ops.broadcast_to(running_var_reshaped, x.shape)

            diff = x - running_mean_broadcasted
            x_norm = diff / (running_var_broadcasted + self.eps) ** 0.5

            return x_norm * weight_broadcasted + bias_broadcasted
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]

        mean = ops.summation(x, axes=(1,)) / self.dim
        mean_reshaped = ops.reshape(mean, (batch_size, 1))
        mean_broadcasted = ops.broadcast_to(mean_reshaped, x.shape)

        diff = x - mean_broadcasted
        var = ops.summation(diff ** 2, axes=(1,)) / self.dim
        var_reshaped = ops.reshape(var, (batch_size, 1))
        var_broadcasted = ops.broadcast_to(var_reshaped, x.shape)

        x_norm = diff / (var_broadcasted + self.eps) ** 0.5
        weight_reshaped = ops.reshape(self.weight, (1, self.dim))
        bias_reshaped = ops.reshape(self.bias, (1, self.dim))
        weight_broadcasted = ops.broadcast_to(weight_reshaped, x.shape)
        bias_broadcasted = ops.broadcast_to(bias_reshaped, x.shape)

        return x_norm * weight_broadcasted + bias_broadcasted
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training and self.p:
            mask = init.randb(*x.shape, p=1.0 - self.p, device=x.device, dtype=x.dtype)

            return x * mask / (1.0 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
