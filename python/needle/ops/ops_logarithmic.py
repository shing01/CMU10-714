from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=-1, keepdims=True)

        diff = Z - max_z
        sum_exp = array_api.sum(array_api.exp(diff), axis=-1, keepdims=True)

        lse = array_api.log(sum_exp) + max_z

        return Z - lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        axis = len(Z.shape) - 1

        sum_out_grad = summation(out_grad, axes=(axis,))
        new_shape = list(Z.shape)
        new_shape[axis] = 1

        sum_grad_reshaped = reshape(sum_out_grad, tuple(new_shape))
        sum_grad_broadcasted = broadcast_to(sum_grad_reshaped, Z.shape)

        return (out_grad - sum_grad_broadcasted * exp(node),)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_reduced = array_api.max(Z, axis=self.axes)

        diff = Z - max_z
        sum_exp = array_api.sum(array_api.exp(diff), axis=self.axes)                                                                                                             
        lse = array_api.log(sum_exp) + max_z_reduced 

        return lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        input_shape = Z.shape
        axes = range(len(input_shape)) if self.axes is None else self.axes
        if isinstance(axes, int): 
            axes = (axes,)

        new_shape = list(input_shape)
        for axis in axes:
            new_shape[axis] = 1
        
        node_reshaped = reshape(node, tuple(new_shape))
        node_broadcasted = broadcast_to(node_reshaped, input_shape)

        grad_reshaped = reshape(out_grad, tuple(new_shape))
        grad_broadcasted = broadcast_to(grad_reshaped, input_shape)

        return (grad_broadcasted * exp(Z - node_broadcasted),)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)