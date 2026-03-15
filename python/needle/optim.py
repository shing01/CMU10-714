"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data + self.weight_decay * p.data
            if p not in self.u:
                self.u[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).data
            self.u[p] = self.momentum * self.u[p] + (1.0 - self.momentum) * grad
            p.data = p.data - self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        total_norm = 0
        for p in self.params:
            if p.grad is None:
                continue
            total_norm += (p.grad.data ** 2).sum()
        total_norm = total_norm ** 0.5
        clip_coeff = max_norm / (total_norm + 1e-6)
        if clip_coeff < 1.0:
            for p in self.params:
                if p.grad is None:
                    continue
                p.grad.data = p.grad.data * clip_coeff
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0 # 迭代步数

        self.m = {} # 一阶矩
        self.v = {} # 二阶矩

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data + self.weight_decay * p.data
            if p not in self.m:
                self.m[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).data
                self.v[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).data
            self.m[p] = self.beta1 * self.m[p] + (1.0 - self.beta1) * grad
            self.v[p] = self.beta2 * self.v[p] + (1.0 - self.beta2) * (grad ** 2)
            m_hat = self.m[p] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1.0 - self.beta2 ** self.t)

            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
