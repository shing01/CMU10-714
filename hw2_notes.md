# CMU 10-714 HW2 总结笔记

## 整体结构

HW2 在 HW1 的 autograd 框架基础上，构建了一个完整的神经网络训练系统：

```
Q1 权重初始化 → Q2 算子与 nn 模块 → Q3 优化器 → Q4 数据加载 → Q5 MLP ResNet 训练
```

所有组件最终在 Q5 中串联起来，完成 MNIST 分类任务。

---

## Q1: 权重初始化 (`init/init_initializers.py`)

四种初始化方法，核心是根据 fan_in / fan_out 计算分布的范围或标准差：

| 方法 | 分布 | 关键公式 |
|------|------|----------|
| Xavier Uniform | U(-a, a) | `a = gain * sqrt(6 / (fan_in + fan_out))` |
| Xavier Normal | N(0, std²) | `std = gain * sqrt(2 / (fan_in + fan_out))` |
| Kaiming Uniform | U(-b, b) | `b = sqrt(2) * sqrt(3 / fan_in)` |
| Kaiming Normal | N(0, std²) | `std = sqrt(2) / sqrt(fan_in)` |

要点：
- Kaiming 系列的 gain 固定为 `sqrt(2)`（ReLU 激活函数）
- 所有函数返回 `(fan_in, fan_out)` 形状的 Tensor
- `**kwargs` 透传给底层的 `rand` / `randn`（用于传递 device、dtype 等）

---

## Q2a: LogSumExp 与 LogSoftmax (`ops/ops_logarithmic.py`)

### LogSumExp

数值稳定公式：`max(z) + log(sum(exp(z - max(z))))`

**compute 要点：**
- `max_z` 用 `keepdims=True` → 用于广播做 `Z - max_z`
- `max_z_reduced` 不用 keepdims → 用于最终加回，保证输出 shape 正确（reduce 掉指定 axes）
- `sum` 也不用 keepdims

```python
max_z = array_api.max(Z, axis=self.axes, keepdims=True)
max_z_reduced = array_api.max(Z, axis=self.axes)
return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z_reduced
```

**gradient 要点：**
- LogSumExp 的梯度就是 softmax：`exp(z - LSE(z))`
- 需要把 `node`（LSE 的输出）和 `out_grad` reshape 回 keepdims 形状再 broadcast
- 与 `Summation.gradient` 相同的 pattern：构建 `new_shape`（被 reduce 的 axis 设为 1）→ reshape → broadcast

```python
new_shape = list(input_shape)
for axis in axes:
    new_shape[axis] = 1
# reshape + broadcast node 和 out_grad，然后：
return grad_broadcasted * exp(Z - node_broadcasted)
```

### LogSoftmax

- compute: `Z - LogSumExp(Z, axis=-1, keepdims=True)`
- gradient: `out_grad - softmax * sum(out_grad)`，其中 softmax = `exp(node)`（node 就是 logsoftmax 的输出）

---

## Q2b: NN 模块 (`nn/nn_basic.py`)

### Linear

- weight: `kaiming_uniform(in_features, out_features)` → 形状 `(in, out)`
- bias: `kaiming_uniform(out_features, 1)` → reshape 成 `(1, out_features)`
  - 注意 bias 的 `fan_in = out_features`，不是 `in_features`
- forward: `X @ weight + broadcast_to(bias, out.shape)`
- 所有可学习参数用 `Parameter()` 包裹

### BatchNorm1d

- 训练时：计算 batch 维度的 mean/var，归一化后乘 weight 加 bias
- 更新 running stats：`running = (1 - momentum) * running + momentum * observed`
- 评估时：用 running_mean / running_var 归一化
- running stats 不用 `Parameter` 包裹（不需要梯度）
- 更新 running stats 时用 `.data` 脱离计算图

### LayerNorm1d

- 与 BatchNorm 类似，但沿 feature 维度（axis=1）归一化
- 没有 running stats，训练和评估行为一致

### Dropout

- 训练时：以概率 `1-p` 生成 mask，`x * mask / (1 - p)`（inverted dropout）
- 评估时：直接返回 x
- 用 `init.randb` 生成伯努利 mask

### 其他简单模块

- **ReLU**: `ops.relu(x)`
- **Flatten**: reshape 成 `(batch_size, -1)`
- **Sequential**: 按顺序调用各子模块
- **Residual**: `x + fn(x)`
- **SoftmaxLoss**: `mean(logsumexp(logits, axes=1) - z_y)`，用 `one_hot` 提取 `z_y`

---

## Q3: 优化器 (`optim.py`)

### SGD（带动量）

```
grad = p.grad.data + weight_decay * p.data    # L2 正则
u = β * u + (1 - β) * grad                     # 动量更新
θ = θ - lr * u                                  # 参数更新
```

### Adam

```
grad = p.grad.data + weight_decay * p.data
m = β1 * m + (1 - β1) * grad                   # 一阶矩
v = β2 * v + (1 - β2) * grad²                  # 二阶矩
m_hat = m / (1 - β1^t)                          # 偏差修正
v_hat = v / (1 - β2^t)
θ = θ - lr * m_hat / (sqrt(v_hat) + eps)
```

**共同要点：**
- 全程用 `.data` 操作，避免把优化器计算纳入计算图（否则 memory_check 测试会挂）
- 动量/矩估计存在字典中，key 是参数对象本身
- weight_decay 是加在梯度上的（L2 正则化）

---

## Q4: 数据加载 (`data/`)

### Transforms

- **RandomFlipHorizontal**: `img[:, ::-1, :]`（沿 W 轴翻转）
- **RandomCrop**: 先 zero-pad 四周，再从 `(pad + shift_x, pad + shift_y)` 处裁剪回原始大小
  - 注意偏移量要加上 padding 作为基准

### MNISTDataset

- `__init__`: 用 `gzip` + `struct` 解析 IDX 格式，images 归一化到 [0,1]，dtype=float32
- `__getitem__`: 取数据后 reshape 成 `(28, 28, 1)` 再 apply transforms
- 必须调用 `super().__init__(transforms)` 让父类设置 `self.transforms`

### DataLoader

- `__iter__`: 生成 index 序列（shuffle 时打乱），按 batch_size 切分，重置计数器
- `__next__`: 取当前 batch 的 indices，从 dataset 取数据，包成 `tuple(Tensor(x) for x in batch_data)`

---

## Q5: MLP ResNet (`apps/mlp_resnet.py`)

### 网络结构

```
ResidualBlock(dim, hidden_dim):
    Residual(Linear → Norm → ReLU → Dropout → Linear → Norm) + ReLU

MLPResNet(dim, hidden_dim, num_blocks, num_classes):
    Linear(dim, hidden_dim) → ReLU → [ResidualBlock × num_blocks] → Linear(hidden_dim, num_classes)
```

- ResidualBlock 内部的 hidden_dim 是 `hidden_dim // 2`
- 必须用循环构建 num_blocks 个 ResidualBlock（测试会传不同的 num_blocks）

### 训练循环

- `epoch()`: 标准的 train/eval 循环，返回 `(error_rate, avg_loss)`
  - 有 opt → `model.train()`，无 opt → `model.eval()`
- `train_mnist()`: 创建数据集/加载器/模型/优化器，训练 N 个 epoch，最后在测试集上评估

---

## 踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| `init/__init__.py` 循环导入 | 文件被 HW2 scaffold 覆盖成了 `needle/__init__.py` 的内容 | 恢复为 `from .init_basic import *` + `from .init_initializers import *` |
| LogSumExp forward shape 不对 | `sum` 用了 `keepdims=True`，输出多了维度 | `sum` 不要 keepdims，只有 `max` 做广播时需要 keepdims |
| Linear bias 初始化报错 | 传了 `shape=` 参数给 `kaiming_uniform` | 用 `kaiming_uniform(out_features, 1)` 再 reshape |
| MNISTDataset 没有 transforms 属性 | 没调用 `super().__init__(transforms)` | 加上 super 调用 |
| RandomCrop 裁剪越界 | 偏移量没加 padding 基准 | `pad + shift_x` 作为起始位置 |
| MLPResNet 参数量不对 | 硬编码 block 数量 + 多余的 Linear/ReLU | 用循环构建，去掉多余层 |
| train_mnist 只跑 1 个 epoch | return 写在 for 循环里面 | 移到循环外 |
| ResidualBlock 的 norm 变成 float | 位置参数传递顺序错误 | 用关键字参数 `norm=norm, drop_prob=drop_prob` |
