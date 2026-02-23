# CMU 10-714 HW1 学习笔记：Needle 自动微分框架

## 一、HW1 目标

HW0 用纯 numpy 手推梯度、手写反向传播。HW1 的目标是从零实现一个自动微分（Autograd）框架 **Needle**，让框架自动完成链式法则的推导和梯度累积，用户只需定义每个算子的前向计算和局部梯度规则。

---

## 二、核心数据结构（autograd.py）

### 2.1 计算图的节点：Value / Tensor

每个 `Tensor` 是计算图中的一个节点，记录了三个关键信息：

```
Tensor
├── op          # 产生该节点的算子（叶节点为 None）
├── inputs      # 该算子的输入节点列表
└── cached_data # 前向计算的缓存结果（numpy array）
```

- **叶节点**：`op=None`，直接由用户创建（如 `ndl.Tensor(np.array(...))`）
- **中间节点**：由某个 `Op.__call__` 产生，`op` 和 `inputs` 记录了它在图中的位置

`requires_grad` 的传播规则：只要任意一个输入节点 `requires_grad=True`，输出节点也为 `True`。

### 2.2 算子：Op / TensorOp

每个算子需要实现两个方法：

| 方法 | 作用 |
|------|------|
| `compute(*args)` | 前向计算，接收 numpy array，返回 numpy array |
| `gradient(out_grad, node)` | 反向计算，接收上游梯度和当前节点，返回对每个输入的局部梯度 |

`gradient_as_tuple` 是辅助方法，保证 `gradient` 的返回值始终是 tuple，避免单输入算子返回裸 Tensor 时被错误地下标索引。

---

## 三、拓扑排序（find_topo_sort）

反向传播需要按**逆拓扑序**遍历计算图。实现方式是对输出节点做后序 DFS：

```python
def topo_sort_dfs(node, visited, topo_order):
    if node in visited:   # 关键：检查 node 是否已访问，不是检查 visited 是否非空
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)   # 后序：子节点先入，当前节点后入
```

后序 DFS 保证：一个节点的所有依赖（输入）都在它之前出现，反转后即为逆拓扑序（从输出到输入）。

**曾犯的 bug**：`if visited:` 检查的是集合是否非空（Python truthy），第一次调用后集合非空，后续所有节点都直接 return，整个排序失效。

---

## 四、反向传播（compute_gradient_of_variables）

核心算法是**反向模式自动微分（Reverse-mode AD）**，即从输出节点出发，沿逆拓扑序将梯度逐层传回：

```python
# 初始化：输出节点的梯度为 1（或用户指定的 out_grad）
node_to_output_grads_list[output_tensor] = [out_grad]

for node in reverse_topo_order:
    # 汇总所有上游传来的梯度（多路径时需要相加）
    grads = sum_node_list(node_to_output_grads_list[node])
    node.grad = grads

    if node.op is None:   # 叶节点，停止传播
        continue

    # 调用算子的 gradient 方法，得到对每个输入的局部梯度
    input_grads = node.op.gradient_as_tuple(grads, node)

    # 将局部梯度分发给各输入节点
    for i, input_node in enumerate(node.inputs):
        node_to_output_grads_list[input_node].append(input_grads[i])
```

**多路径梯度累加**：如果一个节点被多个后续节点使用，它会收到多份梯度，最终 `sum_node_list` 将它们相加，这正是链式法则在 DAG 上的体现。

**曾犯的 bug**：直接用 `node.op.gradient(...)` 而非 `gradient_as_tuple`，导致单输入算子（如 `AddScalar`）返回裸 Tensor 时，`input_grads[0]` 变成对张量数据的下标索引，而非取第一个梯度。

---

## 五、算子实现（ops_mathematic.py）

每个算子的梯度推导遵循链式法则：**局部梯度 × 上游梯度（out_grad）**。

### 关键算子梯度一览

| 算子 | 前向 | 梯度（对输入） |
|------|------|---------------|
| `EWiseMul` | `a * b` | `out_grad * b`, `out_grad * a` |
| `PowerScalar(n)` | `a^n` | `out_grad * n * a^(n-1)` |
| `EWiseDiv` | `a / b` | `out_grad / b`, `-out_grad * a / b^2` |
| `Transpose` | `swapaxes(a, i, j)` | `transpose(out_grad, same_axes)` |
| `Reshape` | `reshape(a, shape)` | `reshape(out_grad, input_shape)` |
| `BroadcastTo` | `broadcast(a, shape)` | `summation(out_grad, broadcast_axes)` |
| `Summation(axes)` | `sum(a, axes)` | `broadcast_to(reshape(out_grad, keepdim_shape), input_shape)` |
| `MatMul` | `a @ b` | `out_grad @ b.T`, `a.T @ out_grad` |
| `Log` | `log(a)` | `out_grad / a` |
| `Exp` | `exp(a)` | `out_grad * exp(a)` |
| `ReLU` | `max(a, 0)` | `out_grad * (a > 0)` |

### BroadcastTo 梯度的处理逻辑

广播的逆操作是对被广播的轴求和。需要找出哪些轴被广播了：
1. 输出比输入多出的前缀维度（`diff = len(out_shape) - len(in_shape)`）
2. 输入中大小为 1 但输出中大于 1 的维度

```python
diff = len(output_shape) - len(input_shape)
axes = list(range(diff))  # 新增的前缀维度
for i in range(len(input_shape)):
    if input_shape[i] == 1 and output_shape[i + diff] > 1:
        axes.append(i + diff)
```

### Summation 梯度的处理逻辑

求和的逆操作是广播。关键是先把被求和的轴恢复为大小 1（reshape），再广播回原始形状：

```python
# 例：input_shape=(3,4,5), axes=(0,2) → sum 后 shape=(4,)
# 先 reshape 为 (1,4,1)，再 broadcast 为 (3,4,5)
for axis in axes:
    new_shape[axis] = 1
grad = reshape(out_grad, new_shape)
grad = broadcast_to(grad, input_shape)
```

### MatMul 批量矩阵乘法的梯度

当输入有 batch 维度时（如 `(B, m, k) @ (k, n)`），梯度形状可能不匹配，需要对多余的前缀维度求和：

```python
grad_a = matmul(out_grad, transpose(rhs))
if grad_a.shape != lhs.shape:
    axes = tuple(range(len(grad_a.shape) - len(lhs.shape)))
    grad_a = summation(grad_a, axes=axes)
```

---

## 六、simple_ml.py：用 Needle 实现训练

### softmax_loss

```python
def softmax_loss(Z, y_one_hot):
    batch_size = Z.shape[0]
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    Z_y = ndl.summation(Z * y_one_hot, axes=(1,))
    return ndl.summation(log_sum_exp - Z_y) / batch_size
```

全程使用 needle ops，目的是让计算图可微。`loss.backward()` 时框架自动沿图反传梯度。

### nn_epoch

```python
for i in range(0, n, batch):
    X_batch = ndl.Tensor(X[i:end])          # numpy → Tensor（叶节点）
    # ... 构造 y_one_hot ...
    logits = ndl.relu(X_batch @ W1) @ W2    # 前向，建立计算图
    loss = softmax_loss(logits, y_tensor)
    loss.backward()                          # 自动反传，W1.grad / W2.grad 被填充

    # 更新：必须用 .numpy() 取值再构造新 Tensor，切断旧计算图
    W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
    W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
```

**为什么更新时必须创建新 Tensor**：如果写 `W1 = W1 - lr * W1.grad`，更新操作本身会成为计算图的一部分，下一轮 `backward()` 会错误地穿透这个更新节点。用 `.numpy()` 取出数值再包装，得到的是没有历史的干净叶节点。

---

## 七、HW0（纯 numpy）vs HW1（Needle Autograd）对比

| 维度 | HW0（纯 numpy） | HW1（Needle） |
|------|----------------|---------------|
| **梯度计算** | 手动推导每个函数的梯度公式，硬编码在训练循环里 | 每个算子只需定义局部梯度规则，框架自动应用链式法则 |
| **softmax_loss 输入** | `Z`（numpy），`y`（1D 整数标签） | `Z`（ndl.Tensor），`y_one_hot`（2D one-hot Tensor） |
| **softmax_loss 返回** | Python float（`np.mean(...)`） | ndl.Tensor（标量，可继续反传） |
| **nn_epoch 反向传播** | 手写 G2、G1、grad_W1、grad_W2 的矩阵运算 | `loss.backward()` 一行搞定 |
| **权重更新** | `W1 -= lr * grad_W1`（in-place，numpy array） | `W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())`（创建新叶节点） |
| **扩展性** | 换网络结构需要重新推导所有梯度 | 只需用已有算子组合，梯度自动处理 |
| **调试难度** | 梯度公式出错直接影响结果，难以定位 | 可以对每个算子单独做梯度检验（gradient_check） |

### 核心思想的转变

HW0 的思路是：**我知道这个函数的梯度是什么，我来写它**。

HW1 的思路是：**我只需要告诉框架每个基本操作的局部梯度，框架用链式法则自动组合出任意复杂函数的梯度**。

这正是 PyTorch、JAX 等现代深度学习框架的核心机制。HW1 实现的 Needle 是一个最小化的原型，后续 HW 会在此基础上添加更多后端（CUDA）、更多算子和更高层的模块（nn.Module、优化器等）。
