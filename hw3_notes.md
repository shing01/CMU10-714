# CMU 10-714 HW3 学习笔记：NDArray 后端实现（CPU + CUDA）

## 一、HW3 目标

HW1/HW2 中 Needle 框架的底层数据全部由 numpy 承载。HW3 的目标是为 Needle 实现一套**自有的 n 维数组库（NDArray）**，包含三层：

1. **Python 层**（`ndarray.py`）：通过操作 shape / strides / offset 实现零拷贝的视图变换
2. **CPU 后端**（`ndarray_backend_cpu.cc`）：用 C++ 实现底层内存操作，通过 pybind11 暴露给 Python
3. **CUDA 后端**（`ndarray_backend_cuda.cu`）：用 CUDA C++ 实现 GPU 并行版本

这是整个框架从"能跑"到"跑得快"的关键一步——后续 HW4 的卷积、RNN 等都将建立在这套后端之上。

---

## 二、核心概念：NDArray 的内存模型

### 2.1 Compact vs Non-compact

NDArray 的核心思想是**一块连续的底层内存（handle）+ 一组元数据（shape, strides, offset）**来描述一个多维数组视图。

```
底层内存 (handle):  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

视图 A (shape=(3,4), strides=(4,1), offset=0):     → compact，行优先连续存储
视图 B (shape=(4,3), strides=(1,4), offset=0):     → non-compact，转置视图
视图 C (shape=(2,2), strides=(4,2), offset=1):     → non-compact，切片视图
```

- **Compact 数组**：strides 等于 `compact_strides(shape)`（即最后一维 stride=1，倒数第二维 stride=shape[-1]，依此类推），offset=0。数据在内存中连续排列。
- **Non-compact 数组**：经过 permute / slice / broadcast 后，strides 不再是标准的 compact 形式，但仍然指向同一块底层内存。

### 2.2 Strides 的含义

`strides[i]` 表示沿第 i 维移动一个元素时，在底层一维数组中需要跳过的元素数。

```python
# shape=(3,4) 的 compact strides
strides = (4, 1)   # 沿 axis=0 跳 4 个元素，沿 axis=1 跳 1 个元素

# 转置后 shape=(4,3)
strides = (1, 4)   # 沿 axis=0 跳 1 个元素，沿 axis=1 跳 4 个元素
```

### 2.3 从多维索引到物理地址

给定多维索引 `(i0, i1, ..., in)`，对应的物理地址为：

```
physical_addr = offset + i0 * strides[0] + i1 * strides[1] + ... + in * strides[n]
```

这个公式是整个 HW3 最核心的公式，Compact / SetItem 的实现都围绕它展开。

---

## 三、Part 1：Python 视图操作（ndarray.py）

这四个方法都**不拷贝内存**，只是创建新的视图（修改 shape / strides / offset）。

### 3.1 reshape(new_shape)

**前提条件**：数组必须是 compact 的（否则无法仅通过改 strides 来重新解释内存布局）。

**实现**：保持 handle 和 offset 不变，用新 shape 计算新的 compact strides。

```python
def reshape(self, new_shape):
    assert self.is_compact(), "Reshape requires compact array"
    assert prod(new_shape) == prod(self._shape), "Product of shapes must match"
    new_strides = NDArray.compact_strides(new_shape)  # 或手动计算
    return NDArray.make(shape=new_shape, strides=new_strides,
                        device=self.device, handle=self._handle, offset=self._offset)
```

### 3.2 permute(new_axes)

**作用**：重排维度顺序。例如 `(0,3,1,2)` 将 BHWC 转为 BCHW。

**实现**：按 `new_axes` 的顺序重排 shape 和 strides，不改变 handle 和 offset。

```python
def permute(self, new_axes):
    new_shape = tuple(self.shape[ax] for ax in new_axes)
    new_strides = tuple(self.strides[ax] for ax in new_axes)
    return NDArray.make(shape=new_shape, strides=new_strides,
                        device=self.device, handle=self._handle, offset=self._offset)
```

**直觉**：permute 后数组通常变成 non-compact 的，后续运算前框架会自动调用 `compact()` 将数据重新排列到连续内存。

### 3.3 broadcast_to(new_shape)

**作用**：将 size=1 的维度"广播"到任意大小，不拷贝数据。

**实现**：对于 `old_shape[i] == 1` 的维度，将 `stride[i]` 设为 0（这样沿该维度移动时始终读同一个元素）。

```python
def broadcast_to(self, new_shape):
    new_strides = []
    for old_s, new_s, old_stride in zip(self.shape, new_shape, self.strides):
        if old_s == new_s:
            new_strides.append(old_stride)
        elif old_s == 1:
            new_strides.append(0)  # 关键：stride=0 实现广播
        else:
            raise ValueError("Shape mismatch")
    return NDArray.make(shape=new_shape, strides=tuple(new_strides),
                        device=self.device, handle=self._handle, offset=self._offset)
```

**stride=0 的妙处**：物理地址公式中 `i * 0 = 0`，无论索引 i 是多少，都读到同一个位置的数据，实现了"虚拟复制"。

### 3.4 \_\_getitem\_\_(slices)

**作用**：支持 `a[1:5, :-1:2, 4, :]` 这样的切片操作。

**实现**：框架的 stub 代码已经将所有索引统一为 `slice(start, stop, step)` 元组。对每个维度：

```python
new_shape[i]   = ceil((stop - start) / step)
new_strides[i] = old_strides[i] * step    # step > 1 时跳过元素
new_offset    += start * old_strides[i]    # 起始位置偏移
```

**注意**：stub 代码已经处理了负索引、None 值、整数索引转 slice 等边界情况，实现时直接用处理好的 `slices` 变量即可，不要重复处理 `idxs`。

---

## 四、Part 2：CPU 后端 — Compact 与 SetItem

### 4.1 核心算法：从 compact index 反算多维索引

CPU 后端的 Compact / EwiseSetitem / ScalarSetitem 三个函数共享同一套索引遍历逻辑。核心问题是：给定一个 compact 的线性索引 `cnt`（0, 1, 2, ...），如何算出它在 strided 数组中的物理地址？

```
步骤：cnt → 多维索引 (i0, i1, ..., in) → 物理地址
```

从最后一维开始，逐维取模和整除：

```cpp
size_t physical_idx = offset;
size_t temp = cnt;
for (int i = ndim - 1; i >= 0; --i) {
    physical_idx += (temp % shape[i]) * strides[i];
    temp /= shape[i];
}
```

**原理**：这和十进制数字拆分是一样的。比如 `cnt=23`，`shape=(3,4,5)`：
- `23 % 5 = 3` → 最后一维索引是 3
- `23 / 5 = 4`，`4 % 4 = 0` → 中间维索引是 0
- `4 / 4 = 1`，`1 % 3 = 1` → 第一维索引是 1
- 多维索引 = (1, 0, 3)

### 4.2 三个函数的区别

| 函数 | 读取方 | 写入方 | 用途 |
|------|--------|--------|------|
| `Compact` | strided (a + offset) | compact (out) | 将 non-compact 视图拷贝为连续内存 |
| `EwiseSetitem` | compact (a) | strided (out + offset) | 将连续数据写入 non-compact 视图 |
| `ScalarSetitem` | 标量 val | strided (out + offset) | 将标量填入 non-compact 视图 |

三者的索引遍历逻辑完全相同，只是读写方向不同。

---

## 五、Part 3：CPU 后端 — 逐元素与标量运算

### 5.1 宏模板化

所有逐元素运算的结构完全相同，只是运算符不同。用 C++ 宏避免重复代码：

```cpp
#define ALIGNED_ARRAY_EWISE(name, op) \
  void name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
    for (size_t i = 0; i < a.size; i++) out->ptr[i] = op; \
  }

#define ALIGNED_ARRAY_SCALAR(name, op) \
  void name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
    for (size_t i = 0; i < a.size; i++) out->ptr[i] = op; \
  }

#define ALIGNED_ARRAY_SINGLE(name, op) \
  void name(const AlignedArray& a, AlignedArray* out) { \
    for (size_t i = 0; i < a.size; i++) out->ptr[i] = op; \
  }
```

### 5.2 运算一览

| 宏调用 | 运算表达式 |
|--------|-----------|
| `EwiseMul` | `a.ptr[i] * b.ptr[i]` |
| `ScalarPower` | `std::pow(a.ptr[i], val)` |
| `EwiseMaximum` | `std::max(a.ptr[i], b.ptr[i])` |
| `EwiseEq` | `a.ptr[i] == b.ptr[i] ? 1.0f : 0.0f` |
| `EwiseLog` | `std::log(a.ptr[i])` |
| `EwiseTanh` | `std::tanh(a.ptr[i])` |

**注意**：比较运算（Eq, Ge）返回的是 `float`（1.0f 或 0.0f），不是 bool。

### 5.3 pybind11 绑定

每实现一个 C++ 函数，都需要在 `PYBIND11_MODULE` 中注册：

```cpp
m.def("ewise_mul", EwiseMul);
m.def("scalar_mul", ScalarMul);
// ...
```

初始代码中这些绑定是被注释掉的，实现函数后必须取消注释，否则 Python 端调用时会报 `AttributeError`。

---

## 六、Part 4：CPU 后端 — Reduce 操作

### 6.1 Python 层的预处理

Python 层在调用 C++ 的 reduce 之前，已经做了关键的预处理：
1. 将要 reduce 的轴 permute 到最后一维
2. 调用 `compact()` 使数据连续

所以 C++ 端收到的数据是这样的：

```
输入 a: [block_0 | block_1 | ... | block_{n-1}]
         ←reduce_size→
输出 out: [result_0, result_1, ..., result_{n-1}]
```

### 6.2 实现

每个输出元素对应输入中连续 `reduce_size` 个元素的归约：

```cpp
// ReduceMax
for (size_t i = 0; i < num_blocks; ++i) {
    scalar_t max_val = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; ++j) {
        max_val = std::max(max_val, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = max_val;
}

// ReduceSum
for (size_t i = 0; i < num_blocks; ++i) {
    scalar_t sum_val = 0.0f;
    for (size_t j = 0; j < reduce_size; ++j) {
        sum_val += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum_val;
}
```

**注意**：ReduceMax 的初始值用 `a[base]`（第一个元素），不能用 0 或 `-inf`；ReduceSum 的初始值用 `0.0f`。

---

## 七、Part 5：CPU 后端 — 矩阵乘法

### 7.1 朴素 Matmul

三重循环实现 `C[m×p] = A[m×n] × B[n×p]`：

```cpp
// 先清零
for (size_t i = 0; i < m * p; ++i) out->ptr[i] = 0.0f;

// ikj 循环顺序（对缓存更友好）
for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < n; ++k) {
        scalar_t A_ik = a.ptr[i * n + k];
        for (size_t j = 0; j < p; ++j) {
            out->ptr[i * p + j] += A_ik * b.ptr[k * p + j];
        }
    }
}
```

**为什么用 ikj 而不是 ijk？** 在 ikj 顺序下，内层循环遍历 B 的一行和 out 的一行，两者在内存中都是连续的，缓存命中率高。ijk 顺序下内层循环遍历 B 的一列，每次跳 p 个元素，缓存不友好。

### 7.2 Tiled Matmul（分块矩阵乘法）

为了进一步利用 CPU 的 SIMD 向量化能力，作业设计了分块方案：

**数据布局**：矩阵被重新排列为 4D tiled 格式：
- `A[m/TILE][n/TILE][TILE][TILE]` — 每个 TILE×TILE 的小块在内存中连续
- `B[n/TILE][p/TILE][TILE][TILE]`
- `out[m/TILE][p/TILE][TILE][TILE]`

**AlignedDot**：TILE×TILE 的小矩阵乘法，结果**累加**到 out（不清零）。使用 `__restrict__` 和 `__builtin_assume_aligned` 提示编译器做向量化：

```cpp
inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {
    a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
    b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
    out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

    for (size_t i = 0; i < TILE; ++i)
        for (size_t k = 0; k < TILE; ++k) {
            scalar_t A_ik = a[i * TILE + k];
            for (size_t j = 0; j < TILE; ++j)
                out[i * TILE + j] += A_ik * b[k * TILE + j];
        }
}
```

- `__restrict__`：告诉编译器 a、b、out 不会互相别名（alias），可以安全地向量化
- `__builtin_assume_aligned`：告诉编译器指针是对齐的，可以使用 AVX 等对齐加载指令
- 作业推荐用 `clang++` 编译，因为 clang 的自动向量化比 gcc 更激进

**MatmulTiled**：在 tile 级别做分块乘法：

```cpp
for (i in m_tiles)
    for (j in p_tiles)
        for (k in n_tiles)
            AlignedDot(A_tile[i][k], B_tile[k][j], Out_tile[i][j]);
```

**关键**：out 必须先清零（`AlignedDot` 是累加的），然后对每对 (i,j) 输出 tile，遍历所有 k 方向的 tile 对，累加结果。

---

## 八、CUDA 编程基础

### 8.1 执行模型

CUDA 的并行模型是**SIMT（Single Instruction, Multiple Threads）**：

```
Grid（网格）
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ...Thread 255
├── Block 1
│   └── ...
└── Block N
    └── ...
```

- **Grid**：一次 kernel 启动的所有线程
- **Block**：一组可以共享 shared memory 和同步的线程（本作业用 256 个线程/block）
- **Thread**：最小执行单元

每个线程通过 `blockIdx.x * blockDim.x + threadIdx.x` 计算自己的全局 ID（gid）。

### 8.2 Kernel 函数的基本模式

```cuda
__global__ void MyKernel(const scalar_t* input, scalar_t* output, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {          // 边界检查！
        output[gid] = f(input[gid]);
    }
}

// Host 端启动
CudaDims dim = CudaOneDim(size);
MyKernel<<<dim.grid, dim.block>>>(input_ptr, output_ptr, size);
```

**边界检查 `if (gid < size)` 是必须的**：因为线程总数 = `num_blocks × 256`，通常会比实际数据量多出一些"多余"线程，这些线程不应该访问越界内存。

### 8.3 Host 与 Device 的分工

| 层次 | 运行位置 | 职责 |
|------|---------|------|
| `void Xxx(...)` | CPU (Host) | 计算 grid/block 维度，准备参数，启动 kernel |
| `__global__ void XxxKernel(...)` | GPU (Device) | 每个线程执行一个元素的计算 |
| `__device__ size_t index_transform(...)` | GPU (Device) | 被 kernel 调用的辅助函数 |

### 8.4 CudaVec：向 Kernel 传递变长数据

CUDA kernel 不能接收 `std::vector`（它是 host 端的堆内存）。框架提供了 `CudaVec` 结构体：

```cpp
#define MAX_VEC_SIZE 8
struct CudaVec {
    uint32_t size;
    int32_t data[MAX_VEC_SIZE];  // 固定大小数组，可以按值传递给 kernel
};
```

Host 端用 `VecToCuda()` 将 `std::vector` 转为 `CudaVec`，然后按值传给 kernel。

---

## 九、Part 6：CUDA 后端 — Compact 与 SetItem

### 9.1 index_transform：共享的索引转换函数

CPU 版本中三个函数各自内联了索引计算逻辑。CUDA 版本提取为一个 `__device__` 函数，被三个 kernel 共享：

```cuda
__device__ size_t index_transform(size_t gid, CudaVec shape, CudaVec strides, size_t offset) {
    size_t physical_idx = offset;
    size_t temp = gid;
    for (int i = shape.size - 1; i >= 0; --i) {
        physical_idx += (temp % shape.data[i]) * strides.data[i];
        temp /= shape.data[i];
    }
    return physical_idx;
}
```

### 9.2 三个 Kernel 的实现

```cuda
// Compact: strided → compact
__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size,
                               CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = a[index_transform(gid, shape, strides, offset)];
    }
}

// EwiseSetitem: compact → strided
__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size,
                                    CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[index_transform(gid, shape, strides, offset)] = a[gid];
    }
}

// ScalarSetitem: scalar → strided
__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size,
                                     CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[index_transform(gid, shape, strides, offset)] = val;
    }
}
```

### 9.3 CPU vs CUDA 的对比

| 维度 | CPU | CUDA |
|------|-----|------|
| 并行方式 | for 循环串行遍历 | 每个线程处理一个元素 |
| 数据传递 | `std::vector` 直接传 | 需要转为 `CudaVec` 按值传递 |
| 索引计算 | 内联在每个函数中 | 提取为 `__device__` 函数复用 |
| 边界检查 | for 循环自然保证 | 需要 `if (gid < size)` |

---

## 十、Part 7：CUDA 后端 — 逐元素与标量运算

### 10.1 宏模板化（CUDA 版）

和 CPU 一样用宏，但有两个关键区别：
1. 每个宏同时生成 `__global__` kernel 和 host 包装函数
2. 宏内用 `name##Kernel` 拼接生成 kernel 函数名

```cuda
#define CUDA_EWISE_OP(name, op) \
  __global__ void name##Kernel(const scalar_t* a, const scalar_t* b, \
                                scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid < size) out[gid] = op; \
  } \
  void name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
    CudaDims dim = CudaOneDim(out->size); \
    name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
  }
```

`CUDA_SCALAR_OP` 和 `CUDA_SINGLE_OP` 同理，分别用于标量运算和一元运算。

### 10.2 CUDA 数学函数

CUDA kernel 中不能使用 `std::` 命名空间的数学函数，需要用 CUDA 内置的版本：

| CPU (`std::`) | CUDA | 说明 |
|---------------|------|------|
| `std::max(a, b)` | `max(a, b)` 或 `fmaxf(a, b)` | CUDA 内置 `max` 对 float 有效 |
| `std::pow(a, b)` | `pow(a, b)` 或 `powf(a, b)` | CUDA 的 `pow` 会自动选择精度 |
| `std::log(a)` | `log(a)` 或 `logf(a)` | 同上 |
| `std::exp(a)` | `exp(a)` 或 `expf(a)` | 同上 |
| `std::tanh(a)` | `tanh(a)` 或 `tanhf(a)` | 同上 |

实际上在 CUDA kernel 中直接写 `log`、`exp`、`tanh`、`pow`、`max` 就能正常工作。

---

## 十一、Part 8：CUDA 后端 — Reduce 操作

作业明确说了 "for simplicity you can perform each reduction in a single CUDA thread"，所以采用最简单的方案：

```cuda
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out,
                                 size_t reduce_size, size_t out_size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < out_size) {
        size_t base = gid * reduce_size;
        scalar_t max_val = a[base];
        for (size_t i = 1; i < reduce_size; ++i) {
            max_val = max(max_val, a[base + i]);
        }
        out[gid] = max_val;
    }
}
```

**并行粒度**：每个线程负责一组 `reduce_size` 个元素的归约。线程总数 = `out_size`（归约后的元素数），不是 `a.size`。所以 host 端用 `CudaOneDim(out->size)` 而不是 `CudaOneDim(a.size)`。

**更高效的方案**（本作业不要求）：使用 warp-level primitives（`__shfl_down_sync`）或 shared memory 做树形归约，可以让一个 block 内的多个线程协作完成一组归约。

---

## 十二、Part 9：CUDA 后端 — 矩阵乘法

### 12.1 朴素方案：2D Grid 并行

每个线程计算输出矩阵的一个元素 `out[i][j]`：

```cuda
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                              uint32_t M, uint32_t N, uint32_t P) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  // 行
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;  // 列

    if (i < M && j < P) {
        scalar_t sum = 0.0f;
        for (size_t k = 0; k < N; ++k) {
            sum += a[i * N + k] * b[k * P + j];
        }
        out[i * P + j] = sum;
    }
}
```

Host 端使用 2D 的 block 和 grid：

```cuda
dim3 block(16, 16, 1);                          // 16×16 = 256 线程/block
dim3 grid((M + 15) / 16, (P + 15) / 16, 1);    // 向上取整
MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
```

### 12.2 与 CPU Matmul 的对比

| 维度 | CPU 朴素 | CPU Tiled | CUDA 朴素 |
|------|---------|-----------|-----------|
| 并行度 | 无 | 无（但 SIMD 向量化） | M×P 个线程并行 |
| 循环 | 三重 (i,k,j) | 三重 tile + AlignedDot | 每线程一重 (k) |
| 内存优化 | ikj 顺序利用缓存 | 分块 + 对齐 + restrict | 无（全局内存） |
| TILE | 8 | 8 | 不使用 tiling |

### 12.3 Shared Memory Tiling 优化（进阶，本作业不强制要求）

朴素方案的瓶颈是全局内存带宽：每个线程独立读取 A 的一行和 B 的一列，大量重复读取。

优化思路：
1. 每个 block 负责输出矩阵的一个 TILE×TILE 块
2. block 内的线程**协作**将 A 和 B 的 tile 加载到 shared memory
3. 从 shared memory 计算，减少全局内存访问

```
for each tile along k dimension:
    cooperative_fetch(A_tile → shared_A)
    cooperative_fetch(B_tile → shared_B)
    __syncthreads()
    partial_sum += shared_A[row][k] * shared_B[k][col]
    __syncthreads()
out[row][col] = partial_sum
```

---

## 十三、构建系统与环境配置

### 13.1 CMake 构建

```bash
mkdir build && cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 ..
make -j$(nproc)
```

输出的 `.so` 文件直接放到 `python/needle/backend_ndarray/` 目录下。

**CMP0146 兼容性问题**：CMake < 3.27 不认识 `CMP0146` policy，需要改为条件设置：

```cmake
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
```

**CUDA 版本兼容性**：系统自带的 CUDA 11.5 的 nvcc 与 GCC 11 的 `std_function.h` 不兼容，需要指定 CUDA 12.6 的 toolkit 路径。

### 13.2 测试命令

```bash
# 需要设置 PYTHONPATH
export PYTHONPATH=./python

# 按 Part 测试
pytest -v -k "(permute or reshape or broadcast or getitem) and cpu"   # Part 1
pytest -v -k "(compact or setitem) and cpu"                           # Part 2
pytest -v -k "(ewise or scalar) and cpu"                              # Part 3
pytest -v -k "reduce and cpu"                                         # Part 4
pytest -v -k "matmul and cpu"                                         # Part 5
pytest -v -k "(compact or setitem) and cuda"                          # Part 6
pytest -v -k "(ewise or scalar) and cuda"                             # Part 7
pytest -v -k "reduce and cuda"                                        # Part 8
pytest -v -k "matmul and cuda"                                        # Part 9

# 全量测试
pytest -v tests/hw3/test_ndarray.py
```

---

## 十四、踩坑记录与常见 Bug

### 14.1 Python 层

| Bug | 原因 | 修复 |
|-----|------|------|
| `SyntaxError: invalid syntax` | `NDArray.make()` 调用中参数之间缺逗号 | 检查每个参数后面的逗号 |
| `self.offset` 报 AttributeError | NDArray 的内部属性带下划线：`self._offset` | 统一使用 `self._offset` |
| `__getitem__` 重复处理索引 | stub 代码已将 idxs 转为 slices，不应再处理 idxs | 直接使用 `slices` 变量 |
| `broadcast_to` 末尾多余的 `]` | 复制粘贴时带入了多余字符 | 删除多余的 `]` |

### 14.2 CPU 后端

| Bug | 原因 | 修复 |
|-----|------|------|
| `ReduceSum` 中用了 `max_val` | 从 `ReduceMax` 复制代码后忘改变量名 | 改为 `sum_val` |
| `AlignedDot` 中 `for (k = 0; < TILE)` | 循环条件缺少变量 `k` | 改为 `k < TILE` |
| 宏中 `AlignedArray$` | `$` 应为 `&`（引用符号打错） | 改为 `AlignedArray&` |
| pybind11 绑定被注释 | 初始代码中绑定是注释状态 | 取消注释对应的 `m.def(...)` |

### 14.3 CUDA 后端

| Bug | 原因 | 修复 |
|-----|------|------|
| `EwiseAdd` / `ScalarAdd` 重复定义 | 原始代码已有手写版本，宏又生成了一遍 | 删除宏中重复的 `CUDA_EWISE_OP(EwiseAdd, ...)` 和 `CUDA_SCALAR_OP(ScalarAdd, ...)` |
| `__dvice__` 拼写错误 | `__device__` 少了 `e` | 改为 `__device__` |
| `tmp` vs `temp` 变量名不一致 | 声明为 `temp`，使用时写成 `tmp` | 统一变量名 |
| `&` 误用为 `%` | 位与运算 `&` 不等于取模 `%` | 改为 `%` |
| CUDA 11.5 + GCC 11 编译失败 | nvcc 与 `std_function.h` 不兼容 | 使用 CUDA 12.6 的 toolkit |

### 14.4 通用经验

1. **复制粘贴是 bug 之源**：ReduceSum 从 ReduceMax 复制后忘改变量名，CUDA 宏从手写函数复制后重复定义。每次复制后务必逐行检查。
2. **编译错误要看完整输出**：`make` 失败时 `tail -5` 可能看不到真正的错误信息，需要看完整输出。
3. **CUDA kernel 的边界检查**：每个 kernel 开头必须 `if (gid < size)`，否则会越界访问 GPU 内存，可能导致静默错误或 crash。
4. **pybind11 绑定**：每实现一个 C++ 函数，都要在 `PYBIND11_MODULE` 中注册，否则 Python 端会报 `AttributeError`。
5. **修改 C++/CUDA 代码后必须重新编译**：`cd build && make`，否则 Python 加载的还是旧的 `.so`。

---

## 十五、HW3 在整个课程中的位置

```
HW0: 纯 numpy 手写前向/反向传播
 ↓
HW1: 实现自动微分框架（计算图 + 链式法则）
 ↓
HW2: 实现 NN 模块（Linear, ReLU, BatchNorm, Optimizer 等）
 ↓
HW3: 实现自有的 NDArray 后端（CPU C++ + CUDA）  ← 你在这里
 ↓
HW4: 在 NDArray 后端上实现卷积、RNN 等高级算子
```

HW3 是框架从"能跑"到"跑得快"的关键转折点。之前所有计算都依赖 numpy，现在有了自己的 C++/CUDA 后端，后续可以：
- 在 GPU 上训练模型
- 实现更底层的优化（tiling、shared memory、向量化）
- 理解 PyTorch/TensorFlow 等框架底层是如何工作的

