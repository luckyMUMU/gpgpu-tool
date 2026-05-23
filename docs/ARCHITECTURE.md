# wgpu-compute-engine 技术架构文档

> **版本**: 0.1.0 | **Rust Edition**: 2021 | **wgpu**: v24 | **生成日期**: 2026-05-22

---

## 1. 项目概述

wgpu-compute-engine 是一个基于 wgpu 的跨平台 GPU 通用计算引擎，为 CPU 密集型任务提供 GPU 并行加速能力。项目采用**能力层与业务层分离**的两层架构，能力层封装 wgpu 底层细节，业务层实现具体算法。

### 1.1 核心特性

| 特性 | 说明 |
|------|------|
| **跨平台** | 支持 Vulkan、Metal、DX12、WebGPU 后端，自动选择高性能适配器 |
| **管线缓存** | 基于 fxhash 的着色器编译缓存，避免重复编译 |
| **缓冲区池化** | 按 2 的幂次分档复用 GPU 缓冲区，减少分配开销 |
| **异步批量提交** | 将多次 CPU-GPU 同步合并为一次 queue.submit()，4-6x 加速 |
| **零拷贝流水线** | GPU 缩放→哈希直通，无需 CPU 中间缓存 |
| **BK-tree 近似搜索** | O(log N) 汉明距离最近邻搜索 |
| **算法可扩展** | 新算法只需实现 `.rs` + `.wgsl` 配对，复用能力层基础设施 |

### 1.2 技术栈

```
wgpu v24          → GPU 抽象层（Vulkan/Metal/DX12/WebGPU）
pollster v0.4     → async → sync 桥接
bytemuck v1       → 零开销类型转换（Pod/Zeroable）
thiserror v2      → 错误类型派生
log v0.4          → 结构化日志
image v0.25       → 可选：图像加载与缩放（feature-gated）
```

---

## 2. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                         业务层 (Business)                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │ Sha256Computer│  │ PerceptualHasher │  │      BkTree        │ │
│  │ (SHA-256 哈希)│  │ ┌─────┐┌──────┐ │  │ (BK-tree 最近邻)    │ │
│  │ + BatchSubmit │  │ │Mean ││Median│ │  │ + hamming_distance │ │
│  └──────┬───────┘  │ └─────┘└──────┘ │  └────────────────────┘ │
│         │          │ + gpu_resize     │                          │
│         │          │ (零拷贝缩放流水线) │                          │
│         │          └────────┬─────────┘                          │
│         │                   │                                    │
│         │    hash_common: PerceptualHashComputer trait + macros  │
└─────────┼───────────────────┼────────────────────────────────────┘
          │                   │
┌─────────┼───────────────────┼────────────────────────────────────┐
│         ▼                   ▼                                    │
│                     能力层 (Capability Layer)                      │
│                                                                  │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐              │
│  │ GpuContext │  │GpuBuffer │  │  ComputePipeline  │              │
│  │ ·device    │  │ ·upload  │  │  ·shader compile  │              │
│  │ ·queue     │  │ ·download│  │  ·bind group      │              │
│  │ ·pipeline  │  │ ·write   │  │  ·dispatch        │              │
│  │   cache    │  └──────────┘  └──────────────────┘              │
│  └────────────┘  ┌──────────┐                                    │
│  ┌────────────┐  │BufferPool│                                    │
│  │GpuBatch    │  │ ·acquire │                                    │
│  │Submitter   │  │ ·release │                                    │
│  │ ·submit    │  └──────────┘                                    │
│  │ ·wait_all  │  ┌──────────┐                                    │
│  └────────────┘  │ GpuError │                                    │
│                  │ (9 variants)                                   │
│                  └──────────┘                                    │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  wgpu (GPU 硬件抽象)   │
              │  Vulkan / Metal / DX12 │
              └───────────────────────┘
```

### 2.1 分层职责

| 层级 | 职责 | 模块 |
|------|------|------|
| **能力层** | wgpu 封装：设备管理、缓冲区传输、管线编译与缓存、批量提交 | `context`, `buffer`, `buffer_pool`, `pipeline`, `batch`, `error` |
| **业务层** | 算法实现：SHA-256、感知哈希（6 种）、BK-tree 近似搜索、GPU 缩放 | `tasks/sha256`, `tasks/phasher`, `tasks/bktree`, `tasks/gpu_resize`, `tasks/hash_common` |

---

## 3. 能力层详细设计

### 3.1 GpuContext — GPU 上下文

**文件**: `src/context.rs` (170 行)

**职责**: 封装 wgpu 的 Instance、Adapter、Device、Queue，提供管线缓存。

**核心结构**:

```
GpuContext
├── _instance: Instance          # wgpu 实例（生命周期持有）
├── adapter: Adapter             # GPU 适配器（HighPerformance 偏好）
├── device: Device               # GPU 设备
├── queue: Queue                 # 命令队列
├── limits: Limits               # 硬件限制
└── pipeline_cache: PipelineCache # 管线缓存
    └── entries: HashMap<(u64, [u32;3]), Arc<ComputePipeline>>
```

**关键设计决策**:

1. **双创建模式**: `new()` (async) + `new_sync()` (pollster 阻塞)
2. **fxhash 管线缓存**: 使用自定义 64-bit fxhash（非 std hash）对 WGSL 源码哈希，以 `(hash, workgroup_size)` 为键缓存 `Arc<ComputePipeline>`
3. **Arc 共享**: 管线通过 `Arc` 共享，业务层可持有引用而无需 borrow

**fxhash 实现**:
```rust
fn fxhash(s: &str) -> u64 {
    let mut hash: u64 = 0x517cc1b727220a95;
    for byte in s.bytes() {
        hash = hash.rotate_left(8) ^ (byte as u64);
        hash = hash.wrapping_mul(0x517cc1b727220a95);
    }
    hash
}
```

**公开 API**:
| 方法 | 说明 |
|------|------|
| `new()` | 异步创建，自动选择高性能适配器 |
| `new_sync()` | 同步创建（pollster 阻塞） |
| `adapter_info()` | 返回适配器名称 + 后端类型 |
| `limits()` | 返回硬件限制 |
| `get_or_create_pipeline()` | 从缓存获取或编译管线 |
| `clear_pipeline_cache()` | 清空管线缓存 |
| `device()` / `queue()` | 获取内部引用 |

### 3.2 GpuBuffer — GPU 缓冲区

**文件**: `src/buffer.rs` (197 行)

**职责**: 封装 wgpu Buffer，提供 CPU-GPU 双向数据传输。

**核心结构**:
```
GpuBuffer
├── buffer: Buffer    # 底层 wgpu 缓冲区
└── size: u64         # 字节大小
```

**创建方式**:

| 方法 | 用途 |
|------|------|
| `from_data<T: Pod>(device, data, usage)` | 从类型化 CPU 数据创建 |
| `from_bytes(device, data, usage)` | 从原始字节创建 |
| `empty(device, size, usage)` | 创建空缓冲区（用于输出） |
| `from_raw(buffer, size)` | 从原始 wgpu Buffer 创建（BufferPool / 内部使用） |

**数据传输**:

| 方法 | 方向 | 说明 |
|------|------|------|
| `write<T>(queue, offset, data)` | CPU → GPU | 类型化写入 |
| `write_bytes(queue, offset, data)` | CPU → GPU | 原始字节写入 |
| `download(device, queue)` | GPU → CPU | staging buffer + map_async 同步下载 |
| `download_with_pool(device, queue, pool)` | GPU → CPU | 使用 BufferPool 暂存池下载 |

**BufferUsage 枚举**:
- `Storage`: 可读写存储缓冲区 → `STORAGE | COPY_DST`
- `Uniform`: 只读 uniform 缓冲区 → `UNIFORM | COPY_DST`

### 3.3 BufferPool — 缓冲区池

**文件**: `src/buffer_pool.rs` (168 行)

**职责**: 按尺寸分档复用 GPU 缓冲区，减少重复分配开销。

**核心结构**:
```
BufferPool
├── pools: RefCell<HashMap<PoolKey, Vec<Buffer>>>  # 按(档位,用途)分组
├── staging_pools: RefCell<HashMap<u64, Vec<Buffer>>> # 暂存池（MAP_READ）
└── max_per_class: usize = 8                        # 每档最大缓存数
```

**分档策略 (size_class)**:

```
尺寸范围          → 档位大小
1 - 256 bytes    → 256
257 - 512        → 512
513 - 1024       → 1024
1025 - 2048      → 2048
...              → 2^n (向上取整)
> 1GB            → 原始尺寸（避免溢出）
```

**获取流程 (acquire)**:
1. 计算请求尺寸对应的档位
2. 尝试从 `(档位, usage)` 组合获取空闲 Buffer
3. 若无，创建新 Buffer（按档位大小分配）

**归还流程 (release)**:
1. 计算 Buffer 的档位
2. 若该档未满（< max_per_class），放入空闲队列
3. 否则丢弃（由 wgpu 回收）
4. 大于 1MB 的缓冲区直接销毁，避免占用显存

**设计原则**:
- 使用 `PoolKey { size_class, usage }` 结构体作为 HashMap 键，区分 Storage/Uniform 用途
- 独立 `staging_pools` 管理 MAP_READ 暂存缓冲区的复用
- `RefCell` 实现内部可变性，`acquire`/`release` 无需 `&mut self`
- 缓冲区从池中取出后归调用者所有

### 3.4 ComputePipeline — 计算管线

**文件**: `src/pipeline.rs` (197 行)

**职责**: 封装 wgpu ComputePipeline，负责 shader 编译、bind group 构建和 dispatch 执行。

**核心结构**:
```
ComputePipeline
├── pipeline: WgpuComputePipeline     # 底层计算管线
├── bind_group_layout: BindGroupLayout # 绑定组布局
└── workgroup_size: [u32; 3]          # 工作组大小
```

**Bind Group Layout** (固定 3 个绑定):

| Binding | 类型 | 可见性 | 用途 |
|---------|------|--------|------|
| 0 | Storage (read_only) | COMPUTE | 输入数据 |
| 1 | Storage (read_write) | COMPUTE | 输出数据 |
| 2 | Uniform | COMPUTE | 参数传递 |

**Dispatch 方法**:

| 方法 | 行为 | 使用场景 |
|------|------|----------|
| `dispatch()` | 创建 encoder → 编码 → 提交 queue | 单次同步执行 |
| `encode_dispatch_into()` | 编码到已有 encoder，不提交 | 批量提交场景 |

### 3.5 GpuBatchSubmitter — 异步批量提交器

**文件**: `src/batch.rs` (174 行)

**职责**: 将多个计算任务编码到同一个 CommandEncoder，通过一次 `queue.submit()` 统一提交。

**核心结构**:
```
GpuBatchSubmitter
├── encoder: Option<CommandEncoder>  # 惰性创建的共享 encoder
├── pending: Vec<PendingJob>         # 已提交未完成的 job
├── device: Option<Device>           # 提交时捕获
└── queue: Option<Queue>             # 提交时捕获
```

**BatchJob 结构** (业务层构建):
```
BatchJob
├── input: GpuBuffer       # 输入数据
├── output: GpuBuffer      # 输出数据
├── params: GpuBuffer      # Uniform 参数
├── pipeline: Arc<ComputePipeline> # 计算管线
└── dispatch: [u32; 3]     # Dispatch 尺寸
```

**工作流程**:
```
submit(job) → 编码 dispatch + copy 到共享 encoder → 记录 PendingJob
submit(job) → 继续编码到同一 encoder
...
wait_all()  → queue.submit(encoder) → device.poll(Wait)
            → 逐个 map_async 读取 staging buffer → 返回结果
```

**设计原则**:
- **算法无关**: 通过 `BatchJob` 描述任意计算任务
- **真批量提交**: 所有 dispatch 编码到一个 encoder，一次 submit
- **延迟下载**: 所有结果延迟到 `wait_all()` 统一执行
- **资源安全**: `wait_all()` 消费所有 pending jobs，无泄漏

**性能数据** (Windows, wgpu Vulkan):

| 场景 | 同步 compute | 异步 batch | 加速比 |
|------|------------|-----------|--------|
| 10 次单条消息 | 16.4ms | 3.8ms | **4.3x** |
| 100 次单条消息 | 164.4ms | 26.2ms | **6.3x** |
| 10 次 100 条批量 | 17.3ms | 3.9ms | **4.4x** |

### 3.6 GpuError — 统一错误类型

**文件**: `src/error.rs` (32 行)

**职责**: 覆盖 GPU 计算全生命周期的错误场景。

```rust
pub enum GpuError {
    NoAdapter,                          // 未找到 GPU 适配器
    DeviceRequest(String),              // 设备请求失败
    ShaderCompile(String),              // 着色器编译失败
    MapFailed(String),                  // 缓冲区映射失败
    Validation(String),                 // GPU 验证错误
    DeviceLost,                         // 设备丢失
    Oom { requested, limit },           // 显存不足
    Internal(String),                   // 内部错误
    InvalidInput(String),               // 无效输入
}
```

使用 `thiserror` 派生，所有变体自带 `Display` 实现。

---

## 4. 业务层详细设计

### 4.1 Sha256Computer — SHA-256 GPU 并行哈希

**文件**: `src/tasks/sha256.rs` (654 行) + `src/tasks/sha256.wgsl` (124 行)

**职责**: 批量计算 SHA-256 哈希，支持单 block 和多 block 两种模式。

**核心结构**:
```
Sha256Computer
├── pipeline: Arc<ComputePipeline>       # 编译后的管线
├── workgroup_size: [u32; 3] = [256,1,1] # 工作组大小
├── buffer_pool: BufferPool              # 缓冲区复用
├── cached_single_block_params: RefCell<HashMap<u32, GpuBuffer>>
└── cached_multi_block_params: RefCell<Option<GpuBuffer>>
```

**双模式设计**:

| 模式 | 条件 | 策略 |
|------|------|------|
| **Single Block** | 消息 ≤ 55 字节 | 所有消息一次 dispatch 并行处理 |
| **Multi Block** | 消息 > 55 字节 | 逐 block 链式处理（数据依赖） |

**Single Block 流程**:
```
输入消息 → PKCS#7 填充到 64 字节 → 打包为 u32 数组
→ 写入 input buffer → dispatch (N messages / 256 workgroups)
→ 下载 output buffer → 解析为 [u8; 32] 数组
```

**Multi Block 流程**:
```
输入消息 → 分割为 64 字节 blocks
→ 对每个 block: (中间哈希 + block) → dispatch → 新中间哈希
→ 最后一个 block 的结果即为最终哈希
```

**Sha256BatchSubmitter** (内部):
- `submit(messages)` → 分类单/多 block → 编码到共享 encoder
- `wait_all()` → 一次提交 → 统一读取结果

### 4.2 PerceptualHasher — 感知图像哈希

**文件**: `src/tasks/phasher.rs` (197 行)

**职责**: 统一的感知图像哈希入口，支持 6 种算法 + GPU 加速缩放。

**核心结构**:
```
PerceptualHasher
├── algorithm: HashAlgorithm              # 算法类型
├── target_width: u32                     # 目标宽度
├── target_height: u32                    # 目标高度
├── computer: Box<dyn PerceptualHashComputer> # 多态计算器
└── gpu_resize: Option<GpuResize>         # GPU 缩放器（可启用）
```

**支持的算法**:

| 算法 | 目标尺寸 | 适用场景 |
|------|---------|---------|
| Mean | 8×8 | 基础相似度比较 |
| Median | 8×8 | 对极端值更鲁棒 |
| Gradient | 8×9 | 边缘敏感 |
| Block | 16×16 | 局部特征保留 |
| VertGradient | 9×8 | 垂直边缘敏感 |
| DoubleGradient | 9×9 | 双向边缘敏感 |

**工作流程** (默认 CPU 缩放路径):
```
输入图像 (任意尺寸灰度像素)
→ resize_grayscale() (CPU 盒式滤波下采样)
→ computer.compute() (GPU 计算哈希)
→ 返回 Vec<u64> 哈希值
```

**零拷贝 GPU 缩放流程** (`with_resize_mode(true)`):
```
输入图像 (任意尺寸灰度像素)
→ gpu_resize.resize_batch_gpu() (GPU box filter 缩放, 结果留在显存)
→ compute_phash_from_gpu_buffer() (直接使用 GPU buffer 计算哈希, 无 CPU 中转)
→ 返回 Vec<u64> 哈希值
```

**分批处理**: 当单批数据超出 GPU 缓冲区大小限制时，自动拆分为多个子批次，
每个子批次独立完成 GPU 零拷贝流水线，合并结果。

### 4.3 hash_common — 感知哈希共享基础设施

**文件**: `src/tasks/hash_common.rs` (202 行)

**职责**: 提供感知哈希的通用 trait、公共计算流程和宏。

**核心组件**:

1. **PhashParams** — Uniform 参数结构体
```rust
#[repr(C)]
pub struct PhashParams {
    pub image_count: u32,  // WGSL params.x
    pub width: u32,        // WGSL params.y
    pub height: u32,       // WGSL params.z
    pub _padding: u32,     // WGSL params.w
}
```

2. **PerceptualHashComputer trait** — 统一接口（新增零拷贝支持）
```rust
pub trait PerceptualHashComputer {
    fn compute(&self, ctx: &GpuContext, images: &[Vec<u8>])
        -> Result<Vec<u64>, GpuError>;
    fn pipeline(&self) -> &Arc<ComputePipeline>;        // 零拷贝需要
    fn workgroup_size(&self) -> [u32; 3];               // 零拷贝需要
}
```

3. **compute_phash()** — 通用 GPU 计算流程
```
像素打包 → GpuBuffer 创建 → dispatch → 下载 → 解析 u64
```

4. **compute_phash_from_gpu_buffer()** — 零拷贝入口
```
接收 GPU buffer → 创建输出 buffer → dispatch → 下载 → 解析 u64
(输入数据已在 GPU 显存中，无需 CPU 中转)
```

5. **宏系统** — 消除 6 个算法的重复代码

### 4.4 算法模块 — 薄封装模式

每个感知哈希算法仅 **14 行代码**，通过宏实现完整功能：

```rust
// mean_hash.rs (14 行)
const MEAN_HASH_WGSL: &str = include_str!("mean_hash.wgsl");

declare_phash_computer!(
    MeanHashComputer,
    "Mean Hash（均值哈希）GPU 计算器...",
    MEAN_HASH_WGSL,
    [256, 1, 1]
);
impl_phash_computer_simple!(MeanHashComputer);
```

### 4.5 GpuResize — GPU 图像缩放

**文件**: `src/tasks/gpu_resize.rs` + `src/tasks/resize.wgsl`

**职责**: 使用 GPU compute shader 进行图像 box filter 缩放，支持零拷贝流水线。

**核心结构**:
```
GpuResize
├── pipeline: Arc<ComputePipeline>  # 缩放管线
├── workgroup_size: [u32; 3]       # 工作组大小
└── buffer_pool: BufferPool         # 缓冲区复用
```

**批量和零拷贝**:
- `resize_batch()`: 批量缩放 + 返回 CPU Vec<u8>（回退路径）
- `resize_batch_gpu()`: 批量缩放 + 返回 GpuBuffer（零拷贝路径）

**WGSL Shader** (`resize.wgsl`):
```wgsl
@group(0) @binding(0) var<storage, read> src_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_pixels: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

// Box filter: 每个 work item 处理一张完整图像
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 计算源像素区域
    // 对目标区域内每个像素，求源区域对应块的像素平均值
}
```

**分批策略**: 根据图像数量、输入输出尺寸计算单批是否超过 `256MB` 缓冲区限制，
超限时自动分批。

### 4.6 BkTree — BK-tree 近似最近邻搜索

**文件**: `src/tasks/bktree.rs` (120 行)

**职责**: 基于汉明距离（Hamming Distance）的度量树，
为感知哈希提供 O(log N) 近似最近邻搜索。

**核心结构**:
```
BkTree
├── root: Option<BkNode>  # 根节点
└── len: usize            # 元素数量
```

**核心算法**:

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `insert(hash)` | O(log N) | 按汉明距离插入子树 |
| `find(hash, threshold)` | O(log N) | 三角不等式剪枝 |
| `find_nearest(hash)` | O(log N) | 自适应剪枝搜索 |

**性能数据** (10 万条哈希):

| 操作 | BK-tree | 暴力搜索 | 加速比 |
|------|---------|---------|--------|
| 查找最近邻 | ~50ns | ~312µs | **~6240x** |
| 阈值搜索 (threshold=5) | ~150ns | ~312µs | **~2080x** |
| 构建 | ~97ms | — | — |

**使用场景**: 结合 `PerceptualHasher` 的输出，实现近似图像检索：
```
感知哈希 → BK-tree 索引 → 阈值搜索返回相似图像
```

### 4.7 算法一览

| 文件 | WGSL | 行数 | 说明 |
|------|------|------|------|
| `sha256.rs` | `sha256.wgsl` | 654 + 124 | SHA-256 并行哈希（参考实现） |
| `phasher.rs` | — | ~200 | 感知哈希编排器 + GPU 缩放集成 |
| `hash_common.rs` | — | ~200 | 共享 trait + 宏 + 零拷贝入口 |
| `gpu_resize.rs` | `resize.wgsl` | ~250 | GPU box filter 缩放 |
| `bktree.rs` | — | ~120 | BK-tree 近似搜索 |
| `mean_hash.rs` | `mean_hash.wgsl` | 14 + ~40 | 均值哈希 |
| `median_hash.rs` | `median_hash.wgsl` | 14 + ~40 | 中值哈希 |
| `gradient_hash.rs` | `gradient_hash.wgsl` | 14 + ~40 | 梯度哈希 |
| `block_hash.rs` | `block_hash.wgsl` | 14 + ~40 | 分块哈希 |
| `vert_gradient_hash.rs` | `vert_gradient_hash.wgsl` | 14 + ~40 | 垂直梯度哈希 |
| `double_gradient_hash.rs` | `double_gradient_hash.wgsl` | 14 + ~40 | 双向梯度哈希 |

---

## 5. 数据流与交互

### 5.1 单次同步计算流程

```
用户调用 Sha256Computer::compute()
    │
    ├── 分类消息: single_block vs multi_block
    │
    ├── Single Block 路径:
    │   ├── 填充 + 打包 → all_words (Vec<u32>)
    │   ├── BufferPool.acquire() → input_buffer, output_buffer
    │   ├── queue.write_buffer() → 写入输入
    │   ├── get_or_create_single_block_params() → params_buffer
    │   ├── pipeline.dispatch() → GPU 执行
    │   ├── output_buffer.download() → 读取结果
    │   └── BufferPool.release() → 归还缓冲区
    │
    ├── Multi Block 路径:
    │   └── 逐 block 循环: dispatch → download → 更新中间哈希
    │
    └── 合并结果 → Vec<[u8; 32]>
```

### 5.2 零拷贝感知哈希流水线

```
PerceptualHasher::compute()  (gpu_resize 启用时)
    │
    ├── 计算分批：根据缓冲区限制拆分子批次
    │
    └── 每个子批次:
        ├── memcpy 打包输入数据到 u32 数组
        ├── BufferPool.acquire() → input_buffer
        ├── queue.write_buffer() → 上传到 GPU
        ├── GpuResize 管线 dispatch → 缩放结果留在 output_buffer
        ├── BufferPool.release(input_buffer)
        │
        ├── compute_phash_from_gpu_buffer():
        │   ├── BufferPool.acquire() → hash_output_buffer
        │   ├── hash 管线 dispatch → 哈希结果写入 hash_output_buffer
        │   ├── hash_output_buffer.download() → 读取 u64 哈希
        │   └── 释放缓冲区
        │
        └── 合并子批次结果 → Vec<u64>
```

### 5.3 异步批量提交流程

```
用户创建 GpuBatchSubmitter
    │
    ├── submit(job_1) → 编码到 encoder (不提交)
    ├── submit(job_2) → 继续编码到同一 encoder
    ├── submit(job_3) → 继续编码到同一 encoder
    │
    └── wait_all()
        ├── queue.submit(encoder.finish()) → 一次提交
        ├── device.poll(Wait) → 等待 GPU 完成
        ├── 逐个 map_async → 读取 staging buffer
        ├── 解析结果
        └── BufferPool.release() → 归还缓冲区
```

### 5.4 管线缓存流程

```
业务层调用 ctx.get_or_create_pipeline(wgsl, workgroup_size)
    │
    ├── fxhash(wgsl_source) → u64 hash
    ├── key = (hash, workgroup_size)
    │
    ├── 缓存命中 → 返回 Arc<ComputePipeline> (无编译)
    │
    └── 缓存未命中:
        ├── ComputePipeline::create() → 编译 shader
        ├── 创建 bind group layout (3 bindings)
        ├── 创建 pipeline layout
        ├── 创建 compute pipeline
        └── 插入缓存 → 返回 Arc
```

---

## 6. 模块依赖关系

```
lib.rs
├── batch          → buffer, error, pipeline, context
├── buffer         → error, buffer_pool
├── buffer_pool    → buffer
├── context        → error, pipeline
├── error          → (无依赖)
├── pipeline       → buffer, error
└── tasks
    ├── mod.rs     → (导出子模块)
    ├── sha256     → buffer, buffer_pool, context, error, pipeline
    ├── phasher    → context, error, hash_common, hash_algorithms, gpu_resize
    ├── hash_common → buffer, context, error, pipeline
    ├── gpu_resize → buffer, buffer_pool, context, error, pipeline
    ├── bktree     → (无依赖, 纯算法)
    ├── mean_hash  → hash_common (macros)
    ├── median_hash → hash_common (macros)
    ├── gradient_hash → hash_common (macros)
    ├── block_hash → hash_common (macros)
    ├── vert_gradient_hash → hash_common (macros)
    └── double_gradient_hash → hash_common (macros)
```

**依赖方向**: 能力层内部单向依赖，业务层依赖能力层，算法间无横向依赖。

---

## 7. 扩展指南

### 7.1 添加新算法

遵循 SHA-256 模式（完整实现）或感知哈希模式（薄封装）：

**模式 A: 完整算法（如 SHA-256）**
```
1. 创建 src/tasks/new_algo.rs + src/tasks/new_algo.wgsl
2. 实现 NewAlgoComputer 结构体，持有 Arc<ComputePipeline>
3. 实现 compute() 方法：打包 → buffer → dispatch → download
4. 可选：实现 batch_submitter() 支持异步批量
5. 在 src/tasks/mod.rs 中导出
```

**模式 B: 感知哈希算法**
```
1. 创建 src/tasks/new_hash.rs + src/tasks/new_hash.wgsl
2. 使用 declare_phash_computer! + impl_phash_computer_simple! 宏
3. 在 HashAlgorithm 枚举中添加新变体
4. 在 phasher.rs 的 new() 中添加匹配分支
5. 在 src/tasks/mod.rs 中导出
```

### 7.2 WGSL Shader 规范

```wgsl
// 固定绑定布局（与 ComputePipeline 一致）
@group(0) @binding(0) var<storage, read> input: array<T>;
@group(0) @binding(1) var<storage, read_write> output: array<T>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

// 入口函数必须为 main
@compute @workgroup_size(N)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ...
}
```

### 7.3 Workgroup 大小选择

| 场景 | 推荐大小 | 说明 |
|------|---------|------|
| 数据并行（SHA-256、感知哈希） | `[256, 1, 1]` | 每条消息/图像一个 work item |
| 需要共享内存 | `[64, 1, 1]` 或 `[16, 16, 1]` | 减少寄存器压力 |

---

## 8. 构建与测试

### 8.1 构建命令

```bash
cargo build                         # 默认构建（无 image feature）
cargo build --features image        # 启用图像支持
cargo build --release               # 发布构建
```

### 8.2 测试

```bash
cargo test                          # 运行所有测试
cargo test sha256                   # 运行特定测试
cargo test --test gpu_resize_test   # 运行 GPU 缩放测试
cargo test --test bktree_test       # 运行 BK-tree 测试
```

### 8.3 基准测试

```bash
cargo bench                         # 运行所有基准测试
cargo bench --bench sha256_bench    # 运行特定基准
cargo bench --bench large_scale_bench --features image  # 大型图像基准
```

基准测试报告生成在 `target/criterion/`。

### 8.4 示例运行

```bash
cargo run --example demo            # 运行演示
```

### 8.5 代码检查

```bash
cargo clippy                        # lint 检查
```

---

## 9. 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 管线缓存哈希算法 | fxhash (自定义) | 比 std hash 更快，无随机种子，确定性哈希 |
| 缓冲区池分档策略 | 2 的幂次 | 简单高效，覆盖常见尺寸范围 |
| 批量提交模式 | submit() + wait_all() | 非 futures，更简单的 API 表面 |
| WGSL 组织方式 | 独立 .wgsl 文件 | 语法高亮、IDE 支持、可维护性 |
| 错误处理 | thiserror 枚举 | 零运行时开销，模式匹配友好 |
| 感知哈希宏 | 声明式宏 | 6 个算法仅 84 行代码，消除重复 |
| 非泛型 GpuBuffer | bytemuck 转换 | 避免单态化膨胀，运行时类型安全 |
| Pipeline 使用 Arc | 引用计数共享 | 业务层可持有管线引用，无需 borrow |
| GPU 缩放输出格式 | 1 u32/像素 | 匹配 hash shader 输入格式，零拷贝直通 |
| 分批策略 | 调用方分批 | phasher 层控制，gpu_resize 只负责单批 |

---

## 10. 已知限制

| 限制 | 说明 |
|------|------|
| GPU 调度开销 | 单次 dispatch 约 1.6ms，小批量场景不如 CPU |
| Multi-block 数据依赖 | SHA-256 多 block 消息无法完全并行，中间 block 需同步 |
| 固定 Bind Group Layout | 所有算法共享 3-binding 布局，灵活性受限 |
| image feature 可选 | 默认不启用图像支持，需显式开启 |
| GPU 缩放质量 | box filter 适合下采样，上采样质量不如 Lanczos |
| 分批粒度 | 分批依据 256MB 硬限制，未考虑实际显存容量 |

---

*文档生成完毕。基于源码分析，覆盖能力层 6 个模块、业务层 10 个模块的完整架构。*