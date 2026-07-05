# GPGPU-tool 技术架构文档

> **版本**: 0.3.0 | **Rust Edition**: 2021 | **wgpu**: v24 | **生成日期**: 2026-06-16

---

## 1. 项目概述

GPGPU-tool 是一个基于 wgpu 的跨平台 GPU 通用计算引擎，为 CPU 密集型任务提供 GPU 并行加速能力。项目采用**能力层与业务层分离**的两层架构，能力层封装 wgpu 底层细节，业务层实现具体算法。当 GPU 不可用时，自动降级到 CPU 实现。

### 1.1 核心特性

| 特性 | 说明 |
|------|------|
| **跨平台** | 支持 Vulkan、Metal、DX12、WebGPU 后端，自动选择高性能适配器 |
| **管线缓存** | 基于 fxhash 的着色器编译缓存，避免重复编译 |
| **缓冲区池化** | 按 2 的幂次分档复用 GPU 缓冲区，减少分配开销 |
| **异步批量提交** | 将多次 CPU-GPU 同步合并为一次 queue.submit()，4-6x 加速 |
| **零拷贝流水线** | GPU 缩放→哈希直通，无需 CPU 中间缓存 |
| **BK-tree 近似搜索** | O(log N) 汉明距离最近邻搜索（64-bit + 变长哈希） |
| **GPU 汉明距离匹配** | 3 管线架构：距离矩阵 + 最近邻 + 大哈希最近邻，778x 加速 |
| **变长哈希支持** | HashBytes 类型支持 64-bit 到 4096-bit 哈希，统一匹配 API |
| **端到端 GPU 图像匹配** | 图像→哈希→匹配一站式 GPU 流水线 |
| **算法可扩展** | 新算法只需实现 `.rs` + `.wgsl` 配对，复用能力层基础设施 |
| **灵活 HashSize** | 支持 8/16/32/64 网格尺寸，输出 64-4096 bit 哈希 |
| **GPU 2D 卷积** | 支持 Full2D 和 Separable 两种卷积模式，3 种边界处理 |
| **二面体变换** | D4 群 8 种旋转/翻转，支持 64/256/1024/4096-bit 哈希 |
| **PDQ 哈希** | GPU DCT-II + CPU 量化，256-bit 频域感知哈希 |
| **GPU 优先，CPU 降级** | GPU 初始化失败时自动降级到 CPU 实现，调用方无感知 |
| **声明式管线构建器** | 链式声明 GPU 处理步骤（Blur→Resize→Hash），自动零拷贝传递 |
| **后端调度抽象** | BackendDispatcher trait 统一 GPU/CPU 执行路径选择 |

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
┌──────────────────────────────────────────────────────────────────────────┐
│                         业务层 (Business)                                 │
│                                                                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐        │
│  │ Sha256Computer│  │ PerceptualHasher │  │      BkTree        │        │
│  │ (SHA-256 哈希)│  │ ┌─────┐┌──────┐ │  │ (BK-tree 最近邻)    │        │
│  │ + BatchSubmit │  │ │Mean ││Median│ │  │ + hamming_distance │        │
│  └──────┬───────┘  │ └─────┘└──────┘ │  └────────────────────┘        │
│         │          │ + gpu_resize     │                                  │
│         │          │ (零拷贝缩放流水线) │                                  │
│         │          └────────┬─────────┘                                  │
│         │                   │                                            │
│         │    hash_common: PerceptualHashComputer trait + macros           │
│                                                                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐        │
│  │GpuConvolution│  │ GpuGaussianBlur  │  │  Dihedral Transforms│        │
│  │ (2D 卷积)    │  │ (高斯模糊)        │  │  (D4 群 8 变换)     │        │
│  │ Full2D/Sep   │  │ (基于可分离卷积)  │  │  8×8 / 16×16       │        │
│  └──────┬───────┘  └────────┬─────────┘  │  32×32 / 64×64     │        │
│         │                   │            └────────────────────┘        │
│                                                                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐        │
│  │ PdqHashGpu   │  │ HashMatcherFacade│  │    pixel_pack      │        │
│  │ (PDQ DCT-II) │  │ Linear/BkTree/   │  │  (u8↔u32 转换)     │        │
│  │ feature: pdq │  │ Chained + Dihedral│  │                    │        │
│  └──────────────┘  └──────────────────┘  └────────────────────┘        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              GPU Hash Matcher (3 管线架构)                     │       │
│  │  ┌──────────────────┐  ┌────────────────────────────────┐   │       │
│  │  │ GpuHashMatcher   │  │ GpuHashMatcherBytes            │   │       │
│  │  │ ·distance_matrix │  │ (变长哈希 64~4096-bit)          │   │       │
│  │  │ ·nearest_neighbor│  │ ·compute_distance_matrix       │   │       │
│  │  │ ·nearest_large   │  │ ·find_nearest_neighbors        │   │       │
│  │  └──────────────────┘  └────────────────────────────────┘   │       │
│  │  ┌──────────────────┐  ┌────────────────────────────────┐   │       │
│  │  │GpuHashMatcher    │  │GpuHashMatcherFacadeBytes       │   │       │
│  │  │  Facade (64-bit) │  │(变长哈希 GPU 匹配门面)          │   │       │
│  │  └──────────────────┘  └────────────────────────────────┘   │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              变长哈希系统 (Variable-Length Hash)               │       │
│  │  ┌────────────┐  ┌──────────────────┐  ┌────────────────┐   │       │
│  │  │ HashBytes  │  │ MatcherBytes     │  │ BkTreeBytes    │   │       │
│  │  │ (Vec<u8>)  │  │ Linear/BkTree/   │  │ (变长 BK-tree) │   │       │
│  │  │ 64~4096bit │  │ Chained/Facade   │  │                │   │       │
│  │  └────────────┘  └──────────────────┘  └────────────────┘   │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              GpuImageMatcher (端到端 GPU 图像匹配)             │       │
│  │  PerceptualHasher + GpuHashMatcherBytes                      │       │
│  │  图像 → GPU 哈希 → GPU 距离矩阵 → CPU 过滤 → 匹配结果        │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
          │                   │
┌─────────┼───────────────────┼────────────────────────────────────────────┐
│         ▼                   ▼                                            │
│                     能力层 (Capability Layer)                              │
│                                                                          │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐                      │
│  │ GpuContext │  │GpuBuffer │  │  ComputePipeline  │                      │
│  │ ·device    │  │ ·upload  │  │  ·shader compile  │                      │
│  │ ·queue     │  │ ·download│  │  ·bind group      │                      │
│  │ ·pipeline  │  │ ·write   │  │  ·dispatch        │                      │
│  │   cache    │  └──────────┘  └──────────────────┘                      │
│  └────────────┘  ┌──────────┐                                            │
│  ┌────────────┐  │BufferPool│                                            │
│  │GpuBatch    │  │ ·acquire │                                            │
│  │Submitter   │  │ ·release │                                            │
│  │ ·submit    │  └──────────┘                                            │
│  │ ·wait_all  │  ┌──────────┐                                            │
│  └────────────┘  │ GpuError │                                            │
│                  │ (11 variants)                                          │
│                  └──────────┘                                            │
│  ┌────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │  GpuPipelineBuilder│  │ BackendDispatcher    │  │  PollCounter     │  │
│  │  (声明式管线构建)    │  │ (GPU/CPU 后端调度)   │  │  (poll 计数器)    │  │
│  └────────────────────┘  └──────────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
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
| **能力层** | wgpu 封装：设备管理、缓冲区传输、管线编译与缓存、批量提交、声明式管线构建、后端调度、poll 计数 | `context`, `buffer`, `buffer_pool`, `pipeline`, `batch`, `error`, `pipeline_builder`, `backend_dispatcher`, `poll_counter` |
| **业务层** | 算法实现：SHA-256、感知哈希（6 种）、BK-tree 近似搜索、GPU 缩放、2D 卷积、高斯模糊、二面体变换、PDQ 哈希、哈希匹配（64-bit + 变长）、GPU 汉明距离匹配、端到端 GPU 图像匹配 | `tasks/sha256`, `tasks/phasher`, `tasks/bktree`, `tasks/bktree_bytes`, `tasks/gpu_resize`, `tasks/hash_common`, `tasks/convolution`, `tasks/gaussian_blur`, `tasks/dihedral`, `tasks/matcher`, `tasks/matcher_bytes`, `tasks/hash_bytes`, `tasks/gpu_matcher`, `tasks/gpu_image_matcher`, `tasks/pdq_hash` |
| **工具层** | 像素格式转换等通用工具 | `pixel_pack` |

---

## 3. 能力层详细设计

### 3.1 GpuContext — GPU 上下文

**文件**: `src/context.rs` (469 行)

**职责**: 封装 wgpu 的 Instance、Adapter、Device、Queue，提供管线缓存。

**核心结构**:

```
GpuContext
├── _instance: Instance          # wgpu 实例（生命周期持有）
├── adapter: Adapter             # GPU 适配器（HighPerformance 偏好）
├── device: Device               # GPU 设备
├── queue: Queue                 # 命令队列
├── limits: Limits               # 硬件限制
├── backend: ComputeBackend      # 当前后端（GPU/CPU）
├── compute_units: u32           # 计算单元数量
├── buffer_pool: BufferPool      # 缓冲区池
└── pipeline_cache: PipelineCache # 管线缓存
    └── entries: HashMap<(u64, [u32;3]), Arc<ComputePipeline>>
```

**关键设计决策**:

1. **双创建模式**: `new()` (async) + `new_sync()` (pollster 阻塞)
2. **fxhash 管线缓存**: 使用自定义 64-bit fxhash（非 std hash）对 WGSL 源码哈希，以 `(hash, workgroup_size)` 为键缓存 `Arc<ComputePipeline>`
3. **Arc 共享**: 管线通过 `Arc` 共享，业务层可持有引用而无需 borrow
4. **内嵌 BufferPool**: GpuContext 内部持有 BufferPool 实例，业务层通过 `ctx.buffer_pool()` 获取
5. **后端标识**: `ComputeBackend` 枚举标识当前运行在 GPU 还是 CPU 模式

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
| `buffer_pool()` | 获取缓冲区池引用 |
| `backend()` | 返回当前计算后端 |
| `compute_units()` | 返回计算单元数量 |

### 3.2 GpuBuffer — GPU 缓冲区

**文件**: `src/buffer.rs` (666 行)

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

**文件**: `src/buffer_pool.rs` (265 行)

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

**文件**: `src/pipeline.rs` (708 行)

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

**PipelineDescriptor**: 支持灵活的绑定配置，包括自定义 binding 数量、Push Constant、entry point 等。

**Dispatch 方法**:

| 方法 | 行为 | 使用场景 |
|------|------|----------|
| `dispatch()` | 创建 encoder → 编码 → 提交 queue | 单次同步执行 |
| `encode_dispatch_into()` | 编码到已有 encoder，不提交 | 批量提交场景 |

### 3.5 GpuBatchSubmitter — 异步批量提交器

**文件**: `src/batch.rs` (665 行)

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

**文件**: `src/error.rs` (38 行)

**职责**: 覆盖 GPU 计算全生命周期的错误场景，包含 GPU 和 CPU 降级两种路径的错误。

```rust
pub enum GpuError {
    NoAdapter,                          // 未找到 GPU 适配器
    DeviceRequest(String),              // 设备请求失败
    ShaderCompile(String),              // 着色器编译失败
    MapFailed(String),                  // 缓冲区映射失败
    Validation(String),                 // GPU 验证错误
    DeviceLost,                         // 设备丢失
    Oom { requested: u64, limit: u64 }, // 显存不足
    Internal(String),                   // 内部错误
    InvalidInput(String),               // 无效输入
    CpuFallback(String),                // CPU 降级执行失败
    Timeout { ms: u64 },               // GPU 计算超时
}
```

使用 `thiserror` 派生，所有 11 个变体自带 `Display` 实现。

**变体说明**:
- `CpuFallback(String)`: CPU 降级路径执行失败时的错误，保证 GPU/CPU 双路径的统一错误处理
- `Timeout { ms: u64 }`: GPU 计算超时，携带超时毫秒数，用于长时间计算场景的超时检测

### 3.7 GpuPipelineBuilder — 声明式管线构建器

**文件**: `src/pipeline_builder.rs` (218 行)

**职责**: 提供声明式 API 构建多步骤 GPU 处理流水线，自动推导零拷贝传递路径。

**核心结构**:
```
GpuPipelineBuilder
└── steps: Vec<GpuPipelineStep>   # 有序处理步骤列表
```

**管线步骤 (GpuPipelineStep)**:

| 步骤 | 说明 | 参数 |
|------|------|------|
| `Blur` | 高斯模糊预处理 | sigma, kernel_size |
| `Resize` | 图像缩放 | width, height |
| `Hash` | 感知哈希计算 | algorithm, hash_size |

**步骤顺序约束**:
- `Hash` 必须是最后一步
- `Blur` 必须在 `Resize` 之前
- 每种步骤最多出现一次

**使用示例**:
```rust
let hashes = GpuPipelineBuilder::new()
    .blur(1.0, 5)
    .resize(8, 8)
    .hash(HashAlgorithm::Mean, HashSize::default())
    .execute(&mut ctx, &images, &widths, &heights)
    .unwrap();
```

### 3.8 BackendDispatcher — GPU/CPU 后端调度

**文件**: `src/backend_dispatcher.rs` (66 行)

**职责**: 根据 `GpuContext.backend()` 自动选择 GPU 或 CPU 执行路径，替代各业务模块中散布的手动分支。

**核心 trait**:
```rust
pub trait BackendDispatcher {
    fn dispatch_gpu<F, C, R>(
        &self,
        ctx: &GpuContext,
        gpu_fn: F,
        cpu_fn: C,
    ) -> Result<R, GpuError>
    where
        F: FnOnce(&GpuContext) -> Result<R, GpuError>,
        C: FnOnce() -> Result<R, GpuError>;
}
```

**DefaultBackendDispatcher**: 默认实现，GPU 模式调用 `gpu_fn(ctx)`，CPU 模式调用 `cpu_fn()`。当处于 CPU 模式且未启用 `cpu-fallback` feature 时，返回 `GpuError::CpuFallback`。

### 3.9 PollCounter — GPU poll 计数器

**文件**: `src/poll_counter.rs` (41 行)

**职责**: 全局 `device.poll(Wait)` 调用计数器，用于量化 GPU 同步开销。

**API**:

| 函数 | 说明 |
|------|------|
| `increment()` | 递增计数器（Relaxed 排序，最小化开销） |
| `get()` | 读取当前计数 |
| `reset()` | 重置为 0 |

---

## 4. 业务层详细设计

### 4.1 Sha256Computer — SHA-256 GPU 并行哈希

**文件**: `src/tasks/sha256.rs` (748 行) + `src/tasks/sha256.wgsl` (124 行)

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

**文件**: `src/tasks/phasher.rs` (1282 行)

**职责**: 统一的感知图像哈希入口，支持 6 种算法 + GPU 加速缩放，含阈值预计算优化。

**核心结构**:
```
PerceptualHasher
├── algorithm: HashAlgorithm              # 算法类型
├── target_width: u32                     # 目标宽度
├── target_height: u32                    # 目标高度
├── computer: Box<dyn PerceptualHashComputer> # 多态计算器
├── gpu_resize: Option<GpuResize>         # GPU 缩放器（可启用）
├── hash_size: HashSize                   # 网格尺寸配置
└── max_batch_size: u64                   # 单批最大缓冲区字节数
```

**支持的算法**:

| 算法 | 目标尺寸（hash_size=8） | 目标尺寸（hash_size=16） | 适用场景 |
|------|------------------------|-------------------------|---------|
| Mean | 8×8 | 16×16 | 基础相似度比较 |
| Median | 8×8 | 16×16 | 对极端值更鲁棒 |
| Gradient | 8×9 | 16×17 | 边缘敏感 |
| Block | 8×8 | 16×16 | 局部特征保留 |
| VertGradient | 9×8 | 17×16 | 垂直边缘敏感 |
| DoubleGradient | 9×9 | 17×17 | 双向边缘敏感 |

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

**阈值预计算优化** (Mean/Median Hash):
- CPU 端预计算阈值（均值/中位数），通过 `f32::to_bits()` 编码为 u32
- 着色器端通过 `bitcast<f32>()` 解码，直接与阈值比较
- 消除 O(N²) 循环（原着色器需遍历所有像素计算均值/中位数）
- `compute_phash_with_thresholds()`: 带预计算阈值的哈希计算入口

**分批处理**: 当单批数据超出 GPU 缓冲区大小限制时，自动拆分为多个子批次，
每个子批次独立完成 GPU 零拷贝流水线，合并结果。

### 4.3 hash_common — 感知哈希共享基础设施

**文件**: `src/tasks/hash_common.rs` (740 行)

**职责**: 提供感知哈希的通用 trait、公共计算流程、宏和阈值预计算支持。

**核心组件**:

1. **PhashParams** — Push Constant 参数结构体
```rust
#[repr(C)]
pub struct PhashParams {
    pub image_count: u32,
    pub width: u32,
    pub height: u32,
    pub hash_size: u32,
}
```

2. **HashSize** — 网格尺寸配置（替代已废弃的 HashBits）
```rust
pub struct HashSize(u32);  // 存储网格边长

impl HashSize {
    pub fn new(size: u32) -> Self;       // 创建指定网格尺寸
    pub fn size(self) -> u32;            // 获取网格边长
    pub fn bits(self) -> u32;            // 哈希位宽 = size²
    pub fn u32s_per_image(self) -> u32;  // 每图 u32 数
    pub fn u64s_per_image(self) -> u32;  // 每图 u64 数
}
```

| hash_size | 网格尺寸 | 哈希位宽 |
|-----------|---------|---------|
| 8  | 8×8   | 64 bit   |
| 16 | 16×16 | 256 bit  |
| 32 | 32×32 | 1024 bit |
| 64 | 64×64 | 4096 bit |

3. **PerceptualHashComputer trait** — 统一接口
```rust
pub trait PerceptualHashComputer {
    fn compute(&self, ctx: &GpuContext, images: &[Vec<u8>])
        -> Result<Vec<u64>, GpuError>;
    fn compute_sized(&self, ctx: &GpuContext, images: &[Vec<u8>], hash_size: HashSize)
        -> Result<Vec<u64>, GpuError>;
    fn pipeline(&self) -> &ComputePipeline;
    fn workgroup_size(&self) -> [u32; 3];
    fn hash_size(&self) -> HashSize;
}
```

4. **compute_phash()** — 通用 GPU 计算流程
```
像素打包 → GpuBuffer 创建 → dispatch → 下载 → 解析 u64
```

5. **compute_phash_from_gpu_buffer()** — 零拷贝入口
```
接收 GPU buffer → 创建输出 buffer → dispatch → 下载 → 解析 u64
(输入数据已在 GPU 显存中，无需 CPU 中转)
```

6. **compute_phash_with_thresholds()** — 阈值预计算入口
```
CPU 预计算每张图像的阈值（均值/中位数）
→ f32::to_bits() 编码为 u32 打包到输入 buffer
→ dispatch → 着色器 bitcast<f32>() 解码阈值
→ 直接与阈值比较生成哈希位 → 消除 O(N²) 循环
```

7. **宏系统** — 消除 6 个算法的重复代码

8. **管线描述符工厂** — `phash_pipeline_descriptor()`: 根据设备 Push Constant 支持情况自动选择 2-binding + Push Constant 或 3-binding + Uniform 布局

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

> **注意**: 宏现在自动生成 `hash_size: HashSize` 字段和 `with_config(ctx, workgroup_size, hash_size)` 构造器。

### 4.5 GpuResize — GPU 图像缩放

**文件**: `src/tasks/gpu_resize.rs` (452 行) + `src/tasks/resize.wgsl`

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

**文件**: `src/tasks/bktree.rs` (299 行)

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

### 4.7 GpuConvolution — GPU 2D 卷积

**文件**: `src/tasks/convolution.rs` (717 行) + `src/tasks/convolution.wgsl` (~118 行)

**职责**: GPU 加速的 2D 卷积计算，支持 Full2D 和 Separable 两种模式，3 种边界处理方式。

**核心结构**:
```
GpuConvolution
├── pipeline: Arc<ComputePipeline>       # 卷积管线
├── workgroup_size: [u32; 3] = [256,1,1] # 工作组大小
└── buffer_pool: BufferPool              # 缓冲区复用
```

**卷积模式 (ConvMode)**:

| 模式 | 说明 | 计算量 |
|------|------|--------|
| `Full2D` | 不可分离 2D 卷积（单趟） | O(W×H×K²) |
| `Separable` | 可分离卷积（水平 + 垂直两趟 1D） | O(W×H×K×2) |

**边界处理 (BorderMode)**:

| 模式 | 说明 |
|------|------|
| `Zero` | 越界像素值为 0 |
| `Clamp` | 钳制到最近的边缘像素 |
| `Reflect` | 半样本对称镜像反射 |

**Uniform 参数 (ConvParams)**:
```rust
#[repr(C)]
struct ConvParams {
    width: u32,
    height: u32,
    kernel_size: u32,       // 卷积核边长（最大 11）
    kernel_radius: u32,     // kernel_size / 2
    border_mode: u32,       // 0=Zero, 1=Clamp, 2=Reflect
    pass_mode: u32,         // 0=Full2D, 1=Horizontal1D, 2=Vertical1D
    _pad1: u32,
    _pad2: u32,
    kernel: [f32; 124],     // 卷积核数据，填充至 124 满足 16 字节对齐
}
```

**Full2D 卷积流程**:
```
输入像素 → pack_u8_to_u32 → 上传到 GPU
→ dispatch (单趟 2D 卷积)
→ 下载结果 → unpack_u32_to_u8 → 返回 Vec<u8>
```

**Separable 卷积流程**:
```
输入像素 → pack_u8_to_u32 → 上传到 GPU
→ 单 encoder 编码两趟 dispatch:
  第一趟: 水平 1D 卷积 (input → intermediate)
  第二趟: 垂直 1D 卷积 (intermediate → output)
→ 一次 queue.submit() 提交两趟
→ 下载结果 → unpack_u32_to_u8 → 返回 Vec<u8>
```

**零拷贝接口**: `convolve_separable_gpu()` 接收 GpuBuffer 输入，返回 GpuBuffer 输出，
支持与下游模块（如高斯模糊、感知哈希）组成零拷贝流水线。

**卷积核限制**: 最大边长 11（11×11 = 121 个 f32），kernel 数组固定 124 个 f32。

### 4.8 GpuGaussianBlur — GPU 高斯模糊

**文件**: `src/tasks/gaussian_blur.rs` (151 行)

**职责**: 基于可分离卷积实现 GPU 高斯模糊，用于感知哈希的图像降噪预处理。

**核心结构**:
```
GpuGaussianBlur
└── convolution: GpuConvolution  # 委托给 GpuConvolution 执行
```

**设计**: GpuGaussianBlur 是 GpuConvolution 的高层封装，自动生成 1D 高斯核后调用可分离卷积。

**高斯核生成**:
- 使用高斯函数 G(x) = exp(-x²/(2σ²))，归一化后使核元素之和为 1
- sigma ≤ 0 时自动计算为 `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`（OpenCV 默认公式）
- kernel_size 必须为正奇数（3/5/7/9/11），最大 11

**公开 API**:

| 方法 | 说明 |
|------|------|
| `blur(ctx, pixels, width, height, kernel_size, sigma)` | CPU 输入 → CPU 输出 |
| `blur_gpu(ctx, input_buffer, count, width, height, kernel_size, sigma)` | GPU 输入 → GPU 输出（零拷贝） |

**边界模式**: 固定使用 `BorderMode::Clamp`（钳制到边缘），适合图像处理场景。

### 4.9 Dihedral Transforms — 二面体变换

**文件**: `src/tasks/dihedral.rs` (1044 行)

**职责**: 对哈希位矩阵执行 D4 群的 8 种旋转/翻转变换，无需重新计算图像哈希即可匹配旋转/翻转后的图像。

**8 种变换**:

| 变换 | 说明 | 数学描述 |
|------|------|----------|
| `original` | 原始 | — |
| `rotate90` | 顺时针旋转 90° | new[col][N-1-row] = old[row][col] |
| `rotate180` | 旋转 180° | new[N-1-row][N-1-col] = old[row][col] |
| `rotate270` | 顺时针旋转 270° | new[N-1-col][row] = old[row][col] |
| `flip_h` | 水平翻转（左右镜像） | new[row][N-1-col] = old[row][col] |
| `flip_v` | 垂直翻转（上下镜像） | new[N-1-row][col] = old[row][col] |
| `flip_diag` | 主对角线翻转（转置） | new[col][row] = old[row][col] |
| `flip_anti_diag` | 反对角线翻转 | new[N-1-col][N-1-row] = old[row][col] |

**支持的矩阵尺寸**:

| 类型 | 矩阵尺寸 | 哈希位宽 | 输出格式 |
|------|---------|---------|---------|
| `DihedralHashes64` | 8×8 | 64 bit | 每种变换 1 个 u64 |
| `DihedralHashes256` | 16×16 | 256 bit | 每种变换 4 个 u64 |
| `DihedralHashes1024` | 32×32 | 1024 bit | 每种变换 16 个 u64 |
| `DihedralHashes4096` | 64×64 | 4096 bit | 每种变换 64 个 u64 |

**核心结构**:
```rust
pub struct DihedralHashes64 {
    pub original: u64,
    pub rotate90: u64,
    pub rotate180: u64,
    pub rotate270: u64,
    pub flip_h: u64,
    pub flip_v: u64,
    pub flip_diag: u64,
    pub flip_anti_diag: u64,
}

pub struct DihedralHashes256 {
    pub original: [u64; 4],
    pub rotate90: [u64; 4],
    pub rotate180: [u64; 4],
    pub rotate270: [u64; 4],
    pub flip_h: [u64; 4],
    pub flip_v: [u64; 4],
    pub flip_diag: [u64; 4],
    pub flip_anti_diag: [u64; 4],
}

pub struct DihedralHashes1024 {
    pub original: [u64; 16],
    pub rotate90: [u64; 16],
    pub rotate180: [u64; 16],
    pub rotate270: [u64; 16],
    pub flip_h: [u64; 16],
    pub flip_v: [u64; 16],
    pub flip_diag: [u64; 16],
    pub flip_anti_diag: [u64; 16],
}

pub struct DihedralHashes4096 {
    pub original: [u64; 64],
    pub rotate90: [u64; 64],
    pub rotate180: [u64; 64],
    pub rotate270: [u64; 64],
    pub flip_h: [u64; 64],
    pub flip_v: [u64; 64],
    pub flip_diag: [u64; 64],
    pub flip_anti_diag: [u64; 64],
}
```

**创建方式**:
- `DihedralHashes64::from_u64(hash)`: 从 u64 哈希值推导所有变体
- `DihedralHashes64::from_bits(bits)`: 从 8×8 位矩阵推导所有变体
- `DihedralHashes256::from_u64_array(hash)`: 从 [u64; 4] 推导所有变体
- `DihedralHashes256::from_bits(bits)`: 从 16×16 位矩阵推导所有变体
- `DihedralHashes1024::from_u64_array(hash)`: 从 [u64; 16] 推导所有变体
- `DihedralHashes1024::from_bits(bits)`: 从 32×32 位矩阵推导所有变体
- `DihedralHashes4096::from_u64_array(hash)`: 从 [u64; 64] 推导所有变体
- `DihedralHashes4096::from_bits(bits)`: 从 64×64 位矩阵推导所有变体

**位打包**: LSB-first 编码，bit i 在 byte i/8 的 bit i%8 位置。

**群性质**: D4 群封闭性保证任意两种变换的组合等价于群中另一种变换（如 rotate90 + flip_h = flip_anti_diag）。

**DihedralTransform trait**: 统一四种尺寸的二面体变换接口，`all_hashes()` 返回所有 8 种变体的迭代器。

### 4.10 Hash Matcher — 哈希匹配策略

**文件**: `src/tasks/matcher.rs` (369 行)

**职责**: 提供统一的哈希匹配抽象和多种策略实现，支持策略模式组合和二面体变换增强。

**核心类型**:

```rust
pub struct MatchResult {
    pub hash: u64,
    pub distance: u32,
    pub quality: Option<f32>,
}

pub trait HashMatcher {
    fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult>;
    fn find_similar_batch(&self, queries: &[u64], threshold: u32) -> Vec<Vec<MatchResult>>;
}
```

**策略实现**:

| 匹配器 | 说明 | 召回率 | 速度 |
|--------|------|--------|------|
| `LinearScanMatcher` | 精确线性扫描 | 100% | O(N) |
| `BkTreeMatcher` | BK-tree 近似匹配 | 近似 | O(log N) |
| `ChainedMatcher` | 责任链组合多个匹配器 | 取决于策略 | 取决于组合 |
| `HashMatcherFacade` | 统一门面 + 二面体变换增强 | 取决于内部策略 | 取决于组合 |

**责任链策略 (ChainStrategy)**:

| 策略 | 说明 |
|------|------|
| `Union` | 合并所有匹配器结果（并集去重） |
| `FirstHit` | 第一个有结果的匹配器后停止 |

**ChainedMatcher 预设**: `bk_tree_plus_linear()` 创建 BK-tree + 线性扫描的 Union 责任链。

**HashMatcherFacade 门面**:
- `linear_scan(hashes)`: 创建精确线性扫描门面
- `bktree(hashes)`: 创建 BK-tree 门面
- `chained(hashes)`: 创建责任链门面
- `with_dihedral()`: 启用二面体变换匹配（对查询哈希的 8 种 D4 变体分别匹配，结果去重）

**二面体变换匹配流程**:
```
query hash → DihedralHashes64::from_u64(query) → 8 种变体
→ 对每种变体调用内部 matcher.find_similar()
→ 合并去重结果 → 返回 Vec<MatchResult>
```

### 4.11 PDQ Hash — PDQ 感知哈希

**文件**: `src/tasks/pdq_hash.rs` (315 行) + `src/tasks/pdq_hash.wgsl` (~60 行)

**Feature flag**: `pdq`（需显式启用）

**职责**: 基于 DCT-II 频域变换的感知哈希算法（Meta/Facebook PDQ），输入 64×64 灰度图像，输出 256-bit 哈希 + 质量评分。

**GPU/CPU 双实现**:

| 实现 | DCT-II | 量化/打包 | 说明 |
|------|--------|----------|------|
| `PdqHashGpu` | GPU 两趟可分离 DCT | CPU 中值量化 + 打包 | GPU 加速频域变换 |
| `PdqHashCpu` | CPU 两趟可分离 DCT | CPU 中值量化 + 打包 | 纯 CPU 降级实现 |

**PdqHashGpu 核心结构**:
```
PdqHashGpu
├── pipeline: Arc<ComputePipeline>       # DCT-II 管线
├── workgroup_size: [u32; 3] = [256,1,1] # 工作组大小
└── buffer_pool: BufferPool              # 缓冲区复用
```

**GPU DCT-II 流程**:
```
64×64 灰度像素 → pack_u8_to_u32 → 上传到 GPU
→ 单 encoder 编码两趟 dispatch:
  第一趟: 水平 1D-DCT (input → intermediate)
  第二趟: 垂直 1D-DCT (intermediate → output)
→ 一次 queue.submit() 提交两趟
→ 下载结果 → bitcast u32→f32 → CPU 量化
```

**CPU 量化流程**:
```
DCT 系数 → 取左上 16×16 = 256 个低频系数
→ 排序求中值 → 每个系数与中值比较生成 bit
→ pack_bits_to_u64 → [u64; 4] (256-bit 哈希)
→ compute_quality → 质量评分 (0.0-1.0)
```

**PdqHashResult**:
```rust
pub struct PdqHashResult {
    pub hash: [u64; 4],  // 256-bit 哈希值
    pub quality: f32,     // 质量评分（0.0-1.0，越高越可靠）
}
```

**质量评分**: 基于 DCT 系数与中值的平均偏差，偏差越大区分度越高，质量越好。

**WGSL Shader** (`pdq_hash.wgsl`):
- 水平模式: 输入 u32 像素值，输出 f32 DCT 系数（bitcast 为 u32 存储）
- 垂直模式: 输入 f32 中间结果（bitcast 为 u32 存储），输出 f32 DCT 系数

**PerceptualHashComputer trait**: `PdqHashGpu` 实现了 `PerceptualHashComputer`，可集成到感知哈希统一框架。

### 4.12 pixel_pack — 像素格式转换

**文件**: `src/pixel_pack.rs` (~9 行)

**职责**: u8↔u32 像素格式转换工具，为 GPU 缓冲区提供标准化的像素打包/解包。

**公开 API**:

| 函数 | 说明 |
|------|------|
| `pack_u8_to_u32(pixels: &[u8]) -> Vec<u32>` | 将 u8 灰度像素数组打包为 u32 数组（每像素 1 个 u32） |
| `unpack_u32_to_u8(data: &[u32], count: usize) -> Vec<u8>` | 将 u32 数组解包为 u8 灰度像素数组 |

**使用场景**: GPU 着色器以 u32 为基本像素单元，CPU 侧以 u8 灰度为主，pixel_pack 提供两者之间的零语义损失转换。

### 4.13 算法一览

| 文件 | WGSL | 行数 | 说明 |
|------|------|------|------|
| `sha256.rs` | `sha256.wgsl` | 748 + 124 | SHA-256 并行哈希（参考实现） |
| `phasher.rs` | — | 1282 | 感知哈希编排器 + GPU 缩放集成 + 阈值预计算 |
| `hash_common.rs` | — | 740 | 共享 trait + 宏 + 零拷贝入口 + 阈值预计算 + Push Constant 支持 |
| `gpu_resize.rs` | `resize.wgsl` | 452 | GPU box filter 缩放 |
| `bktree.rs` | — | 299 | BK-tree 近似搜索（64-bit） |
| `bktree_bytes.rs` | — | 315 | 变长哈希 BK-tree |
| `convolution.rs` | `convolution.wgsl` | 717 + 118 | GPU 2D 卷积（Full2D/Separable） |
| `gaussian_blur.rs` | — | 151 | GPU 高斯模糊（基于可分离卷积） |
| `dihedral.rs` | — | 1044 | D4 群二面体变换（8×8 / 16×16 / 32×32 / 64×64） |
| `matcher.rs` | — | 369 | 哈希匹配策略（Linear/BkTree/Chained/Facade） |
| `matcher_bytes.rs` | — | 411 | 变长哈希匹配策略（Linear/BkTree/Chained/Facade） |
| `hash_bytes.rs` | — | 190 | 变长哈希类型 HashBytes |
| `gpu_matcher.rs` | `hamming.wgsl` | 808 + 303 | GPU 汉明距离匹配器（3 管线架构） |
| `gpu_image_matcher.rs` | — | 230 | 端到端 GPU 图像匹配器 |
| `pdq_hash.rs` | `pdq_hash.wgsl` | 315 + 60 | PDQ 哈希（GPU DCT-II + CPU 量化），feature: pdq |
| `mean_hash.rs` | `mean_hash.wgsl` | 14 + 40 | 均值哈希 |
| `median_hash.rs` | `median_hash.wgsl` | 14 + 40 | 中值哈希 |
| `gradient_hash.rs` | `gradient_hash.wgsl` | 14 + 40 | 梯度哈希 |
| `block_hash.rs` | `block_hash.wgsl` | 14 + 40 | 分块哈希 |
| `vert_gradient_hash.rs` | `vert_gradient_hash.wgsl` | 14 + 40 | 垂直梯度哈希 |
| `double_gradient_hash.rs` | `double_gradient_hash.wgsl` | 14 + 40 | 双向梯度哈希 |

### 4.14 GPU Hash Matcher — GPU 汉明距离匹配器

**文件**: `src/tasks/gpu_matcher.rs` (808 行) + `src/tasks/hamming.wgsl` (303 行)

**职责**: GPU 加速的批量汉明距离计算和并行匹配，支持 64-bit 到 4096-bit 哈希。

**3 管线架构**:

| 管线 | 入口点 | workgroup_size | 用途 |
|------|--------|---------------|------|
| `distance_matrix_pipeline` | `hamming_distance_matrix` | [16, 16, 1] | 计算 N×M 距离矩阵 |
| `nearest_neighbor_pipeline` | `find_nearest_neighbor` | [256, 1, 1] | 最近邻搜索（≤1024-bit 哈希） |
| `nearest_neighbor_large_pipeline` | `find_nearest_neighbor_large` | [32, 1, 1] | 大哈希最近邻（≤4096-bit 哈希） |

**管线选择逻辑**:
```
u32_per_hash ≤ 32（≤1024-bit）→ nearest_neighbor_pipeline（workgroup_size=256）
u32_per_hash > 32（>1024-bit）→ nearest_neighbor_large_pipeline（workgroup_size=32）
```

**核心常量**:

| 常量 | 值 | 说明 |
|------|-----|------|
| `U32_PER_U64_HASH` | 2 | 64-bit 哈希的 u32 数量 |
| `U32_PER_256BIT_HASH` | 8 | 256-bit 哈希的 u32 数量 |
| `U32_PER_1024BIT_HASH` | 32 | 1024-bit 哈希的 u32 数量 |
| `U32_PER_4096BIT_HASH` | 128 | 4096-bit 哈希的 u32 数量 |
| `LARGE_HASH_THRESHOLD` | 32 | 大哈希管线切换阈值 |

**GpuHashMatcher 核心结构**:
```
GpuHashMatcher
├── distance_matrix_pipeline: Arc<ComputePipeline>
├── nearest_neighbor_pipeline: Arc<ComputePipeline>
└── nearest_neighbor_large_pipeline: Arc<ComputePipeline>
```

**4-binding 布局** (与标准 3-binding 不同):

| Binding | 类型 | 用途 |
|---------|------|------|
| 0 | Storage (read_only) | queries 数据 |
| 1 | Storage (read_only) | database 数据 |
| 2 | Storage (read_write) | 输出（距离矩阵或最近邻结果） |
| 3 | Uniform | 参数 (n, m, u32_per_hash, threshold) |

**WGSL 共享内存布局**:

| 入口点 | 共享变量 | 尺寸 | 大小 |
|--------|---------|------|------|
| `hamming_distance_matrix` | `query_tile` | 16×129 = 2064 u32 | 8256 bytes (~8KB) |
| `hamming_distance_matrix` | `db_tile_matrix` | 16×129 = 2064 u32 | 8256 bytes (~8KB) |
| `find_nearest_neighbor` | `db_tile_nearest` | 256×32 = 8192 u32 | 32768 bytes (32KB) |
| `find_nearest_neighbor_large` | `db_tile_nearest_large` | 32×129 = 4128 u32 | 16512 bytes (~16KB) |

**共享内存优化**:
- 距离矩阵: 每个 query/db entry 加载一次 per 16×16 tile，从共享内存复用 16 次（16× 带宽缩减）
- 最近邻: 256 线程协作加载 256 条 db entry，每个线程从共享内存扫描（256× 带宽缩减）
- 大哈希最近邻: 32 线程协作加载 32 条 db entry（32× 带宽缩减）
- Stride 129（非 128）避免 32-way bank conflict（128 % 32 == 0 导致所有线程访问同一 bank）

**OOM 防护**: 当输出缓冲区大小超过 `max_storage_buffer_binding_size` 时，自动将 database 分块处理并合并结果。

**GpuHashMatcherFacade** — 64-bit GPU 匹配器门面:
- 实现 `HashMatcher` trait，适配 CPU 匹配器接口
- 单查询使用 CPU 直接计算（避免 GPU 启动开销）
- 批量查询使用 GPU 距离矩阵（需 `new_with_shared_ctx` 构造）
- 支持 `find_similar_ratio()` 归一化阈值比例查询

**GpuHashMatcherBytes** — 变长哈希 GPU 匹配器:
- 与 `GpuHashMatcher` 功能等价，但支持 64-bit 到 4096-bit 变长哈希
- 内部复用同一个 `hamming.wgsl`，通过 `params.z` (u32_per_hash) 控制哈希长度
- `hash_bytes_to_u32()`: 将 HashBytes 转为 u32 对齐数组（空/零长度哈希防御已添加）

**GpuHashMatcherFacadeBytes** — 变长哈希 GPU 匹配器门面:
- 实现 `HashMatcherBytes` trait
- 批量查询使用 GPU 距离矩阵
- 支持归一化阈值比例查询

**性能数据** (Czkawka 缓存数据，20000 条 1024-bit 哈希):

| 方法 | 规模 | 耗时 | 匹配数 |
|------|------|------|--------|
| CPU 线性扫描 (200 queries × 20000 db) | 4M 比较 | 17.1s | 200 |
| CPU 推算全量 (20000 × 20000) | 400M 比较 | ~1711s | — |
| **GPU 最近邻 (20000 × 20000)** | **400M 比较** | **2.2s** | **20000** |
| GPU 距离矩阵 (500 × 500) | 250K 比较 | 14.6ms | 532 |

**GPU 加速比: 778x** vs CPU 全量最近邻搜索。

### 4.15 Variable-Length Hash — 变长哈希系统

**文件**: `src/tasks/hash_bytes.rs` (190 行)

**职责**: 提供变长哈希类型 `HashBytes`，支持 64-bit 到 4096-bit 的感知哈希，统一匹配 API。

**HashBytes 核心结构**:
```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HashBytes(Vec<u8>);
```

**哈希尺寸对应关系**:

| hash_size | 网格尺寸 | 字节数 | 位宽 |
|-----------|---------|--------|------|
| 8 | 8×8 | 8 | 64 |
| 16 | 16×16 | 32 | 256 |
| 32 | 32×32 | 128 | 1024 |
| 64 | 64×64 | 512 | 4096 |

**创建方式**:

| 方法 | 说明 |
|------|------|
| `from_bytes(bytes)` | 从字节向量创建 |
| `from_u64s(u64s)` | 从 u64 切片创建（GPU 计算结果，小端序） |
| `from_u64(hash)` | 从单个 u64 创建 64-bit 哈希 |

**核心 API**:

| 方法 | 说明 |
|------|------|
| `as_bytes()` | 返回字节切片 |
| `byte_len()` | 返回字节数 |
| `bit_len()` | 返回位宽 |
| `hamming_distance(&other)` | 计算与另一个哈希的汉明距离 |

**MatcherBytes — 变长哈希匹配策略** (`src/tasks/matcher_bytes.rs`, 411 行):

| 匹配器 | 说明 | 召回率 |
|--------|------|--------|
| `LinearScanMatcherBytes` | 精确线性扫描 | 100% |
| `BkTreeMatcherBytes` | BK-tree 近似匹配 | 近似 |
| `ChainedMatcherBytes` | 责任链组合 | 取决于策略 |
| `HashMatcherFacadeBytes` | 统一门面 + 二面体变换增强 | 取决于内部策略 |

**HashMatcherBytes trait**:
```rust
pub trait HashMatcherBytes {
    fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes>;
    fn find_similar_batch(&self, queries: &[HashBytes], threshold: u32) -> Vec<Vec<MatchResultBytes>>;
    fn find_similar_ratio(&self, query: &HashBytes, ratio: f32) -> Vec<MatchResultBytes>;
}
```

**MatchResultBytes**:
```rust
pub struct MatchResultBytes {
    pub hash: HashBytes,
    pub distance: u32,
    pub quality: Option<f32>,
}
```

**归一化阈值比例**: `ratio` 取值 0.0-1.0，内部计算 `threshold = (ratio * bit_len) as u32`。推荐范围：
- 64-bit: 0.0-0.25
- 256-bit: 0.0-0.15
- 1024-bit: 0.0-0.10

**BkTreeBytes — 变长哈希 BK-tree** (`src/tasks/bktree_bytes.rs`, 315 行):
- 与 `BkTree`（硬编码 u64）API 一致，但使用 `HashBytes` 作为哈希类型
- 支持任意长度哈希的插入和搜索
- 三角不等式剪枝，O(log N) 近似最近邻

**命名约定**: `*Bytes` 后缀的匹配器/BK-tree 与 64-bit 版本 API 一致，仅将 `u64` 替换为 `HashBytes`。

### 4.16 GpuImageMatcher — 端到端 GPU 图像匹配器

**文件**: `src/tasks/gpu_image_matcher.rs` (230 行)

**职责**: 组合 `PerceptualHasher`（GPU 感知哈希）和 `GpuHashMatcherBytes`（GPU 距离矩阵），提供一站式 API：批量图像 → GPU 哈希 → GPU 并行匹配。

**核心结构**:
```
GpuImageMatcher
├── hasher: PerceptualHasher       # 感知哈希计算器
├── gpu_matcher: GpuHashMatcherBytes # GPU 距离匹配器
└── hash_size: HashSize            # 哈希尺寸
```

**创建方式**:
- `new(ctx, algorithm)`: 使用默认哈希尺寸（8×8 = 64-bit）
- `with_hash_size(ctx, algorithm, hash_size)`: 指定哈希尺寸

**核心 API**:

| 方法 | 说明 |
|------|------|
| `find_similar(ctx, query_images, query_dims, db_images, db_dims, threshold)` | 端到端相似图像查找 |
| `compute_distance_matrix(ctx, query_images, query_dims, db_images, db_dims)` | 计算完整距离矩阵 |
| `compute_hashes(ctx, images, dims)` | 仅计算图像哈希 |

**find_similar 流程**:
```
1. 计算哈希: query_images → PerceptualHasher::compute() → Vec<u64>
2. 转换类型: Vec<u64> → u64s_to_hash_bytes() → Vec<HashBytes>
   (按 hash_size 分组: 8→1u64, 16→4u64, 32→16u64 per hash)
3. GPU 距离矩阵: GpuHashMatcherBytes::compute_distance_matrix() → Vec<Vec<u32>>
4. CPU 过滤: distance ≤ threshold → Vec<MatchResultBytes>
```

**设计要点**:
- 支持任意 `HashSize`（8/16/32），当 `hash_size ≥ 16` 时产生 256-bit 或更长的哈希
- 内部使用 `GpuHashMatcherBytes` 正确处理变长哈希的距离计算
- `u64s_to_hash_bytes()`: 将 `Vec<u64>` 按 `u64s_per_image` 分组转换为 `Vec<HashBytes>`

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

### 5.5 卷积流水线数据流

```
GpuConvolution::convolve_separable()
    │
    ├── 输入像素 → pixel_pack::pack_u8_to_u32() → Vec<u32>
    │
    ├── BufferPool.acquire() × 3 → input, intermediate, output
    ├── queue.write_buffer() → 上传到 input buffer
    │
    ├── 单 encoder 编码两趟:
    │   ├── 第一趟: 水平 1D 卷积
    │   │   ├── build_params(pass_mode=1) → ConvParams
    │   │   ├── GpuBuffer::from_data() → params_buffer_h
    │   │   └── pipeline.encode_dispatch_into(input → intermediate)
    │   │
    │   └── 第二趟: 垂直 1D 卷积
    │       ├── build_params(pass_mode=2) → ConvParams
    │       ├── GpuBuffer::from_data() → params_buffer_v
    │       └── pipeline.encode_dispatch_into(intermediate → output)
    │
    ├── queue.submit(encoder.finish()) → 一次提交两趟
    ├── output.download_with_pool() → 读取结果
    ├── BufferPool.release() × 3 → 归还缓冲区
    │
    └── pixel_pack::unpack_u32_to_u8() → Vec<u8>
```

**高斯模糊零拷贝流水线**:
```
GpuGaussianBlur::blur_gpu()
    │
    ├── generate_gaussian_kernel_1d() → 1D 高斯核
    └── convolution.convolve_separable_gpu()
        → 输入 GpuBuffer → 输出 GpuBuffer（数据全程在 GPU）
```

### 5.6 PDQ 哈希数据流

```
PdqHashGpu::compute()
    │
    ├── 输入图像 (64×64 灰度) → pixel_pack::pack_u8_to_u32() → Vec<u32>
    │
    ├── BufferPool.acquire() × 3 → input, intermediate, output
    ├── queue.write_buffer() → 上传到 input buffer
    │
    ├── 单 encoder 编码两趟 DCT-II:
    │   ├── 水平 1D-DCT (pass_mode=0): input → intermediate
    │   └── 垂直 1D-DCT (pass_mode=1): intermediate → output
    │
    ├── queue.submit(encoder.finish()) → 一次提交
    ├── output.download_with_pool() → 读取结果
    ├── BufferPool.release() × 3 → 归还缓冲区
    │
    └── CPU 后处理:
        ├── bitcast u32 → f32 → DCT 系数
        ├── 取左上 16×16 低频系数 (256 个)
        ├── 排序求中值 → 二值化 → pack_bits_to_u64()
        └── 返回 Vec<u64> (每图 4 个 u64 = 256-bit)
```

### 5.7 GPU 汉明距离匹配数据流

```
GpuHashMatcher::compute_distance_matrix()
    │
    ├── 输入: queries[N] (u64) + database[M] (u64)
    ├── hashes_to_u32(): 每个 u64 拆为 (lo, hi) 两个 u32
    │
    ├── OOM 检查: N × M × 4 > max_storage_buffer_binding_size?
    │   ├── 否: 单次 dispatch
    │   └── 是: 分块处理 database，合并结果
    │
    ├── 单次 dispatch:
    │   ├── BufferPool.acquire() × 3 → query_buf, db_buf, output_buf
    │   ├── queue.write_buffer() × 2 → 上传 queries + database
    │   ├── params = [n, m, u32_per_hash, 0]
    │   ├── GpuBuffer::from_data() → params_gpu
    │   ├── dispatch [ceil(N/16), ceil(M/16), 1]
    │   │   └── hamming_distance_matrix 入口点
    │   │       └── 共享内存 tile: query_tile(8KB) + db_tile_matrix(8KB)
    │   ├── output.download_with_pool() → 读取 N×M u32 矩阵
    │   └── BufferPool.release() × 3
    │
    └── 解析: 按 N 行 × M 列拆分 → Vec<Vec<u32>>
```

```
GpuHashMatcher::find_nearest_neighbors()
    │
    ├── 输入: queries[N] + database[M] + threshold
    ├── hashes_to_u32() → query_u32, db_u32
    │
    ├── 管线选择:
    │   ├── u32_per_hash ≤ 32 → nearest_neighbor_pipeline (workgroup=256)
    │   │   └── db_tile_nearest: 256×32 = 32KB 共享内存
    │   └── u32_per_hash > 32 → nearest_neighbor_large_pipeline (workgroup=32)
    │       └── db_tile_nearest_large: 32×129 = 16KB 共享内存
    │
    ├── dispatch [ceil(N/workgroup), 1, 1]
    ├── output: N × (index: u32, distance: u32) 对
    │
    └── 解析: 每 2 个 u32 为一组 → Vec<(u32, u32)>
```

### 5.8 端到端 GPU 图像匹配数据流

```
GpuImageMatcher::find_similar()
    │
    ├── Step 1: 计算哈希
    │   ├── query_images → PerceptualHasher::compute() → Vec<u64>
    │   └── db_images → PerceptualHasher::compute() → Vec<u64>
    │
    ├── Step 2: 类型转换
    │   ├── query_u64s → u64s_to_hash_bytes() → Vec<HashBytes>
    │   │   (按 u64s_per_image 分组: 8→1, 16→4, 32→16 per hash)
    │   └── db_u64s → u64s_to_hash_bytes() → Vec<HashBytes>
    │
    ├── Step 3: GPU 距离矩阵
    │   └── GpuHashMatcherBytes::compute_distance_matrix()
    │       ├── hash_bytes_to_u32() → (Vec<u32>, u32_per_hash)
    │       └── dispatch hamming_distance_matrix → Vec<Vec<u32>>
    │
    └── Step 4: CPU 过滤
        └── distance ≤ threshold → Vec<MatchResultBytes>
```

---

## 6. 模块依赖关系

```
lib.rs
├── batch               → buffer, error, pipeline, context
├── buffer              → error, buffer_pool
├── buffer_pool         → buffer
├── context             → error, pipeline, buffer_pool, poll_counter
├── error               → (无依赖)
├── pipeline            → buffer, error
├── pixel_pack          → (无依赖)
├── poll_counter        → (无依赖)
├── pipeline_builder    → context, error, hash_common, phasher
├── backend_dispatcher  → context, error, ComputeBackend
└── tasks
    ├── mod.rs          → (导出子模块)
    ├── sha256          → buffer, buffer_pool, context, error, pipeline
    ├── phasher         → context, error, hash_common, hash_algorithms, gpu_resize
    ├── hash_common     → buffer, context, error, pipeline, batch
    ├── gpu_resize      → buffer, buffer_pool, context, error, pipeline
    ├── bktree          → (无依赖, 纯算法)
    ├── bktree_bytes    → hash_bytes (纯算法)
    ├── hash_bytes      → (无依赖, 纯类型)
    ├── matcher         → bktree, dihedral
    ├── matcher_bytes   → bktree_bytes, dihedral, hash_bytes
    ├── gpu_matcher     → buffer, context, error, pipeline, bktree, matcher, matcher_bytes, hash_bytes
    ├── gpu_image_matcher → context, error, gpu_matcher, hash_bytes, matcher_bytes, phasher, HashSize
    ├── convolution     → buffer, buffer_pool, context, error, pipeline, pixel_pack
    ├── gaussian_blur   → convolution, context, error, buffer_pool
    ├── dihedral        → (无依赖, 纯算法)
    ├── pdq_hash        → buffer, buffer_pool, context, error, pipeline, hash_common, pixel_pack (feature: pdq)
    ├── mean_hash       → hash_common (macros)
    ├── median_hash     → hash_common (macros)
    ├── gradient_hash   → hash_common (macros)
    ├── block_hash      → hash_common (macros)
    ├── vert_gradient_hash → hash_common (macros)
    └── double_gradient_hash → hash_common (macros)
```

**依赖方向**: 能力层内部单向依赖，业务层依赖能力层，算法间通过 matcher、dihedral、hash_bytes 有有限横向依赖。gpu_matcher 是横向依赖最多的模块（依赖 bktree、matcher、matcher_bytes、hash_bytes）。

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

**模式 C: 卷积类算法（如高斯模糊）**
```
1. 创建 src/tasks/new_filter.rs
2. 委托 GpuConvolution 执行，自动生成卷积核
3. 提供零拷贝接口（*_gpu 方法）支持流水线组合
4. 在 src/tasks/mod.rs 中导出
5. 在 src/lib.rs 中重导出公共类型
```

**模式 D: 变长哈希匹配器**
```
1. 创建 src/tasks/new_matcher_bytes.rs
2. 实现 HashMatcherBytes trait
3. 使用 *Bytes 后缀区分于 64-bit 版本
4. 在 src/tasks/mod.rs 中导出
5. 在 src/lib.rs 中重导出公共类型
```

### 7.2 WGSL Shader 规范

```wgsl
// 固定绑定布局（与 ComputePipeline 一致）
@group(0) @binding(0) var<storage, read> input: array<T>;
@group(0) @binding(1) var<storage, read_write> output: array<T>;

struct Params {
    image_count: u32,
    width: u32,
    height: u32,
    hash_size: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(N)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ...
}
```

**Uniform 对齐注意**: WGSL 中 `array<f32, N>` 的 stride 为 4 字节，但 uniform buffer 要求 16 字节对齐。当 uniform 结构体包含大数组时，需确保数组长度满足对齐要求（如 convolution.wgsl 中 kernel 数组填充至 124 个 f32）。

**汉明距离着色器特殊布局**: `hamming.wgsl` 使用 4-binding 布局（2 个只读 Storage + 1 个读写 Storage + 1 个 Uniform），与标准 3-binding 布局不同。

### 7.3 Workgroup 大小选择

| 场景 | 推荐大小 | 说明 |
|------|---------|------|
| 数据并行（SHA-256、感知哈希） | `[256, 1, 1]` | 每条消息/图像一个 work item |
| 汉明距离矩阵 | `[16, 16, 1]` | 2D tile 优化，共享内存复用 |
| 最近邻搜索（≤1024-bit） | `[256, 1, 1]` | 256 线程协作加载 db tile |
| 最近邻搜索（≤4096-bit） | `[32, 1, 1]` | 减少寄存器压力，适配大哈希 |
| 卷积/DCT | `[256, 1, 1]` | 每个像素一个 work item |

---

## 8. 构建与测试

### 8.1 构建命令

```bash
cargo build                         # 默认构建（cpu-fallback 默认开启）
cargo build --features image        # 启用图像支持
cargo build --features pdq          # 启用 PDQ 哈希
cargo build --features "image,pdq"  # 同时启用
cargo build --release               # 发布构建
```

### 8.2 测试

```bash
cargo test                          # 运行所有测试
cargo test sha256                   # 运行特定测试
cargo test --test gpu_resize_test   # 运行 GPU 缩放测试
cargo test --test bktree_test       # 运行 BK-tree 测试
cargo test --test cache_gpu_matcher_test  # 运行缓存数据 GPU 匹配测试
cargo test --features pdq           # 运行 PDQ 哈希测试
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
cargo clippy --features image       # 含 image feature 的 lint
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
| HashSize 替代 HashBits | 结构体替代枚举 | 支持任意 2 的幂次网格尺寸，扩展性更好 |
| PhashParams 4 字段 Push Constant | 16 字节 | 满足 Push Constant 4 字节对齐和 Uniform 16 字节对齐 |
| block_hash 即时计算 block_mean | 函数调用替代大数组 | 避免 DX12 临时寄存器溢出（原 array<u32,4096> 超限） |
| 卷积核数组填充至 124 f32 | 16 字节对齐 | WGSL uniform buffer 要求 16 字节对齐，kernel 数组 stride 4 不满足要求 |
| 高斯模糊委托卷积 | 组合优于继承 | GpuGaussianBlur 封装 GpuConvolution，复用可分离卷积逻辑 |
| 哈希匹配策略模式 | trait + 多实现 | LinearScan/BkTree/Chained 可互换，Facade 封装二面体变换 |
| PDQ GPU+CPU 混合 | GPU DCT + CPU 量化 | DCT 适合 GPU 并行，中值量化需排序不适合 GPU |
| pixel_pack 独立模块 | 单一职责 | u8↔u32 转换被多个业务模块复用，独立避免循环依赖 |
| GPU 汉明距离 3 管线 | 距离矩阵 + 标准最近邻 + 大哈希最近邻 | 不同场景最优管线，共享内存布局适配不同哈希尺寸 |
| 变长哈希 HashBytes | Vec\<u8\> 封装 | 统一 64~4096-bit 哈希类型，*Bytes 后缀与 64-bit API 一致 |
| 管线选择阈值 32 | u32_per_hash > 32 → 大哈希管线 | 1024-bit 为分界，大哈希需更小 workgroup 减少寄存器压力 |
| 共享内存 stride 129 | 避免 bank conflict | 128 % 32 == 0 导致 32-way bank conflict，129 % 32 == 1 分散访问 |
| 阈值预计算 | CPU 预计算 + f32::to_bits() | 消除 O(N²) 着色器循环，Mean/Median Hash 性能优化 |
| 声明式管线构建器 | 链式 API + 自动零拷贝 | 简化多步骤 GPU 流水线配置，降低使用门槛 |
| BackendDispatcher trait | 统一 GPU/CPU 路径选择 | 替代散布的 if ctx.backend() == Cpu 手动分支 |
| PollCounter 全局原子 | Relaxed 排序 | 最小化开销，仅关心最终计数准确性 |
| OOM 防护分块 | database 分块处理 | 输出缓冲区超 max_storage_buffer_binding_size 时自动分块 |

---

## 10. 已知限制

| 限制 | 说明 |
|------|------|
| GPU 调度开销 | 单次 dispatch 约 1.6ms，小批量场景不如 CPU |
| Multi-block 数据依赖 | SHA-256 多 block 消息无法完全并行，中间 block 需同步 |
| image feature 可选 | 默认不启用图像支持，需显式开启 |
| GPU 缩放质量 | box filter 适合下采样，上采样质量不如 Lanczos |
| 分批粒度 | 分批依据 256MB 硬限制，未考虑实际显存容量 |
| HashSize=64 DX12 限制 | hash_size=64 时部分着色器可能因工作寄存器压力在 DX12 后端编译失败 |
| convolution.wgsl Uniform 对齐 | kernel 数组 stride 为 4 字节，不满足 WGSL uniform buffer 16 字节对齐要求，当前通过填充至 124 个 f32 规避 |
| PDQ 哈希需 pdq feature | PDQ 相关功能需 `--features pdq` 显式启用，默认不编译 |
| hamming.wgsl 共享内存 | `db_tile_nearest` 使用 32KB 共享内存（256×32 u32），`db_tile_nearest_large` 使用 16KB（32×129 u32），均在典型 64KB 限制内 |
| hash_bytes_to_u32 空哈希 | 空/零长度哈希返回空结果（防御性检查已添加） |
| 缓存数据测试 JSON 截断 | `load_hashes_from_cache` 仅读取前 N 字节并截断 JSON，避免解析整个 1.4GB 文件 |
| GPU 匹配器单查询开销 | GpuHashMatcherFacade 单查询使用 CPU 直接计算（GPU 启动开销不值得），批量查询才走 GPU |

---

*文档生成完毕。基于源码分析，覆盖能力层 9 个模块、业务层 21 个模块的完整架构。*
