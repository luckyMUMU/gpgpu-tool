# GPU 管线优化规范

## Why

当前 wgpu-tool 项目存在内存管理碎片化（各组件独立持有 BufferPool）、零拷贝管线断裂（Blur → Resize → Hash 未完整串联）、PerceptualHasher 职责过重（调度/降级/算法混合）、缺少声明式管线组合 API 等问题。此外，当前 WGSL 着色器未针对 GPU 硬件特性进行优化：全部使用 1D workgroup_size(256)、无 LDS 共享内存利用、无 Push Constant 传递小参数、无内存合并访问优化。代码审查还发现了多个 Critical 级别的正确性和安全性问题（可分离卷积精度丢失、resize 线程模型不合理、API 越界 panic 等），必须优先修复。基于优化方案文档、RX5700 GPU 并行优化报告和代码审查意见，需要对 GPU 计算管线进行系统性优化，涵盖正确性修复、架构重构和着色器优化三个维度。

## 调研发现

### 代码审查发现（Critical 级别，必须优先修复）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| C1 | `unpack_u32_to_u8` 越界 panic：`data[..count]` 当 count > data.len() 时 panic | pixel_pack.rs:7 | GPU 设备丢失/OOM 时导致不可恢复 panic |
| C2 | 可分离卷积中间结果精度丢失：第一趟输出 `u32(clamp(sum, 0, 255))` 量化为整数，第二趟读取已量化值 | convolution.wgsl:116 | 高斯模糊等预处理精度显著低于 Full2D，影响 GPU/CPU 结果一致性 |
| C3 | `GpuBatchSubmitter` 无 Drop 实现：submit 后未 wait_all 时 encoder 和 staging buffer 泄漏 | batch.rs | 资源泄漏 |
| C4 | resize.wgsl 线程分配模型不合理：每线程处理整张图像，batch 较小时 GPU 利用率极低 | resize.wgsl:37-63, gpu_resize.rs:240 | 2 张图像缩放仅启动 2 个 GPU 线程 |
| C5 | `device()`/`queue()` 在 CPU 降级模式下 panic，形成跨模块 panic 链 | context.rs:297-311, batch.rs | 库级 API 不应有此行为 |

### 代码审查发现（Important 级别）

| # | 问题 | 位置 |
|---|------|------|
| I1 | `PerceptualHasher::compute` 每次创建新 `PHasherCpu` 实例，应缓存 | phasher.rs:201 |
| I2 | `BatchJob` 硬编码 3-buffer 绑定布局，限制通用性 | batch.rs:13-24 |
| I3 | `to_wgpu_usage()` 三处重复实现且行为不一致 | buffer.rs, buffer_pool.rs, pipeline.rs |
| I4 | `BufferPool` 使用 `RefCell` 不支持跨线程共享 | buffer_pool.rs:115-117 |
| I5 | PDQ GPU/CPU 精度不一致：GPU 用 f32，CPU 用 f64 | pdq_hash.rs |
| I6 | `GpuResize::resize_batch` 和 `resize_batch_gpu` 验证逻辑重复 | gpu_resize.rs |
| I7 | SHA-256 链式着色器 O(N) 偏移计算，导致 warp 分化 | sha256_chained.wgsl:43-46 |
| I8 | `GpuContext` 的 fxhash 碰撞风险：管线缓存 key 碰撞导致静默错误 | context.rs:50-57 |
| I9 | BindGroup 每次重新创建，批量场景性能瓶颈 | pipeline.rs:171,198 |
| I10 | `GpuBuffer::from_data` 为所有缓冲区添加 `COPY_SRC`，浪费 GPU 资源配额 | buffer.rs:66 |
| I11 | `GpuResize::resize_batch_gpu` 对空输入返回 `Err`，而 `resize_batch` 返回 `Ok(vec![])` | gpu_resize.rs |
| I12 | `ChainedMatcher::bk_tree_plus_linear` Union 策略下 BK-tree 结果冗余 | matcher.rs:110-117 |

### 内存管理现状
- 各组件（Sha256Computer、GpuConvolution、GpuResize、各哈希算法）各自持有独立 BufferPool 实例，无法跨组件复用缓冲区
- GpuBatchSubmitter 每次提交创建新 staging buffer，未接入 BufferPool
- 零拷贝管线中 BufferPool 传递通过 `buffer_pool()` 方法暴露引用实现跨组件归还，但获取仍需从各自池分配

### 零拷贝管线现状
- `GpuResize.resize_batch_gpu()` → `compute_phash_from_gpu_buffer()` 零拷贝链路已实现
- `GpuConvolution.convolve_separable_gpu()` / `GpuGaussianBlur.blur_gpu()` 返回 GpuBuffer，但未被 PerceptualHasher 集成
- GpuGaussianBlur 作为独立模块存在，PerceptualHasher 不支持高斯模糊预处理步骤

### PerceptualHasher 现状
- `compute()` 方法 120+ 行，混合 CPU 降级判断、尺寸检查、GPU 批量调度、CPU 缩放回退、PDQ 特殊路径
- 降级逻辑散布在各业务方法中，每个 compute 方法重复 `if ctx.backend() == Cpu` 检查
- 无声明式管线组合能力，无法声明"模糊→缩放→哈希"处理流水线

### WGSL 着色器硬件优化现状（基于 RX5700 报告分析）
- **Workgroup Size**：全部 11 个着色器使用 `@workgroup_size(256)` 1D 模式，2D 图像处理场景未利用 8×8=64 的最优配置
- **LDS 共享内存**：无任何着色器使用 `var<workgroup>` 共享内存，卷积/模糊的邻域滤波存在大量冗余全局内存读取
- **Push Constant**：无使用，小参数（如 PhashParams）通过 Uniform/Storage Buffer 传递，增加 Bind Group 切换开销
- **内存合并访问**：无 Morton Swizzle 线程重排，图像处理场景 Cache 命中率未优化
- **FP16 精度**：全部使用 f32，DCT 和哈希计算可降为 f16 翻倍吞吐并减半 VGPR 占用
- **Storage Buffer 对齐**：未强制 256 字节对齐，影响合并写入效率
- **Dispatch 粒度**：未根据 CU 数量动态计算最优 dispatch 量

## What Changes

- **正确性修复（P0）**：修复 unpack_u32_to_u8 越界、可分离卷积精度丢失、GpuBatchSubmitter 资源泄漏
- **线程模型修复（P1）**：重构 resize.wgsl 为每像素一线程模型
- **API 安全性改进（P1）**：device()/queue() 安全化，消除跨模块 panic 链
- **统一内存池**：BufferPool 从各组件自有提升为 GpuContext 级别共享
- **零拷贝管线串联**：PerceptualHasher 支持可选高斯模糊预处理，实现 blur_gpu() → resize_gpu() → hash_gpu() 完整零拷贝链路
- **声明式管线 API**：新增 GpuPipelineBuilder，支持链式声明处理步骤
- **PerceptualHasher 重构**：抽取 BackendDispatcher trait 统一 GPU/CPU 降级逻辑，分离调度策略与算法逻辑
- **WGSL 着色器硬件优化**：基于 RX5700 报告优化 workgroup size、LDS 利用、内存合并访问、Push Constant
- **性能基准建立**：建立单图像延迟、批量吞吐量、内存占用基准，量化优化效果

## Impact

- Affected specs: 感知哈希全部 6 种算法的执行路径和 WGSL 着色器、卷积/模糊/缩放模块的 BufferPool 使用方式和着色器优化
- Affected code:
  - `src/pixel_pack.rs` — 越界保护修复
  - `src/batch.rs` — Drop 实现、BatchJob 动态绑定、接入 BufferPool
  - `src/context.rs` — GpuContext 持有共享 BufferPool、device()/queue() 安全化、fxhash 改进
  - `src/buffer_pool.rs` — 256 字节对齐、to_wgpu_usage 统一
  - `src/buffer.rs` — to_wgpu_usage 统一、COPY_SRC 按需添加
  - `src/pipeline.rs` — PipelineDescriptor 支持 Push Constant、BindGroup 缓存
  - `src/tasks/phasher.rs` — 重构 compute()，集成高斯模糊，分离调度逻辑，缓存 PHasherCpu
  - `src/tasks/hash_common.rs` — compute_phash 系列函数适配共享池
  - `src/tasks/convolution.rs` — BufferPool 引用方式变更、精度修复
  - `src/tasks/convolution.wgsl` — 中间结果 bitcast 精度修复、LDS 共享内存优化、2D workgroup size
  - `src/tasks/gaussian_blur.rs` — BufferPool 引用方式变更
  - `src/tasks/gpu_resize.rs` — BufferPool 引用方式变更、线程模型重构、API 一致性修复
  - `src/tasks/resize.wgsl` — 每像素一线程模型、2D workgroup size
  - `src/tasks/sha256.rs` — BufferPool 引用方式变更
  - `src/tasks/sha256_chained.wgsl` — 前缀和替代 O(N) 偏移扫描
  - `src/tasks/*_hash.wgsl` 6 个着色器 — workgroup size 优化、Push Constant 传递小参数
  - `src/tasks/matcher.rs` — ChainedMatcher 语义修正
  - 新增 `src/pipeline_builder.rs` — 声明式管线 API
  - 新增 `src/backend_dispatcher.rs` — 统一降级策略
  - `benches/` — 新增性能基准

## ADDED Requirements

### Requirement: unpack_u32_to_u8 越界保护（审查 C1）

`unpack_u32_to_u8` SHALL 对 `count` 参数进行边界保护，当 `count > data.len()` 时使用 `count.min(data.len())` 而非 panic。

`unpack_u32_to_u8` SHALL 添加 `debug_assert!(count <= data.len())` 在开发期捕获异常调用。

#### Scenario: GPU 返回数据量不足时安全降级

- **WHEN** GPU 设备丢失导致返回数据量少于预期 count
- **THEN** `unpack_u32_to_u8` 安全截取可用数据，不 panic
- **AND** debug 模式下断言失败提示调用方检查

### Requirement: 可分离卷积中间结果精度修复（审查 C2）

convolution.wgsl 的可分离卷积第一趟（水平 1D）SHALL 使用 `bitcast<u32>(sum)` 存储 f32 中间结果，而非 `u32(clamp(sum, 0.0, 255.0))` 量化为整数。

第二趟（垂直 1D）SHALL 使用 `bitcast<f32>(pixels[idx])` 还原 f32 中间值，再执行卷积计算。

最终输出 SHALL 仍使用 `u32(clamp(sum, 0.0, 255.0))` 量化为像素值。

#### Scenario: 可分离高斯模糊精度与 Full2D 一致

- **WHEN** 用户使用 GpuGaussianBlur 执行可分离高斯模糊
- **THEN** 可分离模式的输出与 Full2D 模式精度一致（差异 ≤ 1）
- **AND** GPU 和 CPU 路径结果一致

### Requirement: GpuBatchSubmitter 资源安全（审查 C3）

`GpuBatchSubmitter` SHALL 实现 `Drop` trait，在 drop 时调用 `clear()` 清理未完成的 pending jobs 和 staging buffers。

`BatchJob` 的绑定列表 SHALL 从硬编码的 3-buffer 改为 `Vec<GpuBuffer>`，支持任意数量的绑定。`submit()` 中 `create_bind_group` SHALL 使用 `BatchJob` 的动态绑定列表而非硬编码 `[&job.input, &job.output, &job.params]`。

#### Scenario: GpuBatchSubmitter 提前 drop

- **WHEN** 用户创建 GpuBatchSubmitter、submit 后未调用 wait_all 就 drop
- **THEN** Drop 实现自动清理 pending jobs 和 staging buffers
- **AND** 无 GPU 资源泄漏

#### Scenario: BatchJob 支持 2-binding 管线

- **WHEN** 用户创建只有 2 个 buffer 的 BatchJob
- **THEN** submit() 正确创建 2-binding 的 BindGroup
- **AND** 不产生 wgpu validation error

### Requirement: resize.wgsl 线程模型重构（审查 C4）

resize.wgsl SHALL 改为每目标像素一个线程的模型，dispatch 维度为 `image_count * dst_w * dst_h`。

着色器 SHALL 从 `global_invocation_id.x` 解码图像索引和像素坐标：
- `img_idx = gid.x / (dst_w * dst_h)`
- `pixel_idx = gid.x % (dst_w * dst_h)`
- `dy = pixel_idx / dst_w`
- `dx = pixel_idx % dst_w`

Rust 端 dispatch 计算逻辑 SHALL 相应更新。

#### Scenario: 批量 2 张图像缩放到 8×8

- **WHEN** 用户批量缩放 2 张图像到 8×8
- **THEN** dispatch 维度为 2×8×8=128 个线程
- **AND** GPU 利用率显著高于当前 2 线程模型

### Requirement: device()/queue() 安全化（审查 C5）

`GpuContext::device()` 和 `queue()` SHALL 在 CPU 降级模式下返回 `Result` 而非 panic。

现有 `try_device()` / `try_queue()` SHALL 成为推荐 API。`device()` / `queue()` SHALL 标记为 `#[deprecated]`，引导调用方使用 `try_` 变体或通过 BackendDispatcher 自动处理后端选择。

#### Scenario: CPU 降级模式下调用 GPU 操作

- **WHEN** 在 CPU 降级模式下意外调用 `ctx.device()`
- **THEN** 返回 `Err(GpuError::...)` 而非 panic
- **AND** 调用方可以优雅处理错误

### Requirement: GpuContext 级别共享 BufferPool

系统 SHALL 在 GpuContext 中持有一个共享的 BufferPool 实例，所有 GPU 计算组件通过 `ctx.buffer_pool()` 获取引用，不再各自持有独立实例。

GpuContext SHALL 提供 `buffer_pool(&self) -> &BufferPool` 方法返回共享池的不可变引用。由于 BufferPool 内部使用 RefCell 实现内部可变性，不可变引用即可完成 acquire/release 操作。

各计算组件（Sha256Computer、GpuConvolution、GpuResize、GpuGaussianBlur、各哈希算法）SHALL 移除自有的 `buffer_pool` 字段，改为在构造时接收 `&BufferPool` 引用或在 dispatch 时从 GpuContext 获取。

#### Scenario: GpuConvolution 使用共享池

- **WHEN** 用户创建 GpuConvolution 实例
- **THEN** GpuConvolution 不持有自有 BufferPool，通过 `ctx.buffer_pool()` 获取共享池
- **AND** 多个 GpuConvolution 实例共享同一池，缓冲区可跨实例复用

### Requirement: to_wgpu_usage 统一（审查 I3）

`to_wgpu_usage()` / `buffer_usages()` 当前在 buffer.rs、buffer_pool.rs、pipeline.rs 三处重复实现且行为不一致。SHALL 统一到 `BufferUsage` 类型上的单一方法，消除重复和不一致。

统一后的方法 SHALL 通过参数区分是否需要 `COPY_SRC` flag，而非在不同文件中隐式添加。

#### Scenario: BufferUsage::Storage 的 wgpu flags 一致

- **WHEN** 任何模块查询 `BufferUsage::Storage` 对应的 wgpu flags
- **THEN** 结果一致，不再因调用位置不同而产生不同 flags
- **AND** `COPY_SRC` 按需显式添加而非隐式包含

### Requirement: BindGroup 缓存（审查 I9）

ComputePipeline 或 GpuContext SHALL 提供 BindGroup 缓存，避免每次 dispatch 重新创建。

缓存 key SHALL 基于管线和缓冲区组合的标识，缓存值 SHALL 使用弱引用或 LRU 淘汰避免内存膨胀。

#### Scenario: 批量 SHA-256 哈希复用 BindGroup

- **WHEN** Sha256BatchSubmitter 对相同参数布局多次 dispatch
- **THEN** BindGroup 从缓存获取而非重新创建
- **AND** 批量场景性能提升

### Requirement: 零拷贝管线完整串联

系统 SHALL 支持从高斯模糊到缩放到哈希计算的完整零拷贝 GPU 流水线。

PerceptualHasher SHALL 支持可选的高斯模糊预处理步骤。当启用时，计算流程为：blur_gpu() → resize_gpu() → hash_gpu()，所有中间数据以 GpuBuffer 形式在 GPU 端传递，不经过 CPU 中转。

PerceptualHasher 的构造函数 SHALL 扩展，支持配置高斯模糊参数（sigma 值和核大小）。

#### Scenario: 带高斯模糊的感知哈希计算

- **WHEN** 用户创建 PerceptualHasher 并启用高斯模糊预处理（sigma=1.0）
- **THEN** 计算流程为：图像上传 → GPU 高斯模糊 → GPU 缩放 → GPU 哈希计算
- **AND** 模糊到缩放、缩放到哈希之间数据不离开 GPU

### Requirement: 声明式管线 API

系统 SHALL 提供 GpuPipelineBuilder 类型，支持链式声明 GPU 处理步骤。

GpuPipelineBuilder SHALL 支持以下操作：
- `blur(sigma, kernel_size)` — 添加高斯模糊步骤
- `resize(width, height)` — 添加图像缩放步骤
- `hash(algorithm, hash_size)` — 添加哈希计算步骤

GpuPipelineBuilder SHALL 自动推导零拷贝传递路径：当连续步骤均支持 GPU 执行时，中间结果以 GpuBuffer 形式传递；当某步骤需要 CPU 降级时，自动插入 CPU↔GPU 数据转换。

GpuPipelineBuilder SHALL 提供 `execute(&self, ctx: &GpuContext, images: &[Vec<u8>]) -> Result<Vec<Vec<u64>>, GpuError>` 方法执行完整管线。

#### Scenario: 声明式管线组合

- **WHEN** 用户编写 `GpuPipelineBuilder::new(ctx).blur(1.0, 5).resize(8, 8).hash(HashAlgorithm::Mean, HashSize::default()).execute(&images)`
- **THEN** 系统自动构建 blur → resize → hash 的零拷贝 GPU 流水线并执行
- **AND** 返回与直接调用 PerceptualHasher 相同格式的哈希结果

### Requirement: BackendDispatcher 统一降级策略

系统 SHALL 提供 BackendDispatcher trait，封装 GPU/CPU 后端选择和降级逻辑。

各业务模块（Sha256Computer、PerceptualHasher 等）SHALL 使用 BackendDispatcher 处理降级，不再各自编写 `if ctx.backend() == Cpu` 分支。

PerceptualHasher SHALL 缓存 `PHasherCpu` 实例（审查 I1），而非每次 compute() 创建新实例。

#### Scenario: SHA-256 使用 BackendDispatcher

- **WHEN** Sha256Computer 执行 compute() 且 GPU 不可用
- **THEN** BackendDispatcher 自动委托到 Sha256Cpu 实现
- **AND** 调用方无需感知后端切换

### Requirement: PerceptualHasher 职责分离

PerceptualHasher 的 compute() 方法 SHALL 拆分为独立的调度阶段：

1. **预处理阶段**（可选高斯模糊）— 返回 GpuBuffer 或 Vec<u8>
2. **缩放阶段** — 返回 GpuBuffer 或 Vec<u8>
3. **哈希计算阶段** — 返回 Vec<u64>

每个阶段 SHALL 可独立调用，也可通过 PerceptualHasher 编排器自动串联。

#### Scenario: 单独调用预处理阶段

- **WHEN** 用户只需对图像进行高斯模糊预处理
- **THEN** 可直接调用 `hasher.preprocess_gpu(ctx, &images)` 获取模糊后的 GpuBuffer
- **AND** 无需执行后续缩放和哈希步骤

### Requirement: WGSL 着色器 Workgroup Size 优化

基于 RX5700 GPU 并行优化报告，2D 图像处理着色器 SHALL 使用 2D workgroup size 以优化缓存命中率和内存合并访问。

具体优化规则：
- **2D 图像处理着色器**（resize.wgsl、convolution.wgsl、6 个哈希着色器）SHALL 支持 `@workgroup_size(8, 8, 1)` 模式
- **1D 批处理着色器**（sha256.wgsl、sha256_chained.wgsl）SHALL 保持 `@workgroup_size(64)` 或 `@workgroup_size(128)`
- PipelineDescriptor 和 ComputePipeline SHALL 支持运行时切换 workgroup size

#### Scenario: 图像缩放使用 2D workgroup

- **WHEN** GpuResize 使用 2D workgroup_size=(8,8,1) 执行缩放
- **THEN** 每个 workgroup 处理 8×8 像素块，内存访问模式为合并写入
- **AND** dispatch_count 计算为 `(ceil(width/8), ceil(height/8), image_count)`

### Requirement: 卷积着色器 LDS 共享内存优化

convolution.wgsl SHALL 使用 LDS（`var<workgroup>`）共享内存优化邻域滤波，消除冗余全局内存读取。

优化策略：
- 可分离卷积的每趟 1D 滤波 SHALL 将当前线程所需的邻域像素预加载到 LDS
- LDS 数据布局 SHALL 使用 Struct of Arrays + padding 避免 Bank Conflict
- LDS 预加载后 SHALL 使用 `workgroupBarrier()` 同步
- 中间结果（水平卷积输出）SHALL 存储在 LDS 中而非写回全局内存

### Requirement: Push Constant 传递小参数

PipelineDescriptor SHALL 支持 Push Constant 布局定义，用于传递 ≤128 字节的小参数。

ComputePipeline::dispatch 和 encode_dispatch_into SHALL 支持设置 Push Constant 数据。

### Requirement: Storage Buffer 256 字节对齐

所有 GPU 缓冲区分配 SHALL 对齐到 256 字节边界，优化合并写入效率。

BufferPool 的 size_class 计算 SHALL 确保分配大小向上对齐到 256 字节的倍数。

### Requirement: Dispatch 粒度优化

Rust 端 dispatch 计算逻辑 SHALL 根据目标 GPU 的 CU 数量动态计算最优 dispatch 量。

GpuContext SHALL 查询并缓存 GPU 适配器的 CU 数量。总 dispatch 量 SHALL ≥ CU 数 × 4。

### Requirement: SHA-256 链式着色器前缀和优化（审查 I7）

sha256_chained.wgsl 的 O(N) 偏移扫描 SHALL 替换为 Rust 端预计算的前缀和表，通过额外输入 buffer 传入着色器，将偏移查找从 O(N) 降为 O(1)。

#### Scenario: 10000 条链式消息偏移查找

- **WHEN** 用户批量计算 10000 条链式 SHA-256 消息
- **THEN** 着色器中偏移查找为 O(1) 而非 O(N)
- **AND** 无 warp 分化问题

### Requirement: ChainedMatcher 语义修正（审查 I12）

`ChainedMatcher::bk_tree_plus_linear` 的 Union 策略 SHALL 修正为 FirstHit 策略，使 BK-tree 快速返回、线性扫描补充的语义正确。

#### Scenario: BK-tree + 线性扫描组合匹配

- **WHEN** 用户使用 `bk_tree_plus_linear` 组合匹配器
- **THEN** BK-tree 先快速返回近似结果，线性扫描补充 BK-tree 漏掉的结果
- **AND** 不存在冗余匹配

### Requirement: fxhash 安全网（审查 I8）

PipelineCache 的 `get_or_create` SHALL 在缓存命中时额外做一次 WGSL 源码的 `==` 比较，防止 fxhash 碰撞导致返回错误管线。

#### Scenario: fxhash 碰撞被检测

- **WHEN** 两个不同 WGSL 着色器产生相同的 fxhash 值
- **THEN** 缓存命中时源码比较发现不匹配，重新编译正确管线
- **AND** 不产生静默计算错误

### Requirement: GpuBuffer COPY_SRC 按需添加（审查 I10）

`GpuBuffer::from_data` 和 `from_bytes` SHALL 不再无条件添加 `COPY_SRC` flag，改为让调用方按需指定。

需要被下载的缓冲区（输出缓冲区）SHALL 显式添加 `COPY_SRC`，纯输入缓冲区 SHALL 不添加。

### Requirement: GpuResize API 一致性（审查 I11）

`resize_batch_gpu` 对空输入 SHALL 返回 `Ok(vec![])` 而非 `Err`，与 `resize_batch` 行为一致。

### Requirement: 性能基准建立

系统 SHALL 建立以下性能基准：
1. 单图像感知哈希延迟
2. 批量处理吞吐量（1/10/100/1000）
3. 内存占用
4. GPU vs CPU 对比
5. 零拷贝 vs 非零拷贝对比
6. Workgroup Size 对比（1D vs 2D）
7. LDS 优化对比
8. Push Constant vs Uniform Buffer 对比
9. 可分离卷积精度修复前后对比

## MODIFIED Requirements

### Requirement: GpuConvolution 构造函数

GpuConvolution 的构造函数 SHALL 不再创建自有 BufferPool，改为接收 GpuContext 引用并使用 `ctx.buffer_pool()` 获取共享池。

### Requirement: GpuResize 构造函数

GpuResize 的构造函数 SHALL 不再创建自有 BufferPool，改为接收 GpuContext 引用并使用 `ctx.buffer_pool()` 获取共享池。

### Requirement: GpuGaussianBlur 构造函数

GpuGaussianBlur 的构造函数 SHALL 不再委托 GpuConvolution 的自有 BufferPool，改为通过 GpuContext 获取共享池。

### Requirement: Sha256Computer 构造函数

Sha256Computer 和 Sha256BatchSubmitter SHALL 不再创建自有 BufferPool，改为使用 GpuContext 共享池。

### Requirement: declare_phash_computer 宏

宏生成的 struct SHALL 移除 `buffer_pool: BufferPool` 字段，改为在需要时从 GpuContext 获取共享池引用。

### Requirement: PipelineDescriptor

PipelineDescriptor SHALL 扩展支持 Push Constant 布局定义。

### Requirement: ComputePipeline dispatch

ComputePipeline::dispatch 和 encode_dispatch_into SHALL 支持可选的 Push Constant 数据参数。

### Requirement: GpuContext::device/queue

`device()` 和 `queue()` SHALL 标记为 `#[deprecated]`，推荐使用 `try_device()` / `try_queue()`。

## REMOVED Requirements

### Requirement: 各组件独立 BufferPool

**Reason**: 统一为 GpuContext 级别共享 BufferPool，消除跨组件缓冲区无法复用的浪费。
**Migration**: 所有 `self.buffer_pool` 使用改为 `ctx.buffer_pool()` 调用。

### Requirement: BatchJob 硬编码 3-buffer

**Reason**: 改为动态绑定列表 `Vec<GpuBuffer>`，支持任意数量的绑定。
**Migration**: `BatchJob { input, output, params }` 改为 `BatchJob { buffers: Vec<GpuBuffer>, ... }`。

## 未来方向（记录但不实施）

1. **内核融合框架**：自动将多个连续 GPU 操作融合为单个 WGSL 着色器
2. **JIT 编译缓存**：运行时动态生成 WGSL 着色器代码
3. **PipelineCache 磁盘持久化**：利用 wgpu PipelineCache 特性
4. **统一门面 API**：提供 `GpgpuTool` 全局门面
5. **多 GPU 支持**：数据并行处理
6. **WebAssembly 优化**：优化 WASM 内存传输
7. **FP16 精度**：DCT 和哈希计算降为 f16
8. **Morton Swizzle 线程重排**：提升 Cache 命中率约 8%
9. **FidelityFX SPD 模式**：单 Pass 多级降采样
10. **GPU 时间戳查询 Profiling**
11. **类型状态模式区分 GPU/CPU 上下文**：在编译期保证安全
12. **PDQ GPU/CPU 精度统一**：统一 f32/f64 精度策略
