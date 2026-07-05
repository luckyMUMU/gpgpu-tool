# 零拷贝流水线吞吐量优化 Spec

## Why

当前零拷贝流水线（`compute_hashes_zero_copy_pipelined`）已实现 CPU/GPU 双缓冲重叠，但 500 张图片测试中吞吐量仅 ~15 图/s，瓶颈分析如下：

1. **CPU 图像解码单线程串行**：CPU 工作线程逐张 `image::open()` → `to_rgba8()` → `rgba_to_grayscale_cpu()`，未利用多核 CPU 并行能力。在 Release 模式下单张解码 ~65ms，是流水线主要瓶颈。
2. **RGBA→灰度转换为逐像素标量循环**：`rgba_to_grayscale_cpu()`（[phasher.rs:706-718](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L706-L718)）对每个像素执行 3 次乘法 + 1 次移位，无 SIMD 向量化，4K 图像需 ~4M 次迭代。
3. **u8→u32 像素打包在 CPU 端执行**：`upload_image_to_gpu()`（[phasher_util.rs:18-43](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher_util.rs#L18-L43)）调用 `pack_u8_to_u32()` 将灰度 u8 扩展为 u32 后上传，CPU 端产生 4× 内存膨胀 + 逐元素拷贝开销，且上传数据量 4× 于实际灰度数据。
4. **GPU 上传与 GPU 计算未重叠**：当前流水线阶段 1 完成所有子批上传后，阶段 2 才统一执行 resize + hash。理想情况下，GPU 处理第 N 批时应同时上传第 N+1 批。

本规范的目标是：**通过四项优化将零拷贝流水线吞吐量提升 2-4×**，使 10K 图像处理时间从 ~11 分钟降至 ~3-5 分钟。

## What Changes

### 优化 1：rayon 多线程并行图像解码（P0）

- **修改 `compute_hashes_zero_copy_pipelined` 和 `compute_hashes_zero_copy_lanczos3_pipelined`**：CPU 工作线程内部使用 rayon 并行迭代处理子批图像，将 `for i in 0..image_count` 串行循环替换为 `rayon::par_iter` 并行解码 + 灰度转换
- **新增 `parallel-cpu` feature 作为前置依赖**：`czkawka-compat` feature 隐含启用 `parallel-cpu`，确保 rayon 可用
- **保持子批背压机制不变**：rayon 并行处理 `sub_batch_size` 张图像后，仍通过 `sync_channel(1)` 向主线程发送子批数据，保持 CPU 最多领先 GPU 一个子批

### 优化 2：SIMD 加速 RGBA→灰度转换（P1）

- **新增 `rgba_to_grayscale_simd()` 函数**：使用 `std::simd`（Rust nightly）或 `wide` crate（stable）实现 SIMD 向量化灰度转换，一次处理 8/16/16 个像素（u8x16/u32x8）
- **运行时降级**：检测 CPU 是否支持 AVX2/SSE4.2，不支持时回退到标量实现
- **修改 `rgba_to_grayscale_cpu()` 为分发器**：根据 feature flag 和运行时检测选择 SIMD 或标量路径
- **保持结果一致性**：SIMD 实现使用相同公式 `(R*77 + G*150 + B*29) >> 8`，结果与标量实现逐位一致

### 优化 3：GPU 端 u8→u32 像素打包（P1）

- **扩展 `color_convert.wgsl`**：新增 `pack_u8_to_u32` 入口点，在 GPU 端完成 u8→u32 打包
- **新增 `upload_raw_to_gpu()` 函数**：直接上传原始 u8 灰度数据到 GPU（不经过 CPU 端打包），数据量减少 4×
- **修改 `preprocess_gpu()` 路径**：零拷贝模式下上传原始 u8 数据后，GPU 端 dispatch 打包着色器生成 u32 缓冲区，再进入 resize + hash 管线
- **保持非零拷贝路径不变**：传统批量模式仍使用 CPU 端打包（小批量时 CPU 打包更快）

### 优化 4：异步 GPU 流水线（上传与计算重叠）（P2）

- **重构 `compute_hashes_zero_copy_pipelined` 为三阶段重叠流水线**：
  - 阶段 A：CPU 线程并行解码 + 灰度转换（rayon）
  - 阶段 B：主线程上传子批 N 到 GPU（`queue.write_buffer`）
  - 阶段 C：GPU 执行子批 N-1 的 resize + hash（dispatch）
  - B 和 C 在不同 CommandEncoder 上并行执行
- **使用双 CommandEncoder 交替提交**：借鉴 `DoubleBufferStaging` 乒乓模式，一个 encoder 负责上传，另一个负责计算
- **子批间无 GPU 同步屏障**：子批 N 的上传与子批 N-1 的计算在同一 `queue.submit()` 中编码，wgpu 驱动自动管理依赖

## Impact

- **Affected specs**:
  - `czkawka-gpu-integration`（零拷贝流水线是其 P1 B3 全零拷贝管线的延伸优化）
  - `gpu-pipeline-optimization`（双缓冲流水线是本次优化的基础）
- **Affected code**:
  - `src/czkawka_compat.rs`（[czkawka_compat.rs:561-661](file:///d:/Code/AI/wgpu-tool/src/czkawka_compat.rs#L561-L661) `compute_hashes_zero_copy_pipelined`；[czkawka_compat.rs:674-767](file:///d:/Code/AI/wgpu-tool/src/czkawka_compat.rs#L674-L767) `compute_hashes_zero_copy_lanczos3_pipelined`）— 重构流水线为 rayon 并行 + 异步重叠
  - `src/tasks/phasher.rs`（[phasher.rs:706-718](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L706-L718) `rgba_to_grayscale_cpu`）— 新增 SIMD 路径
  - `src/tasks/phasher_util.rs`（[phasher_util.rs:18-43](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher_util.rs#L18-L43) `upload_image_to_gpu`）— 新增 `upload_raw_to_gpu` 路径
  - `src/tasks/color_convert.wgsl`（[color_convert.wgsl:30-46](file:///d:/Code/AI/wgpu-tool/src/tasks/color_convert.wgsl#L30-L46) `rgba_to_grayscale`）— 新增 `pack_u8_to_u32` 入口点
  - `src/pixel_pack.rs`（[pixel_pack.rs:6-48](file:///d:/Code/AI/wgpu-tool/src/pixel_pack.rs#L6-L48)）— 新增 SIMD 打包函数（可选）
  - `Cargo.toml`（[Cargo.toml:22](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L22) `rayon`；[Cargo.toml:27](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L27) `parallel-cpu`）— `czkawka-compat` 隐含 `parallel-cpu`
  - `examples/similar_images.rs`（[similar_images.rs:547-560](file:///d:/Code/AI/wgpu-tool/examples/similar_images.rs#L547-L560)）— 新增 `--threads` 参数控制 rayon 线程数
  - 新增 `benches/pipeline_throughput_bench.rs` — 流水线吞吐量基准测试

## ADDED Requirements

### Requirement: rayon 并行图像解码

系统 SHALL 在零拷贝流水线的 CPU 工作线程中使用 rayon 并行迭代处理子批图像解码 + RGBA→灰度转换，充分利用多核 CPU。

`compute_hashes_zero_copy_pipelined` 和 `compute_hashes_zero_copy_lanczos3_pipelined` 的 CPU 工作线程 SHALL：
- 收集 `sub_batch_size` 张图像的索引
- 使用 `rayon::par_iter` 并行执行 `loader(i)` + `rgba_to_grayscale_cpu()` / Lanczos3 缩放
- 并行完成后按原始顺序排列结果，通过 `sync_channel(1)` 发送给主线程

`czkawka-compat` feature SHALL 隐含启用 `parallel-cpu` feature，确保 rayon 可用。

#### Scenario: 8 核 CPU 上 16 张图像子批并行解码

- **WHEN** 在 8 核 CPU 上使用 `sub_batch_size=16` 调用 `compute_hashes_zero_copy_pipelined`
- **THEN** CPU 工作线程使用 8 个 rayon 工作线程并行解码 16 张图像
- **AND** 理论加速比 ~4-6×（受 I/O 和内存带宽限制，非线性扩展）
- **AND** 结果与串行版本逐位一致

#### Scenario: rayon 不可用时降级

- **WHEN** `parallel-cpu` feature 未启用但 `czkawka-compat` 已启用
- **THEN** 编译错误提示 `czkawka-compat` 需要 `parallel-cpu`

### Requirement: SIMD 加速 RGBA→灰度转换

系统 SHALL 提供 `rgba_to_grayscale_simd()` 函数，使用 SIMD 指令并行处理多个像素的 RGBA→灰度转换。

`rgba_to_grayscale_cpu()` SHALL 成为分发器：
- 如果 `simd` feature 启用且 CPU 支持 AVX2/SSE4.2 → 调用 SIMD 路径
- 否则 → 调用标量路径（当前实现）

SIMD 路径 SHALL 使用与标量路径完全相同的公式 `(R*77 + G*150 + B*29) >> 8`，结果逐位一致。

#### Scenario: 4K 图像 SIMD 转换

- **WHEN** 对 3840×2160 RGBA 图像调用 `rgba_to_grayscale_simd()`
- **THEN** 使用 u8x16 SIMD 通道一次处理 16 个像素
- **AND** 吞吐量比标量路径提升 4-8×
- **AND** 输出结果与 `rgba_to_grayscale_cpu()` 标量路径逐字节一致

#### Scenario: 不支持 SIMD 的 CPU

- **WHEN** CPU 不支持 AVX2/SSE4.2 且 `simd` feature 启用
- **THEN** 运行时自动降级到标量路径
- **AND** 功能正常，仅性能不提升

### Requirement: GPU 端 u8→u32 像素打包

系统 SHALL 在 `color_convert.wgsl` 中新增 `pack_u8_to_u32` 入口点，在 GPU 端完成 u8 灰度像素到 u32 的打包。

`phasher_util.rs` SHALL 新增 `upload_raw_to_gpu()` 函数：
- 直接上传原始 u8 灰度数据到 GPU 存储 buffer（不经过 CPU 端 `pack_u8_to_u32`）
- 上传数据量 = `pixel_count` 字节（而非 `pixel_count * 4` 字节）
- 上传后 dispatch `pack_u8_to_u32` 着色器，生成 u32 格式的 GPU buffer

零拷贝流水线 SHALL 使用 `upload_raw_to_gpu()` 替代 `upload_image_to_gpu()`，减少 4× 上传数据量。

传统批量模式（`compute()`、`compute_from_rgba()`）SHALL 保持使用 CPU 端打包不变（小批量时 CPU 打包 + 一次 write_buffer 更高效）。

#### Scenario: 零拷贝模式 4K 图像上传

- **WHEN** 零拷贝模式上传 3840×2160 灰度图像到 GPU
- **THEN** 上传数据量 = 8,294,400 字节（而非 33,177,600 字节）
- **AND** GPU 端 dispatch 打包着色器生成 u32 buffer
- **AND** 后续 resize + hash 管线接收的 u32 buffer 与 CPU 打包版本一致

#### Scenario: 打包着色器正确性

- **WHEN** GPU 端打包着色器处理 `[128, 200, 50, 255]` 四个 u8 像素
- **THEN** 输出 `[128u32, 200u32, 50u32, 255u32]`
- **AND** 与 CPU `pack_u8_to_u32()` 结果一致

### Requirement: 异步 GPU 流水线（上传与计算重叠）

系统 SHALL 重构 `compute_hashes_zero_copy_pipelined` 为三阶段重叠流水线，实现 GPU 上传与 GPU 计算的并行执行。

流水线 SHALL 按以下模式运行：
- **子批 0**：上传 → [等待]
- **子批 1**：上传 + 计算子批 0
- **子批 2**：上传 + 计算子批 1
- ...
- **最后一批**：[等待] + 计算子批 N-1

上传和计算 SHALL 在不同的 CommandEncoder 上编码，通过两次 `queue.submit()` 交替提交，wgpu 驱动自动管理 GPU 端依赖。

主线程 SHALL 在 GPU 计算子批 N-1 时同时接收 CPU 线程的子批 N 数据并上传。

#### Scenario: 流水线重叠执行

- **WHEN** 处理 32 张图像，`sub_batch_size=8`（4 个子批）
- **THEN** 子批 1 的 GPU 上传与子批 0 的 GPU resize+hash 并行执行
- **AND** 总 GPU 时间 ≈ max(上传时间, 计算时间) × 子批数（而非两者之和）
- **AND** 结果与非重叠版本逐位一致

#### Scenario: 单子批降级

- **WHEN** 总图像数 ≤ `sub_batch_size`（仅 1 个子批）
- **THEN** 无重叠可执行，退化为 "上传 → 计算" 串行模式
- **AND** 功能正常，结果正确

## MODIFIED Requirements

### Requirement: `czkawka-compat` feature 依赖

`czkawka-compat` feature SHALL 隐含启用 `parallel-cpu` feature（[Cargo.toml:27](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L27)），确保 rayon 在零拷贝流水线中可用。

当前 `czkawka-compat` 仅依赖 `image` feature，修改后增加 `parallel-cpu` 依赖。

### Requirement: `rgba_to_grayscale_cpu` 分发逻辑

`rgba_to_grayscale_cpu()`（[phasher.rs:706-718](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L706-L718)）SHALL 从直接实现灰度转换改为分发器：
- `simd` feature 启用时 → 调用 `rgba_to_grayscale_simd()`
- 否则 → 调用标量实现（当前逻辑）

### Requirement: `compute_hashes_zero_copy_pipelined` 流水线结构

`compute_hashes_zero_copy_pipelined()`（[czkawka_compat.rs:561-661](file:///d:/Code/AI/wgpu-tool/src/czkawka_compat.rs#L561-L661)）SHALL 从 "阶段 1 全上传 + 阶段 2 全计算" 改为 "子批级上传/计算重叠" 模式。

当前两阶段结构 SHALL 保留为 `compute_hashes_zero_copy_batched()`（非重叠版本），供 `--no-pipeline` 参数使用。

## REMOVED Requirements

无移除项。所有优化均为新增路径或修改现有路径，不删除现有功能。

## 性能预估

基于 500 张图片测试基线（~15 图/s，Release 模式）：

| 优化项 | 预估加速比 | 累计吞吐量 | 说明 |
|--------|-----------|-----------|------|
| 基线（当前） | 1× | ~15 图/s | 单线程解码 + CPU 打包 + 串行上传/计算 |
| +rayon 并行解码 | 3-5× | ~45-75 图/s | 8 核 CPU 并行解码，I/O 仍是瓶颈 |
| +SIMD 灰度转换 | 1.2-1.5× | ~54-112 图/s | 灰度转换占总时间 ~15%，SIMD 提升 4-8× |
| +GPU 端打包 | 1.1-1.3× | ~60-146 图/s | 减少 4× 上传数据量 + CPU 打包开销 |
| +异步流水线 | 1.2-1.5× | ~72-219 图/s | 上传与计算重叠，隐藏 GPU 空闲时间 |

> 注：加速比为预估范围，实际取决于图像尺寸、CPU 核心数、GPU 型号和 I/O 带宽。10K 图像预估总时间从 ~11 分钟降至 1-2 分钟。

## 风险与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| rayon 线程与 GPU 上传竞争内存带宽 | 🟡 中 | 限制 rayon 线程数为 `num_cpus - 1`，留 1 核给主线程 GPU 上传 |
| SIMD 跨平台兼容性 | 🟡 中 | 使用 `wide` crate（stable Rust）而非 `std::simd`（nightly），运行时降级到标量 |
| GPU 端打包增加 1 次 dispatch | 🟢 低 | 打包着色器 workgroup_size=256，4K 图像仅 8K dispatch，开销 < 0.1ms |
| 异步流水线 CommandEncoder 管理 | 🟡 中 | 双 encoder 交替使用，每次 submit 后立即创建新 encoder |
| rayon + std::thread::scope 嵌套 | 🟢 低 | rayon 线程池在 scope 内创建，scope 结束后自动 join |
| 子批顺序保证 | 🟡 中 | rayon `par_iter` 后按索引排序，确保哈希顺序与图像顺序一致 |
