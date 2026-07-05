# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-20
**Updated:** 2026-06-16
**Version:** 0.3.0
**Branch:** master2

## 项目目标

### 长期目标

**提供与 GPU 类型无关的 GPU 加速能力。**

- 上层算法通过稳定抽象访问计算资源，不感知具体 GPU 厂商、驱动、图形 API（Vulkan / Metal / DX12 / WebGPU）的差异
- 通过 wgpu 间接获得跨平台、跨后端的统一 GPU 计算入口，业务层只面向 `GpuContext` / `GpuBuffer` / `ComputePipeline` 抽象，而非 `wgpu::*` 具体类型
- 新增算法无需关心适配器选择、管线缓存、缓冲区池、降级等基础设施，只声明 WGSL 着色器与调度逻辑
- 能力层（`src/*.rs`）封装所有 wgpu 细节，是承载"GPU 类型无关"这一长期目标的核心资产

### 短期目标（当前阶段重点）

**支持图像感知哈希及距离比较（相似度比较）的 GPU 批量并行加速。**

聚焦两条主线：

1. **图像感知哈希** — 6 种算法（Mean / Median / Gradient / Block / DoubleGradient / VertGradient），64~4096-bit，支持批量图像输入与零拷贝 GPU 流水线（blur → resize → hash）
2. **距离比较 / 相似度比较** — GPU 批量汉明距离矩阵、最近邻搜索，支持 64-bit 到变长哈希（HashBytes），并提供端到端 `GpuImageMatcher`（图像 → 哈希 → 匹配）一站式 API

衡量短期目标达成的关键指标：

- 批量并行：大批量输入能自动分块、合并提交、单次 poll，充分填满 GPU 并行度
- 正确性：GPU 与 CPU 降级实现结果一致，跨哈希位宽行为一致
- 端到端可用：调用方无需手写 WGSL 或管理缓冲区即可完成"图像 → 相似图列表"

### 决策原则

**所有架构决策、功能扩展、依赖引入，必须先回答两个问题：**

1. 这是否服务于短期目标（图像感知哈希 + 距离比较的 GPU 批量并行）？
2. 这是否保持或推进了长期目标（GPU 类型无关的抽象）？

二者发生冲突时，短期目标是当前阶段优先项，但不得破坏能力层的 GPU 无关抽象。

### 明确边界：GPU 优先，CPU 降级

#### GPU 层（主要实现）

- ✅ 使用 wgpu 作为跨平台 GPU 计算后端（承载长期目标"GPU 类型无关"的抽象基础）
- ✅ 编写并维护 WGSL 着色器实现核心算法
- ✅ 实现 GPU 缓冲区管理（GpuBuffer、BufferPool 等）
- ✅ 实现 GPU 管线编译与 dispatch
- ✅ 实现 GPU 批量提交（GpuBatchSubmitter 等）
- ✅ 支持 GPU 硬件适配器自动选择
- ✅ 实现 GPU 卷积/高斯模糊/缩放等图像预处理管线（短期目标支撑能力）
- ✅ 实现 GPU 汉明距离矩阵和最近邻搜索（hamming.wgsl）（短期目标核心）
- ✅ 支持变长哈希（64-bit 到 4096-bit）的 GPU 匹配（短期目标核心）

#### CPU 层（降级方案）

- ✅ 当 GPU 初始化失败时自动降级到 CPU 实现
- ✅ 提供与 GPU 层一致的 API 接口
- ✅ CPU 实现作为可选 feature 编译
- ✅ 性能优先使用 GPU，正确性保证 CPU 降级可用

现有代码中涉及 GPU 的部分（`context.rs`、`buffer.rs`、`buffer_pool.rs`、`pipeline.rs`、`batch.rs`、WGSL 着色器文件）属于**核心资产**，应持续优化 GPU 性能并完善 CPU 降级路径。

### 与短期目标的关系（模块归类）

| 模块 | 与短期目标关系 | 说明 |
|------|----------------|------|
| `phasher.rs` / `phasher_pipeline.rs` / `phasher_util.rs` | 🎯 核心 | 图像感知哈希 GPU 批量流水线 |
| `gpu_matcher.rs` / `hamming.wgsl` | 🎯 核心 | 距离比较 GPU 批量加速 |
| `gpu_image_matcher.rs` | 🎯 核心 | 端到端图像相似度 API |
| `matcher*.rs` / `bktree*.rs` / `dihedral.rs` / `hash_bytes.rs` | 🔗 支撑 | 匹配策略、BK-tree、二面体、变长哈希类型 |
| `convolution.rs` / `gaussian_blur.rs` / `gpu_resize.rs` | 🔗 支撑 | 图像预处理（哈希前置流水线） |
| `hash_common.rs` | 🔗 支撑 | 感知哈希公共逻辑/trait/宏 |
| `context.rs` / `pipeline.rs` / `batch.rs` / `buffer*.rs` | 🏗️ 能力层 | GPU 类型无关抽象的承载者（长期目标核心） |
| `sha256.rs` / `sha256_cpu.rs` | ⚪ 范围外 | SHA-256 并行哈希，非短期目标，作为参考实现保留 |
| `pdq_hash.rs` | ⚪ 范围外 | PDQ 哈希（feature-gated），非短期目标核心 |

## OVERVIEW

GPGPU-tool: 跨平台 GPU 加速算法库，**长期目标**为提供与 GPU 类型无关的 GPU 加速能力，**短期目标**为支持图像感知哈希及距离比较（相似度比较）的 GPU 批量并行加速。当 GPU 不可用时，自动降级到 CPU 实现。Rust 2021 edition。**GPU 优先，CPU 降级。**

## STRUCTURE

```
GPGPU-tool/
├── src/                    # Library code (no main.rs - lib crate only)
│   ├── lib.rs              # Public API exports (~40 re-exports)
│   ├── context.rs          # GpuContext: device/queue/pipeline cache
│   ├── buffer.rs           # GpuBuffer: CPU-GPU data transfer
│   ├── buffer_pool.rs      # BufferPool: buffer reuse by size tier
│   ├── pipeline.rs         # ComputePipeline: compute dispatch
│   ├── batch.rs            # GpuBatchSubmitter: async batch submit
│   ├── error.rs            # GpuError: unified error type
│   ├── pixel_pack.rs       # u8↔u32 像素打包/解包工具
│   ├── poll_counter.rs     # GPU poll 计数器
│   ├── pipeline_builder.rs # 声明式管线构建器
│   ├── backend_dispatcher.rs # GPU/CPU 后端调度 trait
│   └── tasks/              # Business-layer algorithms + WGSL shaders
│       ├── sha256.rs/.wgsl           # SHA-256 GPU 并行哈希
│       ├── sha256_cpu.rs             # SHA-256 CPU 降级
│       ├── sha256_chained.wgsl       # SHA-256 链式计算着色器
│       ├── phasher.rs                # 感知哈希编排器（6 种算法）
│       ├── phasher_pipeline.rs       # 感知哈希 GPU 管线（分块/双缓冲/阈值计算）
│       ├── phasher_util.rs           # 感知哈希上传工具函数
│       ├── phasher_cpu.rs            # 感知哈希 CPU 降级
│       ├── hash_common.rs            # 感知哈希公共逻辑、宏、trait
│       ├── mean_hash.rs/.wgsl        # Mean Hash
│       ├── median_hash.rs/.wgsl      # Median Hash
│       ├── block_hash.rs/.wgsl       # Block Hash
│       ├── gradient_hash.rs/.wgsl    # Gradient Hash
│       ├── double_gradient_hash.rs/.wgsl  # Double Gradient Hash
│       ├── vert_gradient_hash.rs/.wgsl    # Vertical Gradient Hash
│       ├── convolution.rs/.wgsl      # GPU 2D 卷积（可分离/不可分离）
│       ├── gaussian_blur.rs          # GPU 高斯模糊（基于可分离卷积）
│       ├── gpu_resize.rs/.wgsl       # GPU 图像缩放
│       ├── dihedral.rs               # 二面体变换（D4 群，64/256/1024/4096-bit）
│       ├── matcher.rs                # 64-bit 哈希匹配策略（线性/BK-tree/责任链/门面）
│       ├── matcher_bytes.rs          # 变长哈希匹配策略（HashBytes 版本）
│       ├── bktree.rs                 # BK-tree 近似最近邻搜索（64-bit）
│       ├── bktree_bytes.rs           # BK-tree 变长哈希版本
│       ├── hash_bytes.rs             # 变长哈希类型 HashBytes
│       ├── hamming.wgsl              # GPU 汉明距离着色器（距离矩阵+最近邻+大哈希管线）
│       ├── gpu_matcher.rs            # GPU 加速汉明距离匹配器（64-bit + 变长）
│       ├── gpu_image_matcher.rs      # 端到端 GPU 图像匹配器
│       ├── pdq_hash.rs/.wgsl         # PDQ 哈希（DCT 频域，feature-gated）
│       └── mod.rs                    # 模块导出
├── tests/                  # Integration tests (23 test files)
├── benches/                # Criterion benchmarks (harness=false)
├── examples/               # Usage demo (demo.rs)
├── openspec/               # Design docs & change management
└── Cargo.toml              # Feature flags: image, cpu-fallback, pdq
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| GPU init/dispatch | `src/context.rs` | Sync (`new_sync`) and async (`new`) creation，失败时自动降级 |
| Buffer management | `src/buffer.rs`, `src/buffer_pool.rs` | GPU Pool tiers by size，CPU 模式使用 Vec 替代 |
| Pipeline caching | `src/context.rs` → `PipelineCache` | fxhash-based, auto-compile，仅 GPU 模式使用 |
| Batch submit | `src/batch.rs` | Async submit, wait_all pattern，CPU 模式同步执行 |
| Pixel pack/unpack | `src/pixel_pack.rs` | u8↔u32 打包工具，GPU 着色器需要 u32 对齐 |
| Pipeline builder | `src/pipeline_builder.rs` | 声明式管线构建器，链式 GPU 处理步骤 |
| Backend dispatch | `src/backend_dispatcher.rs` | GPU/CPU 后端选择 trait |
| SHA-256 GPU | `src/tasks/sha256.rs` + `.wgsl` | GPU 主要实现 |
| SHA-256 CPU | `src/tasks/sha256_cpu.rs` | CPU 降级实现（feature-gated: `cpu-fallback`） |
| Image hashing GPU | `src/tasks/{mean,median,block,gradient,double_gradient,vert_gradient}_hash.rs` | Thin wrappers via macros + `phasher.rs` |
| Image hashing CPU | `src/tasks/phasher_cpu.rs` | CPU 降级实现（feature-gated: `cpu-fallback`） |
| Hash common utils | `src/tasks/hash_common.rs` | Shared traits, macros, `HashSize`, `PerceptualHashComputer` |
| Convolution | `src/tasks/convolution.rs` + `.wgsl` | 2D/可分离卷积，Storage 绑定 |
| Gaussian blur | `src/tasks/gaussian_blur.rs` | 基于 `GpuConvolution` 的可分离高斯模糊 |
| GPU resize | `src/tasks/gpu_resize.rs` + `resize.wgsl` | 双线性插值缩放，支持零拷贝流水线 |
| Dihedral transforms | `src/tasks/dihedral.rs` | D4 群 8 种变换，支持 64/256/1024/4096-bit |
| 64-bit matching | `src/tasks/matcher.rs` | 策略模式 + 责任链 + 门面模式 |
| 变长哈希匹配 | `src/tasks/matcher_bytes.rs` | HashBytes 版本的匹配策略 |
| BK-tree (64-bit) | `src/tasks/bktree.rs` | Hamming 距离近似最近邻搜索 |
| BK-tree (变长) | `src/tasks/bktree_bytes.rs` | 变长哈希 BK-tree |
| 变长哈希类型 | `src/tasks/hash_bytes.rs` | `HashBytes`(Vec<u8>) 变长哈希封装 |
| GPU 汉明距离 | `src/tasks/hamming.wgsl` | 距离矩阵 + 最近邻 + 大哈希管线（≤4096-bit） |
| GPU matcher | `src/tasks/gpu_matcher.rs` | GPU 汉明距离匹配器（3 管线：矩阵/最近邻/大哈希） |
| GPU image matcher | `src/tasks/gpu_image_matcher.rs` | 端到端 GPU 图像匹配器 |
| PDQ hash | `src/tasks/pdq_hash.rs` + `.wgsl` | DCT 频域哈希，feature-gated: `pdq` |
| Error handling | `src/error.rs` | `GpuError` enum, thiserror，支持 GPU/CPU 统一错误 |
| Integration tests | `tests/*_test.rs` | 23 个测试文件，覆盖 GPU 和 CPU 降级路径 |
| 缓存数据测试 | `tests/cache_gpu_matcher_test.rs` | Czkawka 缓存数据驱动的 GPU 匹配验证 |
| Benchmarks | `benches/*_bench.rs` | Criterion, no harness，对比 GPU/CPU 性能 |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `GpuContext` | struct | `context.rs` | Main entry: device, queue, pipeline cache |
| `GpuBuffer` | struct | `buffer.rs` | CPU-GPU buffer transfer |
| `BufferPool` | struct | `buffer_pool.rs` | Size-tiered buffer reuse |
| `BufferPoolConfig` | struct | `buffer_pool.rs` | 缓冲区池配置参数 |
| `ComputePipeline` | struct | `pipeline.rs` | Compute pipeline creation & dispatch |
| `GpuBatchSubmitter` | struct | `batch.rs` | Async batch job submission |
| `GpuError` | enum | `error.rs` | Unified error type |
| `ComputeBackend` | enum | `lib.rs` | GPU/CPU 后端选择器 |
| `GpuPipelineBuilder` | struct | `pipeline_builder.rs` | 声明式管线构建器 |
| `BackendDispatcher` | trait | `backend_dispatcher.rs` | GPU/CPU 后端调度抽象 |
| `Sha256Computer` | struct | `tasks/sha256.rs` | GPU SHA-256 parallel hash（优先） |
| `Sha256Cpu` | struct | `tasks/sha256_cpu.rs` | CPU SHA-256 降级实现 |
| `PerceptualHasher` | struct | `tasks/phasher.rs` | 感知哈希编排器（GPU 优先，6 种算法） |
| `HashAlgorithm` | enum | `tasks/phasher.rs` | 感知哈希算法选择 |
| `PHasherCpu` | struct | `tasks/phasher_cpu.rs` | CPU 感知哈希降级实现 |
| `HashSize` | struct | `tasks/hash_common.rs` | 感知哈希网格尺寸配置 |
| `PhashParams` | struct | `tasks/hash_common.rs` | 感知哈希 uniform buffer 参数 |
| `PerceptualHashComputer` | trait | `tasks/hash_common.rs` | 感知哈希计算抽象接口 |
| `GpuConvolution` | struct | `tasks/convolution.rs` | GPU 2D 卷积计算器（可分离/不可分离） |
| `BorderMode` | enum | `tasks/convolution.rs` | 卷积边界处理模式（Zero/Clamp/Reflect） |
| `ConvMode` | enum | `tasks/convolution.rs` | 卷积模式（Full2D/Separable） |
| `GpuGaussianBlur` | struct | `tasks/gaussian_blur.rs` | GPU 高斯模糊（基于 GpuConvolution） |
| `GpuResize` | struct | `tasks/gpu_resize.rs` | GPU 图像缩放器 |
| `GpuResizeConfig` | struct | `tasks/gpu_resize.rs` | GPU 缩放器配置参数 |
| `HashBytes` | struct | `tasks/hash_bytes.rs` | 变长哈希类型（Vec<u8> 封装） |
| `DihedralHashes64` | struct | `tasks/dihedral.rs` | 64-bit 哈希的二面体变换结果（8 种变体） |
| `DihedralHashes256` | struct | `tasks/dihedral.rs` | 256-bit 哈希的二面体变换结果（8 种变体） |
| `DihedralHashes1024` | struct | `tasks/dihedral.rs` | 1024-bit 哈希的二面体变换结果（8 种变体） |
| `DihedralHashes4096` | struct | `tasks/dihedral.rs` | 4096-bit 哈希的二面体变换结果（8 种变体） |
| `HashMatcher` | trait | `tasks/matcher.rs` | 64-bit 哈希匹配策略抽象接口 |
| `MatchResult` | struct | `tasks/matcher.rs` | 64-bit 匹配结果（hash + distance） |
| `LinearScanMatcher` | struct | `tasks/matcher.rs` | 精确线性扫描匹配器，100% 召回率 |
| `BkTreeMatcher` | struct | `tasks/matcher.rs` | BK-tree 适配器匹配器 |
| `ChainedMatcher` | struct | `tasks/matcher.rs` | 责任链组合匹配器 |
| `ChainStrategy` | enum | `tasks/matcher.rs` | 责任链策略（Union/FirstHit） |
| `HashMatcherFacade` | struct | `tasks/matcher.rs` | 统一门面，封装策略选择和二面体变换增强 |
| `HashMatcherBytes` | trait | `tasks/matcher_bytes.rs` | 变长哈希匹配策略抽象接口 |
| `MatchResultBytes` | struct | `tasks/matcher_bytes.rs` | 变长哈希匹配结果 |
| `LinearScanMatcherBytes` | struct | `tasks/matcher_bytes.rs` | 变长哈希线性扫描匹配器 |
| `BkTreeMatcherBytes` | struct | `tasks/matcher_bytes.rs` | 变长哈希 BK-tree 匹配器 |
| `HashMatcherFacadeBytes` | struct | `tasks/matcher_bytes.rs` | 变长哈希统一门面 |
| `BkTree` | struct | `tasks/bktree.rs` | BK-tree 近似最近邻搜索（64-bit） |
| `BkTreeBytes` | struct | `tasks/bktree_bytes.rs` | 变长哈希 BK-tree |
| `GpuHashMatcher` | struct | `tasks/gpu_matcher.rs` | GPU 汉明距离计算器（64-bit，3 管线） |
| `GpuHashMatcherFacade` | struct | `tasks/gpu_matcher.rs` | 64-bit GPU 加速匹配器门面 |
| `GpuHashMatcherBytes` | struct | `tasks/gpu_matcher.rs` | 变长哈希 GPU 汉明距离计算器 |
| `GpuHashMatcherFacadeBytes` | struct | `tasks/gpu_matcher.rs` | 变长哈希 GPU 匹配器门面 |
| `GpuImageMatcher` | struct | `tasks/gpu_image_matcher.rs` | 端到端 GPU 图像匹配器 |
| `PdqHashGpu` | struct | `tasks/pdq_hash.rs` | PDQ GPU 计算器（feature-gated: `pdq`） |
| `PdqHashCpu` | struct | `tasks/pdq_hash.rs` | PDQ CPU 计算器（feature-gated: `pdq`） |
| `PdqHashResult` | struct | `tasks/pdq_hash.rs` | PDQ 哈希结果（256-bit hash + quality） |

## CONVENTIONS

- **Rust 2021 edition**, wgpu v24
- **No `main.rs`** - library crate only, examples via `cargo run --example`
- **WGSL shaders** co-located with Rust wrappers in `src/tasks/` (`.wgsl` alongside `.rs`)
- **CPU 降级实现** 与 GPU 实现同目录，以 `_cpu.rs` 后缀区分
- **Feature flag**: `image` feature gates image-related deps (optional in prod, required in dev)
- **Feature flag**: `cpu-fallback` 控制是否编译 CPU 降级实现（默认开启，依赖 `sha2` crate）
- **Feature flag**: `pdq` 控制是否编译 PDQ 哈希模块（默认关闭）
- **Benchmarks**: `harness = false` in Cargo.toml - custom Criterion setup
- **Error type**: `thiserror` for `GpuError`,统一处理 GPU/CPU 错误
- **Async pattern**: `pollster::block_on` for sync wrappers，CPU 降级使用同步执行
- **像素对齐**: GPU 着色器使用 u32 per pixel，通过 `pixel_pack.rs` 做 u8↔u32 转换
- **变长哈希**: `HashBytes`(Vec<u8>) 支持任意位宽，`hash_bytes_to_u32` 自动 u32 对齐
- **GPU 匹配器管线选择**: u32_per_hash ≤ 32 用标准管线(workgroup=256)，> 32 用大哈希管线(workgroup=32)

## ANTI-PATTERNS (THIS PROJECT)

- **No `unsafe`** blocks found - pure safe Rust
- **No `main.rs`** - do not add binary entry point without discussion
- **WGSL inline** - shaders are separate `.wgsl` files, NOT inline strings
- **No TODO/FIXME/HACK** markers - clean codebase

## UNIQUE STYLES

- **Pipeline cache** uses custom `fxhash` (not std hash) for shader source hashing
- **BufferPool** uses size-tiered bins (not exact match)
- **Batch submitter** uses `submit()` + `wait_all()` pattern (not futures)
- **Perceptual hash** modules are thin 14-line wrappers delegating to `phasher.rs` + WGSL
- **Convolution** supports both Full2D and Separable modes, separable uses single-encoder two-pass dispatch
- **Matcher** uses Strategy + Chain of Responsibility + Facade pattern combination
- **Dihedral** provides CPU-only bit matrix transforms for rotation-invariant matching
- **GPU matcher** uses 3 pipelines: distance_matrix + nearest_neighbor + nearest_neighbor_large
- **变长哈希** 通过 `HashBytes` + `*Bytes` 后缀的匹配器/BK-tree 实现统一抽象

## COMMANDS

```bash
# Build
cargo build
cargo build --release
cargo build --features image
cargo build --features pdq

# Run example
cargo run --example demo

# Test
cargo test
cargo test --features pdq

# Benchmark (all)
cargo bench

# Benchmark (specific)
cargo bench --bench sha256_bench

# Lint
cargo clippy
```

## NOTES

- `target/` contains build artifacts - excluded from all analysis
- GPU adapter selection: `HighPerformance` preference, all backends
- wgpu backends: Vulkan, Metal, DX12, WebGPU (auto-selected)
- **GPU 初始化失败时自动降级**: 当 wgpu 初始化失败（如无 GPU 驱动、WebAssembly 环境等），自动切换到 CPU 实现
- SHA-256 is the reference implementation - new algorithms should follow its pattern
- Image hash algorithms share common infrastructure via `phasher.rs`
- **CPU 降级设计原则**: CPU 实现保持与 GPU 实现相同的 API 签名，调用方无感知切换
- **⚠️ convolution.wgsl 已修复**: `ConvParams` 参数从 Uniform 绑定改为 Storage 绑定（`var<storage, read>`），解决了 `kernel` 数组不满足 WGSL Uniform 16 字节对齐要求的问题
- **PDQ 哈希通过 `pdq` feature gate 控制**: 默认不编译，需 `cargo build --features pdq` 启用
- **零拷贝流水线**: `GpuResize`、`GpuConvolution`、`GpuGaussianBlur` 均提供 `_gpu` 后缀方法，返回 `GpuBuffer` 而非 `Vec<u8>`，支持 GPU 管线间零拷贝传递
- **GPU 汉明距离着色器** (`hamming.wgsl`): 3 个入口点 — `compute_hamming_matrix`(距离矩阵)、`find_nearest_neighbor`(最近邻≤1024-bit)、`find_nearest_neighbor_large`(大哈希≤4096-bit)
- **4096-bit 哈希支持**: `DihedralHashes4096`(64×64 位矩阵)、`find_nearest_neighbor_large` 管线(workgroup_size=32)
- **缓存数据验证**: `tests/cache_gpu_matcher_test.rs` 使用 Czkawka 缓存数据（20000 条 1024-bit 哈希），GPU 最近邻 2.2s vs CPU 推算 1711s（778x 加速）
- **Mean/Median Hash 优化**: CPU 预计算阈值（均值/中位数），通过 `f32::to_bits()` + 着色器 `bitcast<f32>()` 传入，消除 O(N²) 循环

## 架构决策记录

### ADR-001: GPU 优先，CPU 降级

**状态**: 已接受

**背景**: 项目从纯 CPU 算法库转型为 GPU 加速库

**决策**:
1. 优先使用 wgpu 实现 GPU 加速
2. 当 GPU 不可用时，自动降级到 CPU 实现
3. 对外暴露统一 API，内部处理后端选择

**后果**:
- 需要维护两套实现（GPU + CPU）
- 增加代码复杂度，但提供更好的性能和无 GPU 环境的兼容性
- 测试需要覆盖 GPU 和 CPU 两种路径

### ADR-002: 卷积模块设计

**状态**: 已接受

**背景**: 感知哈希和 PDQ 哈希都需要图像预处理（高斯模糊、DCT 等），存在重复的 GPU dispatch 逻辑

**决策**:
1. 抽取独立的 `GpuConvolution` 模块，支持 Full2D 和 Separable 两种卷积模式
2. `GpuGaussianBlur` 作为 `GpuConvolution` 的上层封装，生成高斯核后调用可分离卷积
3. 所有卷积操作使用 `pixel_pack` 做 u8↔u32 对齐
4. Separable 模式使用单 encoder 两趟 dispatch，减少 GPU 同步开销

**后果**:
- 卷积逻辑集中管理，避免各算法重复实现
- 高斯模糊仅 151 行，职责清晰
- `ConvParams` 已改为 Storage 绑定，解决对齐问题
- 为未来新增图像预处理算法（如 Sobel、Laplacian）提供基础

### ADR-003: 二面体变换与匹配器模式

**状态**: 已接受

**背景**: 图像可能经过旋转/翻转，需要不重新计算哈希即可匹配变换后的图像

**决策**:
1. `dihedral.rs` 纯 CPU 实现二面体变换（D4 群 8 种），不依赖 GPU
2. `matcher.rs` 使用策略模式 + 责任链模式 + 门面模式
3. `HashMatcherFacade` 可选启用二面体变换增强，对查询哈希的 8 种变体分别匹配
4. 支持 64/256/1024/4096-bit 四种哈希尺寸

**后果**:
- 匹配策略可灵活组合，新增策略只需实现 trait
- 二面体变换与匹配逻辑解耦，可独立使用
- 门面模式简化调用方代码，一行即可完成策略选择 + 二面体增强

### ADR-004: PDQ 哈希 Feature Gate

**状态**: 已接受

**背景**: PDQ（Photo DNA Quality）哈希基于 DCT 频域变换，算法复杂度高于其他感知哈希，且使用场景相对独立

**决策**:
1. PDQ 哈希通过 `pdq` feature flag 控制，默认不编译
2. 同时提供 GPU（`PdqHashGpu`，两趟 DCT-II）和 CPU（`PdqHashCpu`）实现
3. GPU 实现完成 DCT 变换，CPU 完成中值量化和哈希打包（混合流水线）
4. `PdqHashGpu` 实现 `PerceptualHashComputer` trait，可集成到 `PerceptualHasher` 编排器

**后果**:
- 默认构建不包含 PDQ，减少编译时间和二进制体积
- 启用 `pdq` feature 后可获得 256-bit 高精度频域哈希
- 混合流水线（GPU DCT + CPU 量化）平衡了性能和实现复杂度

### ADR-005: 变长哈希与 GPU 匹配器

**状态**: 已接受

**背景**: 原 64-bit 匹配器仅支持 u64 哈希，无法处理 256/1024/4096-bit 哈希，且 GPU 汉明距离计算需要按 u32 对齐

**决策**:
1. 引入 `HashBytes`(Vec<u8>) 变长哈希类型，支持任意位宽
2. 创建 `*Bytes` 后缀的匹配器/BK-tree，与 64-bit 版本 API 一致
3. GPU 匹配器使用 3 管线架构：距离矩阵 + 标准最近邻 + 大哈希最近邻
4. `hash_bytes_to_u32` 自动将 HashBytes 转为 u32 对齐数组
5. u32_per_hash > 32 时自动切换到大哈希管线（workgroup_size=32）

**后果**:
- 支持 64-bit 到 4096-bit 哈希的 GPU 并行匹配
- 20000 条 1024-bit 哈希 GPU 最近邻 2.2s，CPU 推算 1711s（778x 加速）
- 需维护 64-bit 和变长两套匹配器 API

### ADR-006: 明确长期目标与短期目标（2026-06-16）

**状态**: 已接受

**背景**: 项目同时承载 SHA-256、感知哈希、PDQ、BK-tree 等多类算法，目标发散。需明确长期方向（GPU 类型无关的加速能力）与当前阶段重点（图像感知哈希 + 距离比较的 GPU 批量并行），为架构演进和资源分配提供依据。

**决策**:
1. **长期目标**：通过能力层（`context.rs` / `pipeline.rs` / `batch.rs` / `buffer*.rs`）封装 wgpu，使上层算法获得与 GPU 类型无关的加速能力。业务层只面向 `GpuContext` / `GpuBuffer` / `ComputePipeline` 抽象。
2. **短期目标**：聚焦图像感知哈希与距离比较的 GPU 批量并行加速。`phasher*`、`gpu_matcher*`、`gpu_image_matcher.rs` 为核心模块。
3. 明确模块归类（核心 / 支撑 / 能力层 / 范围外）。SHA-256 与 PDQ 列为"范围外"，作为参考实现保留，不作为当前阶段投入重点。
4. 决策原则升级为双问题检验：是否服务短期目标 + 是否保持 GPU 类型无关抽象。

**后果**:
- 后续 PR 评估有了明确依据：偏离短期目标的特性需说明理由
- 能力层 wgpu 封装质量被纳入长期目标的维护红线
- SHA-256 / PDQ 仍保留但不再主动扩展，避免目标漂移
