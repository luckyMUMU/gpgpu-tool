# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-20
**Updated:** 2026-05-24
**Commit:** 7bb56f8
**Branch:** master

## 核心目标

本项目定位为**GPU 加速算法库**（GPGPU-tool），优先使用 GPU 加速提供 SHA-256 哈希、感知图像哈希、BK-tree 相似度搜索等算法的高性能实现。当 GPU 不可用时，提供低优先级的 CPU 降级方案。

### 决策原则

**所有架构决策、功能扩展、依赖引入，必须首先回答：这是否符合"GPU 优先，CPU 降级"的目标？**
- GPU 加速是首要目标，优先利用硬件并行能力
- CPU 降级作为后备方案，保证在无 GPU 环境下的可用性
- 架构设计需兼顾两种执行路径的统一抽象

### 明确边界：GPU 优先，CPU 降级

#### GPU 层（主要实现）

- ✅ 使用 wgpu 作为跨平台 GPU 计算后端
- ✅ 编写并维护 WGSL 着色器实现核心算法
- ✅ 实现 GPU 缓冲区管理（GpuBuffer、BufferPool 等）
- ✅ 实现 GPU 管线编译与 dispatch
- ✅ 实现 GPU 批量提交（GpuBatchSubmitter 等）
- ✅ 支持 GPU 硬件适配器自动选择
- ✅ 实现 GPU 卷积/高斯模糊/缩放等图像预处理管线

#### CPU 层（降级方案）

- ✅ 当 GPU 初始化失败时自动降级到 CPU 实现
- ✅ 提供与 GPU 层一致的 API 接口
- ✅ CPU 实现作为可选 feature 编译
- ✅ 性能优先使用 GPU，正确性保证 CPU 降级可用

现有代码中涉及 GPU 的部分（`context.rs`、`buffer.rs`、`buffer_pool.rs`、`pipeline.rs`、`batch.rs`、WGSL 着色器文件）属于**核心资产**，应持续优化 GPU 性能并完善 CPU 降级路径。

## OVERVIEW

GPGPU-tool: GPU 加速算法库，优先使用 GPU 提供 SHA-256 哈希、感知图像哈希、BK-tree 相似度搜索等算法的高性能实现。当 GPU 不可用时，自动降级到 CPU 实现。Rust 2021 edition。**GPU 优先，CPU 降级。**

## STRUCTURE

```
GPGPU-tool/
├── src/                    # Library code (no main.rs - lib crate only)
│   ├── lib.rs              # Public API exports
│   ├── context.rs          # GpuContext: device/queue/pipeline cache
│   ├── buffer.rs           # GpuBuffer: CPU-GPU data transfer
│   ├── buffer_pool.rs      # BufferPool: buffer reuse by size tier
│   ├── pipeline.rs         # ComputePipeline: compute dispatch
│   ├── batch.rs            # GpuBatchSubmitter: async batch submit
│   ├── error.rs            # GpuError: unified error type
│   ├── pixel_pack.rs       # u8↔u32 像素打包/解包工具
│   └── tasks/              # Business-layer algorithms + WGSL shaders
│       ├── sha256.rs/.wgsl           # SHA-256 GPU 并行哈希
│       ├── sha256_cpu.rs             # SHA-256 CPU 降级
│       ├── sha256_chained.wgsl       # SHA-256 链式计算着色器
│       ├── phasher.rs                # 感知哈希编排器（6 种算法）
│       ├── phasher_cpu.rs            # 感知哈希 CPU 降级
│       ├── hash_common.rs            # 感知哈希公共逻辑
│       ├── mean_hash.rs/.wgsl        # Mean Hash
│       ├── median_hash.rs/.wgsl      # Median Hash
│       ├── block_hash.rs/.wgsl       # Block Hash
│       ├── gradient_hash.rs/.wgsl    # Gradient Hash
│       ├── double_gradient_hash.rs/.wgsl  # Double Gradient Hash
│       ├── vert_gradient_hash.rs/.wgsl    # Vertical Gradient Hash
│       ├── convolution.rs/.wgsl      # GPU 2D 卷积（可分离/不可分离）
│       ├── gaussian_blur.rs          # GPU 高斯模糊（基于可分离卷积）
│       ├── gpu_resize.rs             # GPU 图像缩放
│       ├── resize.wgsl               # 缩放着色器
│       ├── dihedral.rs               # 二面体变换（D4 群 8 种旋转/翻转）
│       ├── matcher.rs                # 哈希匹配策略（线性/BK-tree/责任链/门面）
│       ├── bktree.rs                 # BK-tree 近似最近邻搜索
│       ├── pdq_hash.rs/.wgsl         # PDQ 哈希（DCT 频域，feature-gated）
│       └── mod.rs                    # 模块导出
├── tests/                  # Integration tests (per-algorithm)
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
| SHA-256 GPU | `src/tasks/sha256.rs` + `.wgsl` | 668 lines，GPU 主要实现 |
| SHA-256 CPU | `src/tasks/sha256_cpu.rs` | CPU 降级实现（feature-gated: `cpu-fallback`） |
| SHA-256 chained | `src/tasks/sha256_chained.wgsl` | 链式 SHA-256 计算着色器 |
| Image hashing GPU | `src/tasks/{mean,median,block,gradient,double_gradient,vert_gradient}_hash.rs` | All 14-line wrappers + `phasher.rs` (411 lines) |
| Image hashing CPU | `src/tasks/phasher_cpu.rs` | CPU 降级实现（feature-gated: `cpu-fallback`） |
| Hash common utils | `src/tasks/hash_common.rs` | Shared perceptual hash logic, `HashSize`, `PerceptualHashComputer` trait |
| Convolution | `src/tasks/convolution.rs` + `.wgsl` | 393 lines，2D/可分离卷积，⚠️ Uniform 对齐已知问题 |
| Gaussian blur | `src/tasks/gaussian_blur.rs` | 116 lines，基于 `GpuConvolution` 的可分离高斯模糊 |
| GPU resize | `src/tasks/gpu_resize.rs` + `resize.wgsl` | 324 lines，双线性插值缩放，支持零拷贝流水线 |
| Dihedral transforms | `src/tasks/dihedral.rs` | 539 lines，D4 群 8 种变换，支持 64-bit 和 256-bit 哈希 |
| Hash matching | `src/tasks/matcher.rs` | 324 lines，策略模式 + 责任链 + 门面模式 |
| BK-tree | `src/tasks/bktree.rs` | Hamming 距离近似最近邻搜索 |
| PDQ hash | `src/tasks/pdq_hash.rs` + `.wgsl` | 307 lines，DCT 频域哈希，feature-gated: `pdq` |
| Error handling | `src/error.rs` | `GpuError` enum, thiserror，支持 GPU/CPU 统一错误 |
| Integration tests | `tests/*_test.rs` | One per algorithm，覆盖 GPU 和 CPU 降级路径 |
| Benchmarks | `benches/*_bench.rs` | Criterion, no harness，对比 GPU/CPU 性能 |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `GpuContext` | struct | `context.rs:78` | Main entry: device, queue, pipeline cache |
| `GpuBuffer` | struct | `buffer.rs:53` | CPU-GPU buffer transfer |
| `BufferPool` | struct | `buffer_pool.rs:114` | Size-tiered buffer reuse |
| `BufferPoolConfig` | struct | `buffer_pool.rs:13` | 缓冲区池配置参数 |
| `ComputePipeline` | struct | `pipeline.rs:82` | Compute pipeline creation & dispatch |
| `GpuBatchSubmitter` | struct | `batch.rs:42` | Async batch job submission |
| `GpuError` | enum | `error.rs:5` | Unified error type |
| `ComputeBackend` | enum | `lib.rs:140` | GPU/CPU 后端选择器 |
| `Sha256Computer` | struct | `tasks/sha256.rs:54` | GPU SHA-256 parallel hash（优先） |
| `Sha256Cpu` | struct | `tasks/sha256_cpu.rs:10` | CPU SHA-256 降级实现 |
| `PerceptualHasher` | struct | `tasks/phasher.rs:64` | 感知哈希编排器（GPU 优先，6 种算法） |
| `HashAlgorithm` | enum | `tasks/phasher.rs:20` | 感知哈希算法选择 |
| `PHasherCpu` | struct | `tasks/phasher_cpu.rs:12` | CPU 感知哈希降级实现 |
| `HashSize` | struct | `tasks/hash_common.rs:35` | 感知哈希网格尺寸配置 |
| `PhashParams` | struct | `tasks/hash_common.rs:13` | 感知哈希 uniform buffer 参数 |
| `PerceptualHashComputer` | trait | `tasks/hash_common.rs:111` | 感知哈希计算抽象接口 |
| `GpuConvolution` | struct | `tasks/convolution.rs:65` | GPU 2D 卷积计算器（可分离/不可分离） |
| `BorderMode` | enum | `tasks/convolution.rs:22` | 卷积边界处理模式（Zero/Clamp/Reflect） |
| `ConvMode` | enum | `tasks/convolution.rs:43` | 卷积模式（Full2D/Separable） |
| `GpuGaussianBlur` | struct | `tasks/gaussian_blur.rs:10` | GPU 高斯模糊（基于 GpuConvolution） |
| `GpuResize` | struct | `tasks/gpu_resize.rs:57` | GPU 图像缩放器 |
| `GpuResizeConfig` | struct | `tasks/gpu_resize.rs:18` | GPU 缩放器配置参数 |
| `DihedralHashes64` | struct | `tasks/dihedral.rs:216` | 64-bit 哈希的二面体变换结果（8 种变体） |
| `DihedralHashes256` | struct | `tasks/dihedral.rs:277` | 256-bit 哈希的二面体变换结果（8 种变体） |
| `HashMatcher` | trait | `tasks/matcher.rs:23` | 哈希匹配策略抽象接口 |
| `MatchResult` | struct | `tasks/matcher.rs:17` | 匹配结果（hash + distance） |
| `LinearScanMatcher` | struct | `tasks/matcher.rs:40` | 精确线性扫描匹配器，100% 召回率 |
| `BkTreeMatcher` | struct | `tasks/matcher.rs:74` | BK-tree 适配器匹配器 |
| `ChainedMatcher` | struct | `tasks/matcher.rs:99` | 责任链组合匹配器 |
| `ChainStrategy` | enum | `tasks/matcher.rs:30` | 责任链策略（Union/FirstHit） |
| `HashMatcherFacade` | struct | `tasks/matcher.rs:151` | 统一门面，封装策略选择和二面体变换增强 |
| `BkTree` | struct | `tasks/bktree.rs:96` | BK-tree 近似最近邻搜索 |
| `PdqHashGpu` | struct | `tasks/pdq_hash.rs:91` | PDQ GPU 计算器（feature-gated: `pdq`） |
| `PdqHashCpu` | struct | `tasks/pdq_hash.rs:40` | PDQ CPU 计算器（feature-gated: `pdq`） |
| `PdqHashResult` | struct | `tasks/pdq_hash.rs:29` | PDQ 哈希结果（256-bit hash + quality） |

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
- **⚠️ convolution.wgsl Uniform 对齐**: `ConvParams` 参数已改为 Storage 绑定（非 Uniform），避免 `kernel` 数组的 16 字节对齐问题

## ANTI-PATTERNS (THIS PROJECT)

- **No `unsafe`** blocks found - pure safe Rust
- **No TODO/FIXME/HACK** markers - clean codebase
- **No `main.rs`** - do not add binary entry point without discussion
- **WGSL inline** - shaders are separate `.wgsl` files, NOT inline strings

## UNIQUE STYLES

- **Pipeline cache** uses custom `fxhash` (not std hash) for shader source hashing
- **BufferPool** uses size-tiered bins (not exact match)
- **Batch submitter** uses `submit()` + `wait_all()` pattern (not futures)
- **Perceptual hash** modules are thin 14-line wrappers delegating to `phasher.rs` + WGSL
- **Convolution** supports both Full2D and Separable modes, separable uses single-encoder two-pass dispatch
- **Matcher** uses Strategy + Chain of Responsibility + Facade pattern combination
- **Dihedral** provides CPU-only bit matrix transforms for rotation-invariant matching

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

- `target/` contains ~2000 build artifacts - excluded from all analysis
- GPU adapter selection: `HighPerformance` preference, all backends
- wgpu backends: Vulkan, Metal, DX12, WebGPU (auto-selected)
- **GPU 初始化失败时自动降级**: 当 wgpu 初始化失败（如无 GPU 驱动、WebAssembly 环境等），自动切换到 CPU 实现
- SHA-256 is the reference implementation (668 lines) - new algorithms should follow its pattern
- Image hash algorithms share common infrastructure via `phasher.rs`
- **CPU 降级设计原则**: CPU 实现保持与 GPU 实现相同的 API 签名，调用方无感知切换
- **⚠️ convolution.wgsl 已修复**: `ConvParams` 参数从 Uniform 绑定改为 Storage 绑定（`var<storage, read>`），解决了 `kernel` 数组不满足 WGSL Uniform 16 字节对齐要求的问题。Rust 端 `ConvParams` 结构体布局无需修改，仅 `BufferUsage::Uniform` → `BufferUsage::Storage`
- **PDQ 哈希通过 `pdq` feature gate 控制**: 默认不编译，需 `cargo build --features pdq` 启用；PDQ 同时提供 GPU（`PdqHashGpu`）和 CPU（`PdqHashCpu`）实现
- **零拷贝流水线**: `GpuResize`、`GpuConvolution`、`GpuGaussianBlur` 均提供 `_gpu` 后缀方法，返回 `GpuBuffer` 而非 `Vec<u8>`，支持 GPU 管线间零拷贝传递

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
- 高斯模糊仅 116 行，职责清晰
- ⚠️ `ConvParams` 的 uniform buffer 对齐问题（kernel 数组 124 f32）需要后续优化
- 为未来新增图像预处理算法（如 Sobel、Laplacian）提供基础

### ADR-003: 二面体变换与匹配器模式

**状态**: 已接受

**背景**: 图像可能经过旋转/翻转，需要不重新计算哈希即可匹配变换后的图像

**决策**:
1. `dihedral.rs` 纯 CPU 实现二面体变换（D4 群 8 种），不依赖 GPU
2. `matcher.rs` 使用策略模式（`HashMatcher` trait）+ 责任链模式（`ChainedMatcher`）+ 门面模式（`HashMatcherFacade`）
3. `HashMatcherFacade` 可选启用二面体变换增强，对查询哈希的 8 种变体分别匹配
4. 支持 64-bit（`DihedralHashes64`）和 256-bit（`DihedralHashes256`）两种哈希尺寸

**后果**:
- 匹配策略可灵活组合，新增策略只需实现 `HashMatcher` trait
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
