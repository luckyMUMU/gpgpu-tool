# GPGPU-tool

基于 [wgpu](https://github.com/gfx-rs/wgpu) 的跨平台 GPGPU 加速算法库。

**GPU 优先，CPU 降级。** 开箱即用，无需手写 WGSL 着色器。

---

## 能力

| 能力 | 说明 | 入口 |
|------|------|------|
| 感知哈希 (pHash) | 6 种算法（Mean/Median/Gradient/Block/DoubleGradient/VertGradient），支持 64/256/1024/4096-bit | [`PerceptualHasher`](src/tasks/phasher.rs) |
| GPU 汉明距离匹配 | 批量距离矩阵 + 最近邻搜索，支持 64~4096-bit 哈希，778× 加速 | [`GpuHashMatcher`](src/tasks/gpu_matcher.rs) |
| GPU 端到端图像匹配 | 图像 → GPU 哈希 → GPU 并行匹配，一站式 API | [`GpuImageMatcher`](src/tasks/gpu_image_matcher.rs) |
| PDQ 哈希 | 基于 DCT 频域变换的 256-bit 感知哈希（Meta/Facebook PDQ），含质量评分 | [`PdqHashGpu`](src/tasks/pdq_hash.rs) |
| SHA-256 并行哈希 | GPU 并行计算 SHA-256，批量处理任意数量消息 | [`Sha256Computer`](src/tasks/sha256.rs) |
| GPU 2D 卷积 | 不可分离 2D 卷积 + 可分离卷积，支持 Zero/Clamp/Reflect 边界模式 | [`GpuConvolution`](src/tasks/convolution.rs) |
| GPU 高斯模糊 | 基于可分离卷积的高斯模糊，支持自动 sigma 计算 | [`GpuGaussianBlur`](src/tasks/gaussian_blur.rs) |
| 二面体变换 | D4 群 8 种旋转/翻转变换，支持 64/256/1024/4096-bit，u64 位操作优化 | [`DihedralHashes64`](src/tasks/dihedral.rs) |
| 哈希匹配器 | 策略模式：线性扫描/BK-tree/责任链/门面，支持归一化阈值和二面体增强 | [`HashMatcherFacade`](src/tasks/matcher.rs) |
| BK-tree 近似搜索 | 基于汉明距离的最近邻搜索，O(log N) 复杂度 | [`BkTree`](src/tasks/bktree.rs) |
| GPU 批量缩放 | 计算 shader 实现的 box filter 缩放，支持零拷贝流水线 | `PerceptualHasher::with_resize_mode()` |
| 异步批量提交 | 将多次 dispatch 合并为一次 GPU submit，降调度开销 | [`GpuBatchSubmitter`](src/batch.rs) |
| 缓冲区复用池 | 按尺寸分档的缓冲区缓存，减少重复分配 | [`BufferPool`](src/buffer_pool.rs) |

## 快速开始

### 感知哈希（图像相似度）

```rust
use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};

let mut ctx = GpuContext::new_sync().unwrap();
let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Gradient).unwrap();

// 任意尺寸的灰度图像，内部自动缩放
let images = vec![vec![128u8; 256 * 256]];
let dimensions = vec![(256u32, 256u32)];
let hashes = hasher.compute(&ctx, &images, &dimensions).unwrap();
```

### GPU 端到端图像匹配

```rust
use gpgpu_tool::{GpuContext, tasks::gpu_image_matcher::GpuImageMatcher};
use gpgpu_tool::tasks::phasher::HashAlgorithm;

let mut ctx = GpuContext::new_sync().unwrap();
let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Gradient).unwrap();

let query = vec![vec![128u8; 64 * 64]];
let query_dims = vec![(64u32, 64u32)];
let db = vec![vec![200u8; 64 * 64]];
let db_dims = vec![(64u32, 64u32)];

let results = matcher.find_similar(&ctx, &query, &query_dims, &db, &db_dims, 10).unwrap();
```

### GPU 汉明距离矩阵

```rust
use gpgpu_tool::{GpuContext, tasks::gpu_matcher::GpuHashMatcher};

let mut ctx = GpuContext::new_sync().unwrap();
let matcher = GpuHashMatcher::new(&mut ctx).unwrap();

let queries = vec![0u64, 0xFFFF_FFFF_FFFF_FFFF];
let database = vec![0u64, 0x0000_0000_0000_000F, 0xFFFF_FFFF_FFFF_FFFF];

let matrix = matcher.compute_distance_matrix(&ctx, &queries, &database).unwrap();
// 自动分块：输出超过 max_storage_buffer_binding_size 时自动避免 OOM
```

### 自定义哈希位数

```rust
use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}, HashSize};

let mut ctx = GpuContext::new_sync().unwrap();

// 16x16 网格 → 256 bit 哈希
let hasher = PerceptualHasher::with_hash_size(
    &mut ctx, HashAlgorithm::Mean, HashSize::new(16),
).unwrap();

// 32x32 网格 → 1024 bit 哈希
let hasher = PerceptualHasher::with_hash_size(
    &mut ctx, HashAlgorithm::Block, HashSize::new(32),
).unwrap();
```

### 归一化阈值匹配

```rust
use gpgpu_tool::tasks::matcher::HashMatcherFacade;

let hashes = vec![0xA1B2C3D4, 0x12345678, 0x87654321];
let facade = HashMatcherFacade::linear_scan(hashes);

// 使用归一化阈值比例（0.0-1.0），自动适配哈希位宽
let results = facade.find_similar_ratio(0xA1B2C3D4, 0.1);
```

### BK-tree 近似图像检索

```rust
use gpgpu_tool::tasks::bktree::{BkTree, hamming_distance};

let hashes = vec![0xA1B2C3D4, 0x12345678, 0x87654321];
let tree = BkTree::from_hashes(hashes.iter().copied());

let similar = tree.find(0xA1B2C3D4, 5);
let nearest = tree.find_nearest(0xA1B2C3D4);
```

### GPU 2D 卷积

```rust
use gpgpu_tool::{GpuContext, tasks::convolution::{GpuConvolution, BorderMode}};

let mut ctx = GpuContext::new_sync().unwrap();
let conv = GpuConvolution::new(&mut ctx).unwrap();

let pixels = vec![128u8; 256 * 256];
let kernel: Vec<f32> = vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
let result = conv.convolve_2d(&ctx, &pixels, 256, 256, &kernel, 3, BorderMode::Clamp).unwrap();
```

### SHA-256

```rust
use gpgpu_tool::{GpuContext, tasks::sha256::Sha256Computer};

let mut ctx = GpuContext::new_sync().unwrap();
let sha256 = Sha256Computer::new(&mut ctx).unwrap();

let messages = vec![b"hello".to_vec(), b"world".to_vec()];
let hashes = sha256.compute(&ctx, &messages).unwrap();
```

## 特性

| Feature | 默认 | 说明 |
|---------|------|------|
| `cpu-fallback` | ✅ | 启用 CPU 降级实现（SHA-256、感知哈希 + 高斯模糊），依赖 `sha2` crate |
| `image` | ❌ | 启用 `image` crate 集成，支持直接从 `DynamicImage` 计算哈希 |
| `pdq` | ❌ | 启用 PDQ 哈希（GPU DCT + CPU 量化），256-bit 感知哈希 + 质量评分 |

## 架构

```
src/
├── lib.rs                  # 公共 API 入口 + crate 文档
├── context.rs              # GpuContext: 设备/队列/管线缓存
├── buffer.rs               # GpuBuffer: CPU↔GPU 数据传输
├── buffer_pool.rs          # BufferPool: 缓冲区复用池（可配置大缓冲区缓存）
├── pipeline.rs             # ComputePipeline: 计算管线
├── batch.rs                # GpuBatchSubmitter: 异步批量提交
├── error.rs                # GpuError: 统一错误类型（含 Oom/DeviceLost/Timeout）
├── pixel_pack.rs           # 像素打包/解包工具（u8↔u32 对齐）
├── pipeline_builder.rs     # 声明式管线构建器
├── backend_dispatcher.rs   # GPU/CPU 后端调度 trait
├── poll_counter.rs         # GPU poll 计数器
└── tasks/
    ├── phasher.rs           # 感知哈希统一入口（6 种算法 + GPU 缩放）
    ├── phasher_pipeline.rs  # GPU 管线逻辑（分块/双缓冲/阈值计算）
    ├── phasher_util.rs      # 上传工具函数
    ├── phasher_cpu.rs       # 感知哈希 CPU 降级实现（含高斯模糊）
    ├── gpu_matcher.rs       # GPU 汉明距离匹配器（分块/3 管线/64~4096-bit）
    ├── gpu_image_matcher.rs # 端到端 GPU 图像匹配器
    ├── matcher.rs           # 哈希匹配策略（线性/BK-tree/链式/门面 + 归一化阈值）
    ├── matcher_bytes.rs     # 变长哈希匹配策略
    ├── bktree.rs            # BK-tree 近似搜索（64-bit）
    ├── bktree_bytes.rs      # BK-tree 变长哈希版本
    ├── dihedral.rs          # 二面体变换（u64 位操作优化）
    ├── hash_bytes.rs        # 变长哈希类型 HashBytes
    ├── hash_common.rs       # 感知哈希公共参数与计算流程
    ├── sha256.rs / sha256_cpu.rs       # SHA-256 GPU + CPU 降级
    ├── pdq_hash.rs          # PDQ 哈希（GPU DCT + CPU 量化 + 质量评分）
    ├── convolution.rs       # GPU 2D 卷积（Full2D / Separable）
    ├── gaussian_blur.rs     # GPU 高斯模糊
    ├── gpu_resize.rs        # GPU 批量缩放
    ├── mean_hash.rs / .wgsl         # Mean Hash
    ├── median_hash.rs / .wgsl       # Median Hash
    ├── block_hash.rs / .wgsl        # Block Hash
    ├── gradient_hash.rs / .wgsl     # Gradient Hash
    ├── double_gradient_hash.rs / .wgsl  # Double Gradient Hash
    ├── vert_gradient_hash.rs / .wgsl    # Vertical Gradient Hash
    ├── hamming.wgsl          # GPU 汉明距离着色器（3 入口点）
    ├── sha256.wgsl / sha256_chained.wgsl  # SHA-256 着色器
    ├── pdq_hash.wgsl         # PDQ DCT 着色器
    ├── convolution.wgsl      # 2D 卷积着色器
    └── resize.wgsl           # 缩放着色器
```

## 构建 & 测试

```bash
# 构建
cargo build
cargo build --features image
cargo build --features pdq

# 测试
cargo test --features image
cargo test --features "image,pdq"

# 基准测试
cargo bench --features image

# 代码检查
cargo clippy --features image
```

## 支持的后端

| 后端 | 平台 |
|------|------|
| Vulkan | Windows / Linux |
| Metal | macOS / iOS |
| DX12 | Windows |
| WebGPU | Web |

## License

GPL-3.0
