# GPGPU-tool

基于 [wgpu](https://github.com/gfx-rs/wgpu) 的跨平台 GPGPU 加速算法库。

优先使用 GPU 并行加速，当 GPU 不可用时自动降级到 CPU 实现。开箱即用，无需手写 WGSL 着色器。

---

## 能力

| 能力 | 说明 | 入口 |
|------|------|------|
| SHA-256 并行哈希 | GPU 并行计算 SHA-256，批量处理任意数量消息 | [`Sha256Computer`](src/tasks/sha256.rs) |
| 感知哈希 (pHash) | 6 种算法（Mean/Median/Gradient/Block/DoubleGradient/VertGradient） | [`PerceptualHasher`](src/tasks/phasher.rs) |
| PDQ 哈希 | 基于 DCT 频域变换的 256-bit 感知哈希（Meta/Facebook PDQ），含质量评分 | [`PdqHashGpu`](src/tasks/pdq_hash.rs) |
| GPU 2D 卷积 | 不可分离 2D 卷积 + 可分离卷积（水平+垂直两趟 1D），支持 Zero/Clamp/Reflect 边界模式 | [`GpuConvolution`](src/tasks/convolution.rs) |
| GPU 高斯模糊 | 基于可分离卷积的高斯模糊，支持自动 sigma 计算 | [`GpuGaussianBlur`](src/tasks/gaussian_blur.rs) |
| 二面体变换 | 对哈希位矩阵执行 D4 群 8 种旋转/翻转变换，无需重新计算即可匹配旋转/翻转图像 | [`DihedralHashes64`](src/tasks/dihedral.rs) / [`DihedralHashes256`](src/tasks/dihedral.rs) |
| 哈希匹配器 | 策略模式统一接口：线性扫描、BK-tree、责任链组合，支持二面体变换增强 | [`HashMatcherFacade`](src/tasks/matcher.rs) |
| BK-tree 近似搜索 | 基于汉明距离的最近邻搜索，O(log N) 复杂度 | [`BkTree`](src/tasks/bktree.rs) |
| GPU 批量缩放 | 计算 shader 实现的 box filter 缩放，支持零拷贝流水线 | `PerceptualHasher::with_resize_mode()` |
| 异步批量提交 | 将多次 dispatch 合并为一次 GPU submit，降调度开销 | [`GpuBatchSubmitter`](src/batch.rs) |
| 缓冲区复用池 | 按尺寸分档的缓冲区缓存，减少重复分配 | [`BufferPool`](src/buffer_pool.rs) |

## 快速开始

### SHA-256

```rust
use gpgpu_tool::{GpuContext, tasks::sha256::Sha256Computer};

let mut ctx = GpuContext::new_sync().unwrap();
let sha256 = Sha256Computer::new(&mut ctx).unwrap();

let messages = vec![b"hello".to_vec(), b"world".to_vec()];
let hashes = sha256.compute(&ctx, &messages).unwrap();
```

### 感知哈希（图像相似度）

```rust
use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};

let mut ctx = GpuContext::new_sync().unwrap();
let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

// 任意尺寸的灰度图像，内部自动缩放到算法目标尺寸
let images = vec![vec![128u8; 256 * 256]];
let dimensions = vec![(256u32, 256u32)];
let hashes = hasher.compute(&ctx, &images, &dimensions).unwrap();
```

### GPU 加速缩放 + 哈希（零拷贝流水线）

```rust
use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};

let mut ctx = GpuContext::new_sync().unwrap();

// use_gpu_resize = true 启用 GPU 端缩放
let hasher = PerceptualHasher::with_resize_mode(
    &mut ctx, HashAlgorithm::Mean, true,
).unwrap();

let images = vec![vec![128u8; 1024 * 1024]];
let dimensions = vec![(1024u32, 1024u32)];
let hashes = hasher.compute(&ctx, &images, &dimensions).unwrap();
```

### BK-tree 近似图像检索

```rust
use gpgpu_tool::tasks::bktree::{BkTree, hamming_distance};

let hashes = vec![0xA1B2C3D4, 0x12345678, 0x87654321];
let tree = BkTree::from_hashes(hashes.iter().copied());

// 查找汉明距离 ≤ 5 的近似图像
let similar = tree.find(0xA1B2C3D4, 5);
// 查找最近邻
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

### GPU 高斯模糊

```rust
use gpgpu_tool::{GpuContext, tasks::gaussian_blur::GpuGaussianBlur};

let mut ctx = GpuContext::new_sync().unwrap();
let blur = GpuGaussianBlur::new(&mut ctx).unwrap();

let pixels = vec![128u8; 256 * 256];
let result = blur.blur(&ctx, &pixels, 256, 256, 5, 1.0).unwrap();
```

### 哈希匹配器（策略模式）

```rust
use gpgpu_tool::tasks::matcher::HashMatcherFacade;

let hashes = vec![0xA1B2C3D4, 0x12345678, 0x87654321];

// 精确线性扫描
let facade = HashMatcherFacade::linear_scan(hashes.clone());

// BK-tree 快速匹配
let facade = HashMatcherFacade::bktree(hashes.clone());

// 责任链（BK-tree + 线性扫描）
let facade = HashMatcherFacade::chained(hashes.clone());

// 启用二面体变换增强（匹配旋转/翻转图像）
let facade = HashMatcherFacade::linear_scan(hashes).with_dihedral();

let results = facade.find_similar(0xA1B2C3D4, 5);
```

## 特性

| Feature | 默认 | 说明 |
|---------|------|------|
| `cpu-fallback` | ✅ | 启用 CPU 降级实现（SHA-256、感知哈希），依赖 `sha2` crate |
| `image` | ❌ | 启用 `image` crate 集成，支持直接从 `DynamicImage` 计算哈希 |
| `pdq` | ❌ | 启用 PDQ 哈希（GPU DCT + CPU 量化），256-bit 感知哈希 + 质量评分 |

## 架构

```
src/
├── lib.rs              # 公共 API 入口 + crate 文档
├── context.rs          # GpuContext: 设备/队列/管线缓存
├── buffer.rs           # GpuBuffer: CPU↔GPU 数据传输
├── buffer_pool.rs      # BufferPool: 缓冲区复用池
├── pipeline.rs         # ComputePipeline: 计算管线
├── batch.rs            # GpuBatchSubmitter: 异步批量提交
├── error.rs            # GpuError: 统一错误类型
├── pixel_pack.rs       # 像素打包/解包工具（u8↔u32 对齐）
└── tasks/
    ├── sha256.rs       # SHA-256 并行哈希（GPU）
    ├── sha256_cpu.rs   # SHA-256 CPU 降级实现
    ├── phasher.rs      # 感知哈希统一入口（6 种算法 + GPU 缩放）
    ├── phasher_cpu.rs  # 感知哈希 CPU 降级实现
    ├── pdq_hash.rs     # PDQ 哈希（GPU DCT + CPU 量化）
    ├── convolution.rs  # GPU 2D 卷积（不可分离 + 可分离）
    ├── gaussian_blur.rs # GPU 高斯模糊
    ├── dihedral.rs     # 二面体变换（D4 群 8 种旋转/翻转）
    ├── matcher.rs      # 哈希匹配器（策略模式：线性扫描/BK-tree/责任链/门面）
    ├── bktree.rs       # BK-tree 近似搜索
    ├── hash_common.rs  # 感知哈希公共参数与计算流程
    ├── gpu_resize.rs   # GPU 批量缩放
    ├── *_hash.rs       # 各感知哈希算法薄封装
    ├── sha256.wgsl             # SHA-256 着色器
    ├── sha256_chained.wgsl     # SHA-256 链式着色器
    ├── pdq_hash.wgsl           # PDQ DCT 着色器
    ├── convolution.wgsl        # 2D 卷积着色器
    ├── resize.wgsl             # 缩放着色器
    └── *_hash.wgsl             # 各感知哈希算法着色器
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
