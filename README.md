# wgpu-compute-engine

基于 [wgpu](https://github.com/gfx-rs/wgpu) 的跨平台 GPU 并行加速引擎。

为 CPU 密集型计算任务提供 GPU 并行加速能力，开箱即用，无需手写 WGSL 着色器。

---

## 能力

| 能力 | 说明 | 入口 |
|------|------|------|
| SHA-256 并行哈希 | GPU 并行计算 SHA-256，批量处理任意数量消息 | [`Sha256Computer`](src/tasks/sha256.rs) |
| 感知哈希 (pHash) | 6 种算法（Mean/Median/Gradient/Block/DoubleGradient/VertGradient） | [`PerceptualHasher`](src/tasks/phasher.rs) |
| BK-tree 近似搜索 | 基于汉明距离的最近邻搜索，O(log N) 复杂度 | [`BkTree`](src/tasks/bktree.rs) |
| GPU 批量缩放 | 计算 shader 实现的 box filter 缩放，支持零拷贝流水线 | `PerceptualHasher::with_resize_mode()` |
| 异步批量提交 | 将多次 dispatch 合并为一次 GPU submit，降调度开销 | [`GpuBatchSubmitter`](src/batch.rs) |
| 缓冲区复用池 | 按尺寸分档的缓冲区缓存，减少重复分配 | [`BufferPool`](src/buffer_pool.rs) |

## 快速开始

### SHA-256

```rust
use wgpu_compute_engine::{GpuContext, tasks::sha256::Sha256Computer};

let mut ctx = GpuContext::new_sync().unwrap();
let sha256 = Sha256Computer::new(&mut ctx).unwrap();

let messages = vec![b"hello".to_vec(), b"world".to_vec()];
let hashes = sha256.compute(&ctx, &messages).unwrap();
```

### 感知哈希（图像相似度）

```rust
use wgpu_compute_engine::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};

let mut ctx = GpuContext::new_sync().unwrap();
let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

// 任意尺寸的灰度图像，内部自动缩放到算法目标尺寸
let images = vec![vec![128u8; 256 * 256]];
let dimensions = vec![(256u32, 256u32)];
let hashes = hasher.compute(&ctx, &images, &dimensions).unwrap();
```

### GPU 加速缩放 + 哈希（零拷贝流水线）

```rust
use wgpu_compute_engine::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};

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
use wgpu_compute_engine::tasks::bktree::{BkTree, hamming_distance};

let hashes = vec![0xA1B2C3D4, 0x12345678, 0x87654321];
let tree = BkTree::from_hashes(hashes.iter().copied());

// 查找汉明距离 ≤ 5 的近似图像
let similar = tree.find(0xA1B2C3D4, 5);
// 查找最近邻
let nearest = tree.find_nearest(0xA1B2C3D4);
```

## 特性

| Feature | 说明 |
|---------|------|
| `image` | 启用 `image` crate 集成，支持直接从 `DynamicImage` 计算哈希 |

## 架构

```
src/
├── lib.rs           # 公共 API 入口 + crate 文档
├── context.rs       # GpuContext: 设备/队列/管线缓存
├── buffer.rs        # GpuBuffer: CPU↔GPU 数据传输
├── buffer_pool.rs   # BufferPool: 缓冲区复用池
├── pipeline.rs      # ComputePipeline: 计算管线
├── batch.rs         # GpuBatchSubmitter: 异步批量提交
├── error.rs         # GpuError: 统一错误类型
└── tasks/
    ├── sha256.rs    # SHA-256 并行哈希
    ├── phasher.rs   # 感知哈希统一入口
    ├── bktree.rs    # BK-tree 近似搜索
    └── *.wgsl       # WGSL 着色器
```

## 构建 & 测试

```bash
# 构建
cargo build
cargo build --features image

# 测试
cargo test --features image

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

MIT