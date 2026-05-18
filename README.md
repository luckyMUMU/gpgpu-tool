# wgpu-compute-engine

基于 [wgpu](https://github.com/gfx-rs/wgpu) 的跨平台 GPU 通用计算引擎，为 CPU 密集型任务提供 GPU 并行加速能力。

## 特性

- **跨平台**：基于 wgpu，支持 Vulkan、Metal、DX12、WebGPU
- **架构分层**：能力层（GPU 上下文、缓冲区、管线）与业务层（具体算法）解耦
- **管线缓存**：自动缓存编译后的着色器，避免重复编译
- **异步批量提交**：`GpuBatchSubmitter` 将多次 CPU-GPU 同步合并为一次，显著降低调度开销
- **缓冲区池化**：`BufferPool` 按尺寸分档复用 GPU 缓冲区，减少分配开销

## 架构

```
┌─────────────────────────────────────────┐
│           业务层 (Business)              │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ SHA-256  │ │ 图像哈希 │ │  待扩展  │ │
│  └────┬─────┘ └────┬─────┘ └────┬────┘ │
└───────┼────────────┼────────────┼──────┘
        │            │            │
┌───────┼────────────┼────────────┼──────┐
│       ▼            ▼            ▼      │
│      能力层 (Capability Layer)          │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │GpuContext│ │GpuBuffer │ │Compute  │ │
│  │(设备/队列│ │(缓冲区管 │ │Pipeline │ │
│  │/管线缓存)│ │理/传输) │ │(调度)   │ │
│  └──────────┘ └──────────┘ └─────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │BufferPool│ │GpuBatch  │ │ Chunker │ │
│  │(缓冲区复 │ │Submitter │ │(分批)   │ │
│  │ 用)      │ │(异步提交)│ │         │ │
│  └──────────┘ └──────────┘ └─────────┘ │
└─────────────────────────────────────────┘
                    │
                    ▼
              wgpu (Vulkan/Metal/DX12)
```

## 快速开始

### 同步接口

```rust
use wgpu_compute_engine::{GpuContext, tasks::sha256::Sha256Computer};

let mut ctx = GpuContext::new_sync()?;
let sha256 = Sha256Computer::new(&mut ctx)?;

let messages = vec![b"hello".to_vec(), b"world".to_vec()];
let hashes = sha256.compute(&ctx, &messages)?;

for (i, hash) in hashes.iter().enumerate() {
    println!("消息 {}: {:02x?}", i, hash);
}
```

### 异步批量接口（推荐）

```rust
let mut ctx = GpuContext::new_sync()?;
let sha256 = Sha256Computer::new(&mut ctx)?;

let mut submitter = sha256.batch_submitter(&ctx);

// 多次提交，不阻塞
submitter.submit(&vec![b"batch1".to_vec()])?;
submitter.submit(&vec![b"batch2".to_vec()])?;
submitter.submit(&vec![b"batch3".to_vec()])?;

// 统一等待所有结果
let results = submitter.wait_all()?;
// results: Vec<(usize, [u8; 32])>
```

### 运行示例

```bash
cargo run --example demo
```

## 性能

测试平台：Windows (wgpu Vulkan 后端)

### SHA-256 单 block 消息（32 字节）

| 消息数量 | GPU（同步） | GPU（异步批量） | CPU（单线程） |
|---------|-----------|---------------|-------------|
| 1 条 | 1.6ms | - | 52ns |
| 1000 条 | 1.8ms | - | 52µs |
| 10000 条 | 3.5ms | - | 523µs |

### 异步批量提交加速比

| 场景 | 同步 compute | 异步 batch | 加速比 |
|------|------------|-----------|--------|
| 10 次单条消息 | 16.4ms | **3.8ms** | **4.3x** |
| 100 次单条消息 | 164.4ms | **26.2ms** | **6.3x** |
| 10 次 100 条批量 | 17.3ms | **3.9ms** | **4.4x** |

> 注：GPU 在单条消息场景下慢于 CPU（调度开销 ~1.6ms），但在大批量或异步批量场景下具备实用价值。

## 运行基准测试

```bash
cargo bench
```

基准测试报告将生成在 `target/criterion/` 目录下。

## 项目结构

```
.
├── src/
│   ├── lib.rs           # 库入口与公共 API 导出
│   ├── context.rs       # GpuContext：设备、队列、管线缓存
│   ├── buffer.rs        # GpuBuffer：CPU-GPU 数据传输
│   ├── buffer_pool.rs   # BufferPool：缓冲区复用
│   ├── pipeline.rs      # ComputePipeline：计算管线与调度
│   ├── batch.rs         # GpuBatchSubmitter：异步批量提交
│   ├── chunker.rs       # Chunker：通用分批工具
│   ├── error.rs         # GpuError：统一错误类型
│   └── tasks/           # 业务层算法实现
│       ├── sha256.rs    # SHA-256 GPU 并行哈希
│       ├── sha256.wgsl  # SHA-256 计算着色器
│       └── ...          # 其他算法
├── tests/               # 集成测试
├── benches/             # 性能基准测试
├── examples/            # 使用示例
└── openspec/            # 设计文档与变更管理
```

## 已实现的算法

- [x] SHA-256 并行哈希（单 block 批量 + 多 block 链式）
- [ ] 更多算法待扩展...

## 许可证

MIT
