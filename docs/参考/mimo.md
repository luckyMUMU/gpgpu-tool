# GPU Compute Library — 基于 wgpu 的通用并行计算框架

## 一、架构设计

```
┌─────────────────────────────────────────────────────────┐
│                   业务层  Task Layer                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │  SHA256   │  │  Matrix   │  │   Sort    │  ···      │
│  │  Computer │  │  Multiply │  │           │           │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘           │
│        │               │               │                │
├────────┼───────────────┼───────────────┼────────────────┤
│        ▼               ▼               ▼                │
│              能力层  Capability Layer                     │
│  ┌──────────────────────────────────────────────┐       │
│  │  ComputePipeline  — 着色器编译 + 调度执行      │       │
│  ├──────────────────────────────────────────────┤       │
│  │  GpuBuffer        — 类型化缓冲区 + 上传/下载   │       │
│  ├──────────────────────────────────────────────┤       │
│  │  GpuContext       — 设备发现 + 队列管理        │       │
│  └──────────────────────────────────────────────┘       │
│                        │                                 │
│                   ┌────▼────┐                            │
│                   │  wgpu   │                            │
│                   └─────────┘                            │
└─────────────────────────────────────────────────────────┘
```

**设计原则：**
- 能力层（`GpuContext` / `GpuBuffer` / `ComputePipeline`）完全不感知具体业务，只提供 GPU 设备管理、数据搬运、着色器调度三项基础能力
- 业务层（`tasks/*`）只关心 WGSL 着色器编写 + Rust 侧的编解码，通过能力层 API 完成 GPU 计算
- 新增计算任务只需：编写 WGSL 着色器 + 实现一个 Rust wrapper struct

## 二、项目结构

```
gpu-compute/
├── Cargo.toml
├── src/
│   ├── lib.rs              # 公开导出
│   ├── error.rs            # 错误类型
│   ├── context.rs          # GpuContext — 设备/队列
│   ├── buffer.rs           # GpuBuffer — 缓冲区管理
│   ├── pipeline.rs         # ComputePipeline — 着色器调度
│   └── tasks/
│       ├── mod.rs           # 任务 trait + 模块声明
│       └── sha256.rs        # SHA256 实现（含嵌入式 WGSL）
└── tests/
    └── sha256_test.rs       # 正确性验证
```

## 三、完整实现

### 3.1 `Cargo.toml`

```toml
[package]
name = "gpu-compute"
version = "0.1.0"
edition = "2021"
description = "GPU-agnostic parallel compute library powered by wgpu"

[dependencies]
wgpu = "23"
bytemuck = { version = "1", features = ["derive"] }
thiserror = "2"
pollster = "0.4"
log = "0.4"

[dev-dependencies]
sha2 = "0.10"
env_logger = "0.11"
```

---

### 3.2 `src/error.rs`

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,

    #[error("device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("buffer mapping failed: {0}")]
    MapFailed(String),

    #[error("shader error: {0}")]
    Shader(String),

    #[error("{0}")]
    Validation(String),
}
```

---

### 3.3 `src/context.rs`

```rust
use crate::error::GpuError;

/// 持有 wgpu Device 和 Queue，是整个库的入口。
/// 同一 GpuContext 可在多线程间共享 (通过 Arc)。
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: String,
}

impl GpuContext {
    /// 阻塞方式初始化 (适用于同步代码)
    pub fn new_sync() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    /// 异步初始化
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?})", info.name, info.backend);
        log::info!("gpu-compute: adapter = {}", adapter_info);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-compute device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    /// GPU 适配器信息
    pub fn adapter_info(&self) -> &str {
        &self.adapter_info
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// 阻塞等待所有已提交的 GPU 工作完成
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::Maintain::Wait);
    }
}
```

---

### 3.4 `src/buffer.rs`

```rust
use bytemuck::Pod;
use crate::context::GpuContext;

/// 缓冲区用途标记
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// GPU 存储缓冲区 (compute shader 读写，可下载)
    Storage,
    /// Uniform 缓冲区 (compute shader 只读参数)
    Uniform,
}

/// 封装 wgpu::Buffer，提供类型化的上传 / 下载接口。
pub struct GpuBuffer {
    pub(crate) inner: wgpu::Buffer,
    pub(crate) byte_size: u64,
}

impl GpuBuffer {
    // ─── 创建 ───────────────────────────────────────

    /// 从 CPU 数据创建 GPU 缓冲区
    pub fn from_data<T: Pod>(ctx: &GpuContext, data: &[T], usage: BufferUsage) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let byte_size = bytes.len() as u64;
        let flags = Self::make_flags(usage);

        let inner = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu-compute buf"),
                contents: bytes,
                usage: flags,
            });

        Self { inner, byte_size }
    }

    /// 创建指定字节大小的空缓冲区 (用于输出)
    pub fn empty(ctx: &GpuContext, byte_size: u64, usage: BufferUsage) -> Self {
        assert!(byte_size > 0, "buffer size must be > 0");
        let flags = Self::make_flags(usage);

        let inner = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-compute buf"),
            size: byte_size,
            usage: flags,
            mapped_at_creation: false,
        });

        Self { inner, byte_size }
    }

    // ─── 读写 ───────────────────────────────────────

    /// 将 CPU 数据写入已有缓冲区 (queue.write_buffer)
    pub fn write<T: Pod>(&self, ctx: &GpuContext, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        assert!(
            bytes.len() as u64 <= self.byte_size,
            "write size exceeds buffer size"
        );
        ctx.queue.write_buffer(&self.inner, 0, bytes);
    }

    /// 从 GPU 下载数据到 CPU (通过 staging buffer)
    /// 注意：仅对 Storage 类型缓冲区有效
    pub fn download<T: Pod>(&self, ctx: &GpuContext) -> Vec<T> {
        // 1. 创建 staging buffer
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-compute staging"),
            size: self.byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. 编码 copy 命令
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-compute download encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.inner, 0, &staging, 0, self.byte_size);
        ctx.queue.submit(Some(encoder.finish()));

        // 3. 映射并读取
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("channel closed")
            .map_err(|e| e.to_string())
            .expect("map failed");

        let mapped = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        result
    }

    // ─── 内部 ───────────────────────────────────────

    fn make_flags(usage: BufferUsage) -> wgpu::BufferUsages {
        match usage {
            BufferUsage::Storage => {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
            }
            BufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }
    }
}
```

> **注意：** 需要在使用处引入 `use wgpu::util::DeviceExt;`（提供 `create_buffer_init`）。

---

### 3.5 `src/pipeline.rs`

```rust
use crate::buffer::GpuBuffer;
use crate::context::GpuContext;
use crate::error::GpuError;

/// 计算管线 — 编译后的 WGSL 着色器 + 自动推导的 bind group layout。
/// 可复用：同一 pipeline 配合不同 buffer 反复 dispatch。
pub struct ComputePipeline {
    pub(crate) inner: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    /// 从 WGSL 源码创建计算管线。
    /// entry_point 默认为 "main"。
    pub fn from_wgsl(
        ctx: &GpuContext,
        wgsl_source: &str,
        label: Option<&str>,
    ) -> Result<Self, GpuError> {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });

        let inner = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label,
                layout: None, // 自动从着色器推导 bind group layout
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group_layout = inner.get_bind_group_layout(0);

        Ok(Self {
            inner,
            bind_group_layout,
        })
    }

    /// 执行一次计算调度。
    ///
    /// - `bindings`: `[(binding_index, &GpuBuffer), ...]`
    /// - `workgroups`: `(x, y, z)` 工作组数量
    ///
    /// 命令提交到队列后立即返回；调用方如需同步可使用 `GpuContext::wait_idle()`
    /// 或在后续 `download()` 时隐式等待。
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        bindings: &[(u32, &GpuBuffer)],
        workgroups: (u32, u32, u32),
    ) {
        // 构建 bind group
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .map(|(binding, buf)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buf.inner.as_entire_binding(),
            })
            .collect();

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-compute bind group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        });

        // 编码 + 提交
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-compute encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inner);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        ctx.queue.submit(Some(encoder.finish()));
    }
}
```

---

### 3.6 `src/tasks/mod.rs`

```rust
pub mod sha256;
```

---

### 3.7 `src/tasks/sha256.rs` — 完整 SHA256 实现

这是业务层的核心：WGSL 着色器 + Rust 侧编解码。

```rust
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

// ────────────────────────────────────────────────────────
// WGSL 着色器 (SHA-256 单 block 实现)
// ────────────────────────────────────────────────────────
const SHA256_WGSL: &str = r#"
// ── bindings ────────────────────────────────────────────
//   @binding(0)  storage read      messages  — 输入 (已填充, 每条 16×u32)
//   @binding(1)  storage read_write hashes    — 输出 (每条 8×u32)
//   @binding(2)  uniform            params    — 参数

@group(0) @binding(0) var<storage, read>       messages : array<u32>;
@group(0) @binding(1) var<storage, read_write> hashes   : array<u32>;
@group(0) @binding(2) var<uniform>             params   : Params;

struct Params {
    message_count : u32,
};

// ── SHA-256 round constants ─────────────────────────────
const K : array<u32, 64> = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

// ── 位运算原语 ──────────────────────────────────────────
fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn ch(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ ((x ^ 0xFFFFFFFFu) & z); }
fn maj(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (x & z) ^ (y & z); }
fn bsig0(x: u32) -> u32 { return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u); }
fn bsig1(x: u32) -> u32 { return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u); }
fn ssig0(x: u32) -> u32 { return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u); }
fn ssig1(x: u32) -> u32 { return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u); }

// ── 入口点 ──────────────────────────────────────────────
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.message_count) { return; }

    // 1) 读取 16 个 u32 (一个 64 字节 block)
    var w : array<u32, 64>;
    let base = idx * 16u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        w[i] = messages[base + i];
    }

    // 2) 扩展消息调度表
    for (var i = 16u; i < 64u; i = i + 1u) {
        w[i] = ssig1(w[i - 2u]) + w[i - 7u] + ssig0(w[i - 15u]) + w[i - 16u];
    }

    // 3) 初始化工作变量
    var a = 0x6a09e667u; var b = 0xbb67ae85u;
    var c = 0x3c6ef372u; var d = 0xa54ff53au;
    var e = 0x510e527fu; var f = 0x9b05688cu;
    var g = 0x1f83d9abu; var h = 0x5be0cd19u;

    // 4) 64 轮压缩
    for (var i = 0u; i < 64u; i = i + 1u) {
        let t1 = h + bsig1(e) + ch(e, f, g) + K[i] + w[i];
        let t2 = bsig0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // 5) 写入最终哈希值
    let out = idx * 8u;
    hashes[out + 0u] = 0x6a09e667u + a;
    hashes[out + 1u] = 0xbb67ae85u + b;
    hashes[out + 2u] = 0x3c6ef372u + c;
    hashes[out + 3u] = 0xa54ff53au + d;
    hashes[out + 4u] = 0x510e527fu + e;
    hashes[out + 5u] = 0x9b05688cu + f;
    hashes[out + 6u] = 0x1f83d9abu + g;
    hashes[out + 7u] = 0x5be0cd19u + h;
}
"#;

// ────────────────────────────────────────────────────────
// Rust 侧: Uniform 参数结构体 (与 WGSL struct Params 对齐)
// ────────────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Sha256Params {
    message_count: u32,
    _pad: [u32; 3], // 对齐到 16 字节
}

// ────────────────────────────────────────────────────────
// Sha256Computer — 业务层公开 API
// ────────────────────────────────────────────────────────

/// GPU SHA-256 并行计算器。
///
/// 支持对 **多条消息** 并行计算 SHA-256。
/// 当前限制：每条消息长度 ≤ 55 字节（单 block 处理）。
pub struct Sha256Computer {
    pipeline: ComputePipeline,
}

impl Sha256Computer {
    const WORKGROUP_SIZE: u32 = 256;

    /// 创建实例 (编译着色器，仅需一次)
    pub fn new(ctx: &GpuContext) -> Result<Self, GpuError> {
        let pipeline =
            ComputePipeline::from_wgsl(ctx, SHA256_WGSL, Some("SHA256 compute pipeline"))?;
        Ok(Self { pipeline })
    }

    /// 批量计算 SHA-256。
    ///
    /// ```ignore
    /// let hashes = computer.compute(&ctx, &[b"hello", b"world"])?;
    /// assert_eq!(hashes.len(), 2);
    /// ```
    pub fn compute(
        &self,
        ctx: &GpuContext,
        messages: &[&[u8]],
    ) -> Result<Vec<[u8; 32]>, GpuError> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        // ── 1. 预处理：填充消息为 64 字节 block，转为大端 u32 ──
        let mut padded_words: Vec<u32> = Vec::with_capacity(messages.len() * 16);
        for msg in messages {
            let block = pad_single_block(msg);
            for chunk in block.chunks_exact(4) {
                padded_words.push(u32::from_be_bytes(chunk.try_into().unwrap()));
            }
        }

        // ── 2. 创建 GPU 缓冲区 ──
        let msg_buf = GpuBuffer::from_data(ctx, &padded_words, BufferUsage::Storage);
        let hash_buf = GpuBuffer::empty(
            ctx,
            (messages.len() * 32) as u64,
            BufferUsage::Storage,
        );

        let params = Sha256Params {
            message_count: messages.len() as u32,
            _pad: [0; 3],
        };
        let param_buf = GpuBuffer::from_data(ctx, &[params], BufferUsage::Uniform);

        // ── 3. 调度 GPU 计算 ──
        let num_workgroups =
            (messages.len() as u32 + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;

        self.pipeline.dispatch(
            ctx,
            &[(0, &msg_buf), (1, &hash_buf), (2, &param_buf)],
            (num_workgroups, 1, 1),
        );

        // ── 4. 下载结果 ──
        let raw: Vec<u32> = hash_buf.download(ctx);

        let mut results = Vec::with_capacity(messages.len());
        for i in 0..messages.len() {
            let mut hash = [0u8; 32];
            for j in 0..8 {
                hash[j * 4..j * 4 + 4]
                    .copy_from_slice(&raw[i * 8 + j].to_be_bytes());
            }
            results.push(hash);
        }

        Ok(results)
    }
}

// ────────────────────────────────────────────────────────
// SHA-256 填充 (CPU 侧预处理)
// ────────────────────────────────────────────────────────

/// 将消息填充为单个 64 字节 block (FIPS 180-4 §5.1.1)。
/// 要求 `msg.len() <= 55`，否则需要多 block 支持。
fn pad_single_block(msg: &[u8]) -> [u8; 64] {
    assert!(
        msg.len() <= 55,
        "message length {} exceeds single-block limit (55 bytes)",
        msg.len()
    );
    let mut block = [0u8; 64];
    block[..msg.len()].copy_from_slice(msg);
    block[msg.len()] = 0x80;
    let bit_len = (msg.len() as u64) * 8;
    block[56..64].copy_from_slice(&bit_len.to_be_bytes());
    block
}
```

---

### 3.8 `src/lib.rs`

```rust
//! # gpu-compute
//!
//! 基于 wgpu 的通用 GPU 并行计算框架。
//! 将任意 GPU 抽象为 CPU 的并行加速器。
//!
//! ## 架构
//!
//! - **能力层**: `GpuContext`, `GpuBuffer`, `ComputePipeline`
//! - **业务层**: `tasks::sha256::Sha256Computer` 等

pub mod error;
pub mod context;
pub mod buffer;
pub mod pipeline;
pub mod tasks;

// 便捷重导出
pub use error::GpuError;
pub use context::GpuContext;
pub use buffer::{GpuBuffer, BufferUsage};
pub use pipeline::ComputePipeline;
pub use tasks::sha256::Sha256Computer;
```

---

### 3.9 正确性测试 `tests/sha256_test.rs`

```rust
use gpu_compute::{GpuContext, Sha256Computer};
use sha2::{Digest, Sha256};

#[test]
fn sha256_matches_reference() {
    let _ = env_logger::try_init();
    let ctx = GpuContext::new_sync().expect("failed to init GPU");
    println!("adapter: {}", ctx.adapter_info());

    let computer = Sha256Computer::new(&ctx).expect("failed to create SHA256 computer");

    let test_cases: Vec<&[u8]> = vec![
        b"",
        b"a",
        b"abc",
        b"hello",
        b"hello world",
        b"The quick brown fox jumps over the lazy dog",
        &[0xffu8; 55], // 最大单 block 长度
    ];

    let gpu_hashes = computer.compute(&ctx, &test_cases).expect("GPU compute failed");
    assert_eq!(gpu_hashes.len(), test_cases.len());

    for (i, (msg, gpu_hash)) in test_cases.iter().zip(gpu_hashes.iter()).enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(msg);
        let cpu_hash: [u8; 32] = hasher.finalize().into();

        assert_eq!(
            gpu_hash, &cpu_hash,
            "mismatch at case {}: msg={:?}\n  gpu  = {}\n  cpu  = {}",
            i,
            if msg.len() <= 20 { String::from_utf8_lossy(msg).to_string() } else { format!("{} bytes", msg.len()) },
            hex::encode(gpu_hash),
            hex::encode(cpu_hash),
        );
    }
}
```

> 测试需要添加 `hex = "0.4"` 到 `[dev-dependencies]`。

---

### 3.10 使用示例 `examples/demo.rs`

```rust
use gpu_compute::{GpuContext, Sha256Computer};

fn main() {
    env_logger::init();

    // 1) 初始化 GPU 上下文 (一次性)
    let ctx = GpuContext::new_sync().expect("init GPU");
    println!("GPU: {}", ctx.adapter_info());

    // 2) 创建 SHA256 计算器 (编译着色器，一次性)
    let sha256 = Sha256Computer::new(&ctx).expect("init SHA256");

    // 3) 批量并行计算
    let messages: Vec<&[u8]> = (0..10_000)
        .map(|i| format!("message-{i}").into_bytes())
        .collect::<Vec<_>>()
        .iter()
        .map(|v| v.as_slice())
        .collect();

    let start = std::time::Instant::now();
    let hashes = sha256.compute(&ctx, &messages).expect("GPU SHA256");
    let elapsed = start.elapsed();

    println!(
        "Computed {} SHA-256 hashes in {:.2?} ({:.0} hashes/sec)",
        hashes.len(),
        elapsed,
        hashes.len() as f64 / elapsed.as_secs_f64()
    );

    // 打印前 3 个结果
    for (i, hash) in hashes.iter().take(3).enumerate() {
        println!("  [{i}] {}", hex::encode(hash));
    }
}
```

---

## 四、扩展指南 — 如何新增计算任务

以"GPU 并行排序"为例，只需三步：

```rust
// src/tasks/gpu_sort.rs

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::pipeline::ComputePipeline;

const SORT_SHADER: &str = r#"
// ... WGSL bitonic sort ...
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }
"#;

pub struct GpuSorter {
    pipeline: ComputePipeline,
}

impl GpuSorter {
    pub fn new(ctx: &GpuContext) -> Result<Self, crate::GpuError> {
        let pipeline = ComputePipeline::from_wgsl(ctx, SORT_SHADER, Some("sort"))?;
        Ok(Self { pipeline })
    }

    pub fn sort_u32(&self, ctx: &GpuContext, data: &mut [u32]) {
        let buf = GpuBuffer::from_data(ctx, data, BufferUsage::Storage);
        let params = GpuBuffer::from_data(ctx, &[data.len() as u32], BufferUsage::Uniform);

        // 多轮 dispatch (bitonic sort 需要 log(n) 轮)
        for stage in 0..32u32 {
            for step in (0..=stage).rev() {
                self.pipeline.dispatch(
                    ctx,
                    &[(0, &buf), (1, &params)],
                    ((data.len() as u32 + 255) / 256, 1, 1),
                );
            }
        }

        let sorted: Vec<u32> = buf.download(ctx);
        data.copy_from_slice(&sorted);
    }
}
```

**关键模式：**
- 业务 struct 持有一个 `ComputePipeline`
- `new()` 编译着色器（一次性开销）
- 公开方法中：创建 buffer → dispatch → download

---

## 五、架构决策与后续优化方向

| 维度 | 当前方案 | 可优化方向 |
|------|----------|-----------|
| **多 block 消息** | 单 block（≤55 字节） | 多轮 dispatch：每轮处理一个 block，轮间用前一轮输出作为初始状态 |
| **管线缓存** | 每个 task 自行持有 pipeline | 全局 `PipelineCache<HashMap<String, Pipeline>>` |
| **Buffer 池** | 每次 compute 新建 buffer | `BufferPool` 复用显存，减少分配开销 |
| **异步批处理** | 同步提交 + 阻塞等待 | 异步 `Future` API + 多 command buffer 并行编码 |
| **错误恢复** | panic / expect | 设备丢失检测 + 自动重新初始化 |
| **多 GPU** | 单适配器 | `GpuContextPool` 管理多设备，任务级负载均衡 |

这个架构的核心优势：**能力层与业务层完全解耦**。新增任何 GPU 计算任务（矩阵乘、向量检索、图像处理……）只需写一个 WGSL 着色器 + 薄薄一层 Rust wrapper，复用全部底层能力。