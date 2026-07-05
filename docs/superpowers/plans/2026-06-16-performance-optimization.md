---
change: performance-optimization
design-doc: docs/superpowers/specs/2026-06-16-performance-optimization-design.md
base-ref: 13ec266714d35b140fc99375b0785ad222847559
---

# 性能优化与显存安全 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过两阶段重构（能力层优先 + 业务层跟进）解决 DX12 OOM 崩溃、GPU 同步开销、着色器效率、内存管理与抽象边界五类瓶颈，同时守护"GPU 类型无关抽象"长期目标。

**Architecture:** 采用方案 C（能力层优先 + 业务层跟进）。阶段 1 聚焦 `context.rs`/`pipeline.rs`/`batch.rs`/`buffer*.rs` 的重构（P0 显存安全 + P1 GPU 同步 + P4 抽象边界），让能力层成为稳定基座；阶段 2 基于新 API 优化业务层（P2 着色器 + P3 内存 + P5 逐图路径 + P6 验证）。所有优化不破坏 `GpuContext`/`GpuBuffer`/`ComputePipeline` 抽象边界。

**Tech Stack:** Rust 2021 edition, wgpu v24, WGSL, thiserror, fxhash, pollster, Criterion（基准测试）

**Design Doc:** `docs/superpowers/specs/2026-06-16-performance-optimization-design.md`

**OpenSpec Tasks:** `openspec/changes/performance-optimization/tasks.md`

---

## 文件结构概览

### 阶段 1（能力层重构）

| 文件 | 责任 | 变更类型 |
|------|------|---------|
| `src/context.rs` | `GpuContext` + `PipelineCache` + `UniformPool`（新增） | 修改：注册 UncapturedErrorHandler、`backend()` OOM 检查、PipelineCache RefCell 化、新增 UniformPool |
| `src/buffer_pool.rs` | `BufferPool` 缓冲区池 | 修改：`acquire` 返回 `Result<Buffer, GpuError>`，预检查 `max_storage_buffer_binding_size` |
| `src/batch.rs` | `GpuBatchSubmitter` + `DualEncoderSubmitter` | 修改：`wait_all` 保留引用、解放 DualEncoderSubmitter |
| `src/pipeline.rs` | `ComputePipeline` | 修改：`dispatch_with_params` 改用 `UniformPool`，`encode_dispatch_with_params_into` 增加 `queue` 参数 |
| `src/error.rs` | `GpuError` | 修改：完善 `Oom` 错误变体字段 |
| `src/backend_dispatcher.rs` | `BackendDispatcher` | 修改：验证 `dispatch_gpu` 响应 `backend()` 动态切换 |
| `src/buffer.rs` | `GpuBuffer` | 修改：`download`/`download_batch` 标记 `#[deprecated]` |
| `src/lib.rs` | 公共 API 导出 | 修改：导出 `UniformPool` 相关 API |

### 阶段 2（业务层优化）

| 文件 | 责任 | 变更类型 |
|------|------|---------|
| `src/tasks/resize.wgsl` + `gpu_resize.rs` | GPU 图像缩放 | 修改：积分图 SAT 两阶段调度 |
| `src/tasks/block_hash.wgsl` | Block Hash 着色器 | 修改：预计算 block 均值图 |
| `src/tasks/pdq_hash.wgsl` + `pdq_hash.rs` | PDQ 哈希 | 修改：cos 表预计算 + 低频系数裁剪 + 2D workgroup |
| `src/tasks/convolution.wgsl` | GPU 卷积 | 修改：Full2D LDS 共享内存 |
| `src/tasks/hamming.wgsl` | GPU 汉明距离 | 修改：并行归约最近邻 |
| `src/tasks/dihedral.rs` | 二面体变换 | 修改：64/256-bit 直接位操作 |
| `src/tasks/gpu_matcher.rs` | GPU 匹配器 | 修改：扁平化距离矩阵 + `find_nearest_neighbors` 分块 |
| `src/tasks/matcher.rs` + `matcher_bytes.rs` | 匹配策略 | 修改：`Arc` 共享 + 先 filter 后 clone + dihedral 批量查询 |
| `src/tasks/phasher.rs` + `phasher_pipeline.rs` + `phasher_util.rs` | 感知哈希编排 | 修改：尺寸分组 + 预处理合并 + Mean/Median GPU 阈值 |
| `src/tasks/gpu_image_matcher.rs` | 端到端匹配器 | 修改：新增 `GpuImageMatcherIndex` + `index_database` |
| `src/tasks/hash_common.rs` | 感知哈希公共逻辑 | 修改：`compute_phash` 分批 + `compute_phash_from_gpu_buffer` 分批 |
| `tests/cache_perf_test.rs` | 缓存数据性能测试 | 新建 |
| `tests/common/cache_loader.rs` | Czkawka 缓存解析 | 新建 |
| `benches/cache_matcher_bench.rs` | 缓存数据基准测试 | 新建 |

---

## 阶段 1：能力层重构（P0 + P1 + P4）

### Task 1: P0-层1 — 注册 UncapturedErrorHandler

**Files:**
- Modify: `src/context.rs:202-280`（`init_gpu`）
- Modify: `src/context.rs:284-334`（`init_software_adapter`）

**依赖:** 无

- [x] **Step 1: 在 `src/context.rs` 模块级新增原子标志**

在 `src/context.rs` 顶部（`use` 语句之后）新增：

```rust
use std::sync::atomic::{AtomicBool, Ordering};

/// GPU 显存不足标志，由 UncapturedErrorHandler 设置
pub(crate) static GPU_OOM_FLAG: AtomicBool = AtomicBool::new(false);
/// GPU 设备丢失标志，由 UncapturedErrorHandler 设置
pub(crate) static GPU_DEVICE_LOST_FLAG: AtomicBool = AtomicBool::new(false);
```

- [x] **Step 2: 在 `init_gpu` 中 Device 创建后注册 handler**

定位 `init_gpu` 函数中 `let (device, queue) = adapter.request_device(...)` 之后，新增：

```rust
device.push_error_handler(Box::new(|e| {
    match e {
        wgpu::Error::OutOfMemory { .. } => {
            log::error!("GPU 显存不足: {:?}", e);
            GPU_OOM_FLAG.store(true, Ordering::SeqCst);
        }
        wgpu::Error::DeviceLost { .. } => {
            log::error!("GPU 设备丢失: {:?}", e);
            GPU_DEVICE_LOST_FLAG.store(true, Ordering::SeqCst);
        }
        _ => log::warn!("GPU 验证错误: {:?}", e),
    }
}));
```

- [x] **Step 3: 在 `init_software_adapter` 中同步注册 handler**

在 `init_software_adapter` 函数中 Device 创建后，添加与 Step 2 相同的 handler 注册代码。

- [x] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过，无错误

- [x] **Step 5: 提交**

```bash
git add src/context.rs
git commit -m "feat(context): 注册 UncapturedErrorHandler 设置 OOM/DeviceLost 原子标志"
```

---

### Task 2: P0-层2 — `BufferPool::acquire` 预检查（breaking change）

**Files:**
- Modify: `src/buffer_pool.rs:157-177`（`acquire` 方法）
- Modify: `src/error.rs:24-25`（`GpuError::Oom`）

**依赖:** Task 1

- [x] **Step 1: 完善 `GpuError::Oom` 变体**

在 `src/error.rs` 中修改 `Oom` 变体，增加请求大小与限制字段：

```rust
#[error("GPU 显存不足: 请求 {requested} 字节超过限制 {limit} 字节")]
Oom {
    /// 请求分配的字节数
    requested: u64,
    /// 设备限制的最大字节数
    limit: u64,
},
```

- [x] **Step 2: 修改 `acquire` 签名与预检查逻辑**

在 `src/buffer_pool.rs` 中修改 `acquire` 方法：

```rust
pub fn acquire(
    &self,
    device: &Device,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer, GpuError> {
    let max_binding = device.limits().max_storage_buffer_binding_size as u64;
    if size > max_binding {
        return Err(GpuError::Oom {
            requested: size,
            limit: max_binding,
        });
    }
    // ... 现有池化逻辑（保持不变，仅返回类型包裹 Ok）...
}
```

- [x] **Step 3: 编译验证（预期失败，调用点未更新）**

Run: `cargo build`
Expected: 编译失败，约 15 处调用点报错（这是预期的，将在 Task 3 修复）

- [x] **Step 4: 提交（中间状态，标记 WIP）**

```bash
git add src/buffer_pool.rs src/error.rs
git commit -m "refactor(buffer_pool): acquire 返回 Result 并预检查 max_storage_buffer_binding_size"
```

---

### Task 3: P0-层2 — 同步修改 `acquire` 调用点

**Files:**
- Modify: `src/buffer.rs`（`download_with_pool` 等）
- Modify: `src/batch.rs`（`ensure_initialized`）
- Modify: `src/tasks/hash_common.rs`（`compute_phash`）
- Modify: `src/tasks/phasher_util.rs`（`upload_image_to_gpu`）
- Modify: `src/tasks/gpu_matcher.rs`（`compute_distance_matrix` 等）
- Modify: `src/tasks/convolution.rs`
- Modify: `src/tasks/gpu_resize.rs`
- Modify: `src/tasks/pdq_hash.rs`
- Modify: `src/tasks/sha256.rs`

**依赖:** Task 2

- [x] **Step 1: 用 Grep 定位所有 `acquire` 调用点**

Run: 在 IDE 中使用 Grep 搜索 `\.acquire\(` 模式，限定 `src/` 目录
Expected: 找到约 15 处调用点

- [x] **Step 2: 逐个修改调用点，使用 `?` 传播错误**

对每个调用点，将 `let buffer = pool.acquire(device, size, usage);` 改为 `let buffer = pool.acquire(device, size, usage)?;`。若所在函数返回类型非 `Result<_, GpuError>`，需同步修改函数签名。

示例（`src/buffer.rs` `download_with_pool`）：

```rust
pub fn download_with_pool(
    // ...
) -> Result<Vec<u8>, GpuError> {
    let staging = pool.acquire(device, size, BufferUsage::COPY_DST | BufferUsage::MAP_READ)?;
    // ...
}
```

- [x] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过，无错误

- [x] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [x] **Step 5: 提交**

```bash
git add src/buffer.rs src/batch.rs src/tasks/hash_common.rs src/tasks/phasher_util.rs src/tasks/gpu_matcher.rs src/tasks/convolution.rs src/tasks/gpu_resize.rs src/tasks/pdq_hash.rs src/tasks/sha256.rs
git commit -m "refactor: 同步 acquire 调用点处理 Result 返回值"
```

---

### Task 4: P0-层3 — `compute_phash` 内部分批

**Files:**
- Modify: `src/tasks/hash_common.rs:267-322`（`compute_phash`）
- Modify: `src/tasks/hash_common.rs:497-535`（`compute_phash_from_gpu_buffer`）

**依赖:** Task 3

- [x] **Step 1: 重构 `compute_phash` 增加分批逻辑**

在 `src/tasks/hash_common.rs` 中将 `compute_phash` 拆分为外层分批 + 内层单批：

```rust
pub fn compute_phash(
    pipeline: &Arc<ComputePipeline>,
    ctx: &GpuContext,
    images: &[Vec<u8>],
    // ... 其他参数 ...
) -> Result<Vec<u64>, GpuError> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let pixels_per_image = images[0].len();
    let per_image_bytes = (pixels_per_image * 4) as u64;
    let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;

    // 单图超限直接报错
    if per_image_bytes > max_binding {
        return Err(GpuError::InvalidInput(format!(
            "单图尺寸 {} 字节超过 max_storage_buffer_binding_size {} 字节",
            per_image_bytes, max_binding
        )));
    }

    // 按显存预算分批（安全系数 0.5）
    let max_batch = ((DEFAULT_MAX_BATCH_SIZE / per_image_bytes) as usize).max(1);

    let mut all_hashes = Vec::with_capacity(images.len());
    for chunk in images.chunks(max_batch) {
        let chunk_hashes = compute_phash_single_batch(
            pipeline, ctx, chunk, /* 其他参数 */
        )?;
        all_hashes.extend(chunk_hashes);
    }
    Ok(all_hashes)
}

/// 单批哈希计算（原 compute_phash 的核心逻辑）
fn compute_phash_single_batch(
    pipeline: &Arc<ComputePipeline>,
    ctx: &GpuContext,
    images: &[Vec<u8>],
    // ... 其他参数 ...
) -> Result<Vec<u64>, GpuError> {
    // 原 compute_phash 的实现体
}
```

- [x] **Step 2: 同步修改 `compute_phash_from_gpu_buffer`**

在 `src/tasks/hash_common.rs:497-535` 中，对 `compute_phash_from_gpu_buffer` 应用相同的分批模式。由于输入已是 GPU buffer，分批逻辑基于 `u32s_per_image` 估算单批最大图像数。

- [x] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [x] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [x] **Step 5: 提交**

```bash
git add src/tasks/hash_common.rs
git commit -m "feat(hash_common): compute_phash 内部按显存预算自动分批"
```

---

### Task 5: P0-层3 — 逐图路径单图尺寸检查

**Files:**
- Modify: `src/tasks/phasher_pipeline.rs`（`compute_gpu_per_image_pipeline`）
- Modify: `src/tasks/gpu_image_matcher.rs`（`compute_hashes`）

**依赖:** Task 4

- [ ] **Step 1: 在 `compute_gpu_per_image_pipeline` 循环内增加单图尺寸检查**

在 `src/tasks/phasher_pipeline.rs` 的逐图处理循环开头，增加：

```rust
for (i, image) in images.iter().enumerate() {
    let per_image_bytes = (image.len() * 4) as u64;
    let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;
    if per_image_bytes > max_binding {
        return Err(GpuError::InvalidInput(format!(
            "图像 {} 尺寸 {} 字节超过 max_storage_buffer_binding_size {} 字节",
            i, per_image_bytes, max_binding
        )));
    }
    // ... 原处理逻辑 ...
}
```

- [ ] **Step 2: 在 `gpu_image_matcher.rs` `compute_hashes` 入口增加显存预算分批加固**

在 `compute_hashes` 入口处，调用 `compute_phash` 之前增加显存预算检查，确保单批不超过 `max_storage_buffer_binding_size`。若超限，内部调用 `compute_phash` 的分批逻辑（Task 4 已实现）。

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 5: 提交**

```bash
git add src/tasks/phasher_pipeline.rs src/tasks/gpu_image_matcher.rs
git commit -m "feat: 逐图路径与 image_matcher 入口增加显存预算分批加固"
```

---

### Task 6: P0-层4 — 运行时 OOM → CPU 降级

**Files:**
- Modify: `src/context.rs`（`backend()` 方法）
- Modify: `src/backend_dispatcher.rs:47-66`（`dispatch_gpu`）

**依赖:** Task 1

- [ ] **Step 1: 修改 `GpuContext::backend()` 增加 OOM 标志检查**

在 `src/context.rs` 中修改 `backend()` 方法：

```rust
impl GpuContext {
    pub fn backend(&self) -> ComputeBackend {
        if self.backend == ComputeBackend::Gpu && GPU_OOM_FLAG.load(Ordering::SeqCst) {
            log::warn!("检测到 GPU OOM，切换到 CPU 降级模式");
            return ComputeBackend::Cpu;
        }
        if self.backend == ComputeBackend::Gpu && GPU_DEVICE_LOST_FLAG.load(Ordering::SeqCst) {
            log::warn!("检测到 GPU 设备丢失，切换到 CPU 降级模式");
            return ComputeBackend::Cpu;
        }
        self.backend
    }
}
```

- [ ] **Step 2: 验证 `DefaultBackendDispatcher::dispatch_gpu` 响应动态切换**

检查 `src/backend_dispatcher.rs:47-66` 的 `dispatch_gpu` 实现，确认它调用 `ctx.backend()` 而非缓存字段。若已调用，无需修改；若缓存字段，改为调用 `backend()` 方法。

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 5: 提交**

```bash
git add src/context.rs src/backend_dispatcher.rs
git commit -m "feat(context): backend() 动态检查 OOM/DeviceLost 标志触发 CPU 降级"
```

---

### Task 7: P1 — `wait_all` 保留引用

**Files:**
- Modify: `src/batch.rs:268-270, 324-326, 341-343`（`GpuBatchSubmitter::wait_all`）
- Modify: `src/batch.rs:532-627`（`DualEncoderSubmitter::wait_all`）

**依赖:** 无

- [ ] **Step 1: 修改 `GpuBatchSubmitter::wait_all` 仅清空 encoder 和 pending**

在 `src/batch.rs` 中定位 `GpuBatchSubmitter::wait_all` 的实现，修改清空逻辑：

```rust
fn wait_all(&mut self) -> Result<Vec<Vec<u8>>, GpuError> {
    // ... 现有 submit + poll + 读取逻辑（保持不变）...

    // 仅清空 encoder 和 pending，保留 device/queue/pool
    self.encoder = None;
    self.pending.clear();
    // 不再清空 self.device / self.queue / self.pool

    Ok(results)
}
```

- [ ] **Step 2: 同步修改 `DualEncoderSubmitter::wait_all`**

在 `src/batch.rs:532-627` 中对 `DualEncoderSubmitter::wait_all` 应用相同修改，保留 device/queue/pool 引用。

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过（验证连续 `wait_all` 调用不报错）

- [ ] **Step 5: 提交**

```bash
git add src/batch.rs
git commit -m "refactor(batch): wait_all 保留 device/queue/pool 引用支持连续批量提交"
```

---

### Task 8: P1 — 解放 `DualEncoderSubmitter`

**Files:**
- Modify: `src/batch.rs:389, 414, 653, 660`（移除 `#[cfg(feature = "dual-encoder")]`）
- Modify: `Cargo.toml`（`dual-encoder` feature 改为默认启用）

**依赖:** Task 7

- [ ] **Step 1: 移除 `DualEncoderSubmitter` 的 `#[cfg(feature = "dual-encoder")]` 标注**

在 `src/batch.rs` 中搜索所有 `#[cfg(feature = "dual-encoder")]` 标注（约 4 处），移除这些标注，使 `DualEncoderSubmitter` 作为默认实现。

- [ ] **Step 2: 在 `Cargo.toml` 中将 `dual-encoder` 改为默认 feature**

```toml
[features]
default = ["cpu-fallback", "dual-encoder"]
dual-encoder = []
# ... 其他 feature ...
```

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 5: 提交**

```bash
git add src/batch.rs Cargo.toml
git commit -m "feat(batch): 解放 DualEncoderSubmitter 作为默认实现"
```

---

### Task 9: P1 — 跨 chunk 复用 `GpuBatchSubmitter`

**Files:**
- Modify: `src/tasks/phasher_pipeline.rs:110, 217`

**依赖:** Task 7, Task 8

- [ ] **Step 1: 将 `batch` 提升到 chunk 循环外**

在 `src/tasks/phasher_pipeline.rs` 中定位 chunk 处理循环（约 110 行和 217 行），将 `GpuBatchSubmitter::new()` 提升到循环外：

```rust
let mut batch = GpuBatchSubmitter::new();
for &(start, end) in &chunks {
    batch.ensure_initialized(ctx)?;  // 首次初始化，后续复用
    // ... 编码 dispatch ...
    let results = batch.wait_all()?;  // 仅清空 encoder/pending，保留引用
}
```

- [ ] **Step 2: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 3: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 4: 提交**

```bash
git add src/tasks/phasher_pipeline.rs
git commit -m "perf(phasher_pipeline): 跨 chunk 复用 GpuBatchSubmitter 减少同步开销"
```

---

### Task 10: P4 — 新增 `UniformPool` 下沉到 `GpuContext`

**Files:**
- Modify: `src/context.rs`（新增 `UniformPool` 结构 + `GpuContext` 字段 + API）

**依赖:** 无

- [ ] **Step 1: 在 `src/context.rs` 新增 `UniformPool` 结构**

```rust
use std::cell::RefCell;
use std::collections::HashMap;

/// Uniform 缓冲区池，按 size tier 缓存可复用缓冲区
pub(crate) struct UniformPool {
    /// 按 size tier 缓存的可复用 Uniform 缓冲区
    buffers: RefCell<HashMap<u64, Vec<GpuBuffer>>>,
    /// 每 tier 最大缓存数
    max_per_tier: usize,
}

impl UniformPool {
    /// 创建新的 UniformPool
    pub(crate) fn new(max_per_tier: usize) -> Self {
        Self {
            buffers: RefCell::new(HashMap::new()),
            max_per_tier,
        }
    }

    /// 计算 size tier（向上取整到 2 的幂）
    fn size_tier(size: u64) -> u64 {
        if size <= 16 { 16 }
        else if size <= 32 { 32 }
        else if size <= 64 { 64 }
        else if size <= 128 { 128 }
        else if size <= 256 { 256 }
        else if size <= 512 { 512 }
        else { size.next_power_of_two() }
    }

    /// 获取指定大小的 Uniform 缓冲区
    pub(crate) fn acquire(&self, device: &Device, size: u64) -> GpuBuffer {
        let tier = Self::size_tier(size);
        let mut buffers = self.buffers.borrow_mut();
        if let Some(pool) = buffers.get_mut(&tier) {
            if let Some(buffer) = pool.pop() {
                return buffer;
            }
        }
        GpuBuffer::create(device, tier, BufferUsage::Uniform)
    }

    /// 归还 Uniform 缓冲区
    pub(crate) fn release(&self, buffer: GpuBuffer) {
        let tier = Self::size_tier(buffer.size());
        let mut buffers = self.buffers.borrow_mut();
        let pool = buffers.entry(tier).or_insert_with(Vec::new);
        if pool.len() < self.max_per_tier {
            pool.push(buffer);
        }
    }
}
```

- [ ] **Step 2: 在 `GpuContext` 新增 `uniform_pool` 字段与公开 API**

```rust
pub struct GpuContext {
    // ... 现有字段 ...
    uniform_pool: UniformPool,
}

impl GpuContext {
    /// 获取指定大小的 Uniform 缓冲区（从池中复用或新建）
    pub fn acquire_uniform(&self, size: u64) -> Result<GpuBuffer, GpuError> {
        let device = self.device()?;
        Ok(self.uniform_pool.acquire(device, size))
    }

    /// 归还 Uniform 缓冲区到池中
    pub fn release_uniform(&self, buffer: GpuBuffer) {
        self.uniform_pool.release(buffer);
    }
}
```

- [ ] **Step 3: 在 `GpuContext` 构造函数中初始化 `uniform_pool`**

在 `init_gpu` 和 `init_software_adapter` 中创建 `GpuContext` 时，初始化 `uniform_pool: UniformPool::new(32)`。

- [ ] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 5: 提交**

```bash
git add src/context.rs
git commit -m "feat(context): 新增 UniformPool 下沉 Uniform 缓冲区管理到 GpuContext"
```

---

### Task 11: P4 — `ComputePipeline` 改用 `UniformPool`

**Files:**
- Modify: `src/pipeline.rs:544-609`（`dispatch_with_params` + `encode_dispatch_with_params_into`）
- Modify: `src/tasks/hash_common.rs:599`（消除业务层绕过封装）

**依赖:** Task 10

- [ ] **Step 1: 修改 `dispatch_with_params` 使用 `ctx.acquire_uniform`**

在 `src/pipeline.rs` 中修改 `dispatch_with_params`：

```rust
pub fn dispatch_with_params(
    &self,
    ctx: &GpuContext,
    params: &[u8],
    bindings: &[&GpuBuffer],
    workgroup_count: [u32; 3],
) -> Result<(), GpuError> {
    let uniform_buffer = ctx.acquire_uniform(params.len() as u64)?;
    ctx.queue()?.write_buffer(uniform_buffer.raw(), 0, params);
    // ... dispatch 逻辑 ...
    ctx.release_uniform(uniform_buffer);
    Ok(())
}
```

- [ ] **Step 2: 修改 `encode_dispatch_with_params_into` 增加 `queue` 参数并复用 UniformPool**

```rust
pub fn encode_dispatch_with_params_into(
    &self,
    ctx: &GpuContext,
    encoder: &mut CommandEncoder,
    params: &[u8],
    bindings: &[&GpuBuffer],
    workgroup_count: [u32; 3],
) -> Result<(), GpuError> {
    let uniform_buffer = ctx.acquire_uniform(params.len() as u64)?;
    ctx.queue()?.write_buffer(uniform_buffer.raw(), 0, params);
    // ... encode 逻辑 ...
    ctx.release_uniform(uniform_buffer);
    Ok(())
}
```

- [ ] **Step 3: 消除 `hash_common.rs:599` 业务层绕过封装**

将 `hash_common.rs:599` 直接创建 Uniform 缓冲区的代码，改为调用 `pipeline.encode_dispatch_with_params_into`。

- [ ] **Step 4: 移除 `pipeline.rs` 内部的 `reusable_uniform: RefCell<Option<GpuBuffer>>`**

删除 `ComputePipeline` 结构中的 `reusable_uniform` 字段及其相关逻辑，统一由 `GpuContext::UniformPool` 管理。

- [ ] **Step 5: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 6: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 7: 提交**

```bash
git add src/pipeline.rs src/tasks/hash_common.rs
git commit -m "refactor(pipeline): dispatch_with_params 改用 UniformPool 消除业务层绕过封装"
```

---

### Task 12: P4 — `PipelineCache` RefCell 化

**Files:**
- Modify: `src/context.rs:34-67`（`PipelineCache` 结构）
- Modify: `src/context.rs:428-438`（`get_or_create_pipeline`）

**依赖:** Task 10

- [ ] **Step 1: 将 `PipelineCache` 的 `entries`、`generations`、`next_gen` 改为 `RefCell`**

```rust
struct PipelineCache {
    entries: RefCell<HashMap<u64, Arc<ComputePipeline>>>,
    generations: RefCell<HashMap<u64, u64>>,
    next_gen: RefCell<u64>,
    max_entries: usize,
}
```

- [ ] **Step 2: 修改 `get_or_create` 接受 `&self` 并使用 `borrow_mut`**

```rust
impl PipelineCache {
    pub fn get_or_create(
        &self,  // ← 改为 &self
        device: &Device,
        descriptor: &PipelineDescriptor,
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        let hash = compute_hash(descriptor);
        {
            let entries = self.entries.borrow();
            if let Some(pipeline) = entries.get(&hash) {
                return Ok(Arc::clone(pipeline));
            }
        }
        let pipeline = Arc::new(ComputePipeline::new(device, descriptor)?);
        {
            let mut entries = self.entries.borrow_mut();
            entries.insert(hash, Arc::clone(&pipeline));
        }
        {
            let mut generations = self.generations.borrow_mut();
            let mut next_gen = self.next_gen.borrow_mut();
            generations.insert(hash, *next_gen);
            *next_gen += 1;
        }
        Ok(pipeline)
    }
}
```

- [ ] **Step 3: 修改 `GpuContext::get_or_create_pipeline` 接受 `&self`**

```rust
impl GpuContext {
    pub fn get_or_create_pipeline(
        &self,  // ← 改为 &self
        descriptor: &PipelineDescriptor,
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        let device = self.device()?;
        self.pipeline_cache.get_or_create(device, descriptor)
    }
}
```

- [ ] **Step 4: 修复所有 `get_or_create_pipeline` 调用点**

搜索所有 `get_or_create_pipeline` 调用点，将 `&mut ctx` 改为 `&ctx`。涉及文件：`src/tasks/*.rs` 中所有调用 `ctx.get_or_create_pipeline` 的位置。

- [ ] **Step 5: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 6: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 7: 提交**

```bash
git add src/context.rs src/tasks/
git commit -m "refactor(context): PipelineCache RefCell 化 get_or_create_pipeline 接受 &self"
```

---

### Task 13: P4 — `PipelineCache` LRU O(1) arena

**Files:**
- Modify: `src/context.rs:76-87`（`PipelineCache` LRU 实现）

**依赖:** Task 12

- [ ] **Step 1: 新增 arena + 双向链表节点结构**

```rust
struct LruNode {
    hash: u64,
    prev: Option<usize>,
    next: Option<usize>,
}

struct PipelineCache {
    entries: RefCell<HashMap<u64, Arc<ComputePipeline>>>,
    generations: RefCell<HashMap<u64, u64>>,
    next_gen: RefCell<u64>,
    /// LRU arena
    arena: RefCell<Vec<LruNode>>,
    /// hash → arena 索引
    index: RefCell<HashMap<u64, usize>>,
    head: RefCell<Option<usize>>,  // LRU 链表头（最近使用）
    tail: RefCell<Option<usize>>,  // LRU 链表尾（最久未用）
    max_entries: usize,
}
```

- [ ] **Step 2: 实现 O(1) 的 `touch`（移到头部）、`evict_tail`（淘汰尾部）、`insert`（插入头部）**

```rust
impl PipelineCache {
    fn touch(&self, hash: u64) {
        // 移动节点到头部，O(1)
    }

    fn evict_tail(&self) -> Option<u64> {
        // 移除尾部节点，O(1)，返回被淘汰的 hash
    }

    fn insert_head(&self, hash: u64) {
        // 插入新节点到头部，O(1)
    }
}
```

- [ ] **Step 3: 在 `get_or_create` 中集成 LRU 逻辑**

在 `get_or_create` 命中缓存时调用 `touch(hash)`，插入新条目时调用 `insert_head(hash)`，若超 `max_entries` 调用 `evict_tail()` 并从 `entries`/`generations`/`index` 中移除。

- [ ] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 5: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 6: 提交**

```bash
git add src/context.rs
git commit -m "perf(context): PipelineCache LRU 改 O(1) arena + 双向链表"
```

---

### Task 14: P1 — 标记 `download` 系列为 `#[deprecated]` + 推广 `_with_pool`

**Files:**
- Modify: `src/buffer.rs`（`download` + `download_batch`）
- Modify: `src/tasks/gpu_matcher.rs`（推广 `download_batch_with_pool`）

**依赖:** Task 3

- [ ] **Step 1: 标记 `download` 和 `download_batch` 为 `#[deprecated]`**

在 `src/buffer.rs` 中为 `download` 和 `download_batch` 添加：

```rust
#[deprecated(note = "使用 download_with_pool 替代，合并多个 download 为单次 poll")]
pub fn download(/* ... */) -> Result<Vec<u8>, GpuError> { /* ... */ }

#[deprecated(note = "使用 download_batch_with_pool 替代，合并多个 download 为单次 poll")]
pub fn download_batch(/* ... */) -> Result<Vec<Vec<u8>>, GpuError> { /* ... */ }
```

- [ ] **Step 2: 在 `gpu_matcher.rs` 等核心模块推广 `download_batch_with_pool`**

搜索 `src/tasks/gpu_matcher.rs` 中所有 `download` 调用，改为 `download_batch_with_pool`，合并多个独立 download 为单次 poll。

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过（可能有 deprecation 警告，需修复剩余调用点）

- [ ] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 所有现有测试通过

- [ ] **Step 5: 提交**

```bash
git add src/buffer.rs src/tasks/gpu_matcher.rs
git commit -m "refactor(buffer): 标记 download 系列为 deprecated 推广 _with_pool 版本"
```

---

### Task 15: 阶段 1 集成验证

**Files:** 无（仅验证）

**依赖:** Task 1-14 全部完成

- [ ] **Step 1: 运行完整测试套件**

Run: `cargo test`
Expected: 所有测试通过

- [ ] **Step 2: 运行 PDQ feature 测试**

Run: `cargo test --features pdq`
Expected: 所有测试通过

- [ ] **Step 3: 运行 clippy**

Run: `cargo clippy`
Expected: 无错误（警告可接受）

- [ ] **Step 4: 运行 clippy with image feature**

Run: `cargo clippy --features image`
Expected: 无错误

- [ ] **Step 5: 提交阶段 1 完成标记**

```bash
git commit --allow-empty -m "chore: 阶段 1（能力层重构 P0+P1+P4）完成"
```

---

## 阶段 2：业务层优化（P2 + P3 + P5 + P6）

### Task 16: P2 — `resize.wgsl` 积分图（SAT）实现

**Files:**
- Modify: `src/tasks/resize.wgsl`
- Modify: `src/tasks/gpu_resize.rs`

**依赖:** Task 15

- [ ] **Step 1: 在 `resize.wgsl` 新增积分图构建 pass**

```wgsl
struct SatParams {
    src_width: u32,
    src_height: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> sat: array<u32>;
@group(0) @binding(2) var<uniform> params: SatParams;

@compute @workgroup_size(16, 16)
fn compute_sat_horizontal(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // 水平前缀和：sat[y * width + x] = src[y][x] + sat[y][x-1]
    if (gid.x >= params.src_width || gid.y >= params.src_height) {
        return;
    }
    let idx = gid.y * params.src_width + gid.x;
    let prev = select(0u, sat[idx - 1u], gid.x > 0u);
    sat[idx] = src[idx] + prev;
}

@compute @workgroup_size(16, 16)
fn compute_sat_vertical(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // 垂直前缀和：sat[y * width + x] = sat[y][x] + sat[y-1][x]
    if (gid.x >= params.src_width || gid.y >= params.src_height) {
        return;
    }
    let idx = gid.y * params.src_width + gid.x;
    let prev = select(0u, sat[idx - params.src_width], gid.y > 0u);
    sat[idx] = sat[idx] + prev;
}
```

- [ ] **Step 2: 新增 SAT 查询 pass**

```wgsl
@compute @workgroup_size(8, 8)
fn resize_sat(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // O(1) 查询：sum = SAT[x1,y1] - SAT[x0,y1] - SAT[x1,y0] + SAT[x0,y0]
    // mean = sum / ((x1-x0) * (y1-y0))
    // ... 实现略，参考设计文档 4.1 ...
}
```

- [ ] **Step 3: 在 `gpu_resize.rs` 增加两阶段调度**

新增 `resize_with_sat` 方法，仅在大比例下采样场景（src/dst ≥ 4×）启用：

```rust
pub fn resize_with_sat(
    &self,
    ctx: &GpuContext,
    src: &GpuBuffer,
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
) -> Result<GpuBuffer, GpuError> {
    // 1. 构建 SAT
    // 2. 查询 SAT 生成 dst
    // 仅在 src/dst >= 4x 时调用
}
```

- [ ] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 5: 运行 resize 测试验证一致性**

Run: `cargo test --test gpu_resize_test`
Expected: 测试通过，SAT 路径与原区域平均路径输出一致

- [ ] **Step 6: 提交**

```bash
git add src/tasks/resize.wgsl src/tasks/gpu_resize.rs
git commit -m "perf(resize): 积分图 SAT 实现大比例下采样 O(1) 查询"
```

---

### Task 17: P2 — `block_hash.wgsl` 预计算 block 均值

**Files:**
- Modify: `src/tasks/block_hash.wgsl`

**依赖:** Task 15

- [ ] **Step 1: 新增 `compute_block_means` 预处理 pass**

```wgsl
@compute @workgroup_size(8, 8)
fn compute_block_means(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // 将 64×64 图像下采样为 8×8 block 均值图（64 个 u32）
    // block_mean[bx, by] = mean(src[bx*8..bx*8+8, by*8..by*8+8])
}
```

- [ ] **Step 2: 新增 `block_hash_with_means` 主 pass**

```wgsl
@compute @workgroup_size(8, 8)
fn block_hash_with_means(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // 从 64 个 block 均值读取，消除 O(block_area) 循环
    // 提取 m_current = block_mean(bx, by) 一次，复用于 left/top 分支
    let bx = gid.x;
    let by = gid.y;
    let m_current = block_means[by * 8u + bx];
    let m_left = select(m_current, block_means[by * 8u + bx - 1u], bx > 0u);
    let m_top = select(m_current, block_means[(by - 1u) * 8u + bx], by > 0u);
    // ... 哈希位计算 ...
}
```

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行 block_hash 测试验证一致性**

Run: `cargo test --test block_hash_test`
Expected: 测试通过，预计算路径与原路径输出一致

- [ ] **Step 5: 提交**

```bash
git add src/tasks/block_hash.wgsl
git commit -m "perf(block_hash): 预计算 block 均值图消除 O(block_area) 循环"
```

---

### Task 18: P2 — `pdq_hash.wgsl` cos 表预计算 + 系数裁剪

**Files:**
- Modify: `src/tasks/pdq_hash.wgsl`
- Modify: `src/tasks/pdq_hash.rs`

**依赖:** Task 15

- [ ] **Step 1: 预计算 64×64 余弦查找表作为 storage buffer**

在 `pdq_hash.rs` 中新增 cos 表初始化逻辑：

```rust
fn create_cos_table(ctx: &GpuContext) -> Result<GpuBuffer, GpuError> {
    let mut cos_table = vec![0f32; 64 * 64];
    for u in 0..64 {
        for x in 0..64 {
            cos_table[u * 64 + x] = ((std::f32::consts::PI * (2.0 * x as f32 + 1.0) * u as f32) / 128.0).cos();
        }
    }
    // 上传到 GPU storage buffer
}
```

- [ ] **Step 2: 修改 `pdq_hash.wgsl` 从 storage buffer 读取 cos 值**

```wgsl
@group(0) @binding(3) var<storage, read> cos_table: array<f32, 4096>;

fn cos_value(u: u32, x: u32) -> f32 {
    return cos_table[u * 64u + x];
}
```

- [ ] **Step 3: workgroup 改为 2D (8,8,1)，利用 LDS 共享 cos 表**

```wgsl
@compute @workgroup_size(8, 8, 1)
fn dct_2d(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    var lds_cos: array<f32, 64>;
    // 协作加载 cos 表到 LDS
    // ... DCT 计算从 LDS 读取 ...
}
```

- [ ] **Step 4: 新增第 3 个 pass `extract_low_freq` 提取 16×16 低频系数**

```wgsl
@compute @workgroup_size(16, 16, 1)
fn extract_low_freq(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // 从 64×64 DCT 系数中提取左上角 16×16 低频系数
    // 减少 93.75% 下载量
}
```

- [ ] **Step 5: 编译验证（需 pdq feature）**

Run: `cargo build --features pdq`
Expected: 编译通过

- [ ] **Step 6: 运行 PDQ 测试验证一致性**

Run: `cargo test --features pdq`
Expected: 测试通过，cos 表路径与原路径输出一致

- [ ] **Step 7: 提交**

```bash
git add src/tasks/pdq_hash.wgsl src/tasks/pdq_hash.rs
git commit -m "perf(pdq): cos 表预计算 + 2D workgroup + 低频系数裁剪"
```

---

### Task 19: P2 — `convolution.wgsl` Full2D LDS 共享内存

**Files:**
- Modify: `src/tasks/convolution.wgsl:120-130`

**依赖:** Task 15

- [ ] **Step 1: 新增 `lds_2d` 共享内存声明**

```wgsl
const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const R: u32 = 2u;  // kernel radius

var<workgroup> lds_2d: array<f32, (WG_X + 2u * R) * (WG_Y + 2u * R)>;
```

- [ ] **Step 2: 协作加载 halo 区域像素到 LDS**

```wgsl
@compute @workgroup_size(8, 8)
fn convolve_full_2d_lds(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // 协作加载 (WG_X+2R)×(WG_Y+2R) 像素到 lds_2d
    let lx = lid.x;
    let ly = lid.y;
    // 加载中心 + halo 区域
    // ... 实现略 ...

    workgroupBarrier();

    // 卷积计算从 LDS 读取
    let gx = wid.x * WG_X + lx;
    let gy = wid.y * WG_Y + ly;
    var sum: f32 = 0.0;
    for (var ky: u32 = 0u; ky < 2u * R + 1u; ky++) {
        for (var kx: u32 = 0u; kx < 2u * R + 1u; kx++) {
            let lds_x = lx + kx;
            let lds_y = ly + ky;
            sum += lds_2d[lds_y * (WG_X + 2u * R) + lds_x] * kernel[ky * (2u * R + 1u) + kx];
        }
    }
    // 写入输出
}
```

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行 convolution 测试验证一致性**

Run: `cargo test --test convolution_test`
Expected: 测试通过，LDS 路径与原路径输出一致

- [ ] **Step 5: 提交**

```bash
git add src/tasks/convolution.wgsl
git commit -m "perf(convolution): Full2D 模式增加 LDS 共享内存消除全局内存重复读取"
```

---

### Task 20: P2 — `hamming.wgsl` 并行归约最近邻

**Files:**
- Modify: `src/tasks/hamming.wgsl:183-200`

**依赖:** Task 15

- [ ] **Step 1: 重写 `find_nearest_neighbor` 为并行归约**

```wgsl
var<workgroup> wg_best_dist: array<u32, 256u>;
var<workgroup> wg_best_idx: array<u32, 256u>;

@compute @workgroup_size(256)
fn find_nearest_neighbor_parallel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_id = lid.x;
    let db_offset = wid.x * 256u;

    // 每线程计算 1 个 db 条目的距离
    var best_dist: u32 = 0xFFFFFFFFu;
    var best_idx: u32 = 0xFFFFFFFFu;
    if (db_offset + local_id < db_count) {
        let dist = compute_distance(query, db[db_offset + local_id]);
        best_dist = dist;
        best_idx = db_offset + local_id;
    }

    // 写入 shared memory
    wg_best_dist[local_id] = best_dist;
    wg_best_idx[local_id] = best_idx;
    workgroupBarrier();

    // 树形归约
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            let other_dist = wg_best_dist[local_id + stride];
            let other_idx = wg_best_idx[local_id + stride];
            if (other_dist < wg_best_dist[local_id]) {
                wg_best_dist[local_id] = other_dist;
                wg_best_idx[local_id] = other_idx;
            }
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // 线程 0 写入结果
    if (local_id == 0u) {
        result[wid.x] = vec2<u32>(wg_best_idx[0], wg_best_dist[0]);
    }
}
```

- [ ] **Step 2: 优化 `hamming_distance_matrix` 加载阶段**

修改 `hamming_distance_matrix` 入口点，所有 256 线程协作加载 query_tile 和 db_tile_matrix。

- [ ] **Step 3: 保留串行路径作为 fallback**

将原 `find_nearest_neighbor` 重命名为 `find_nearest_neighbor_serial`，保留作为后端不支持 subgroup 时的 fallback。

- [ ] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 5: 运行 hamming 测试验证一致性**

Run: `cargo test --test gpu_matcher_test`
Expected: 测试通过，并行归约路径与串行路径输出一致

- [ ] **Step 6: 提交**

```bash
git add src/tasks/hamming.wgsl
git commit -m "perf(hamming): find_nearest_neighbor 改并行归约 256 线程协作扫描"
```

---

### Task 21: P2 — PDQ CPU DCT 快速算法

**Files:**
- Modify: `src/tasks/pdq_hash.rs`（CPU DCT 路径）

**依赖:** Task 18

- [ ] **Step 1: 实现快速 DCT 算法替代朴素实现**

在 `pdq_hash.rs` CPU 路径中，将 O(N²) 朴素 DCT 替换为快速 DCT（如 AAN 算法或 rustdct crate）。

若引入 `rustdct` crate，需在 `Cargo.toml` 的 `[dependencies]` 中添加（仅 pdq feature 启用时）：

```toml
[dependencies]
rustdct = { version = "0.7", optional = true }

[features]
pdq = ["dep:rustdct"]
```

- [ ] **Step 2: 编译验证**

Run: `cargo build --features pdq`
Expected: 编译通过

- [ ] **Step 3: 运行 PDQ 测试验证一致性**

Run: `cargo test --features pdq`
Expected: 测试通过，快速 DCT 与朴素 DCT 输出一致

- [ ] **Step 4: 提交**

```bash
git add src/tasks/pdq_hash.rs Cargo.toml
git commit -m "perf(pdq): CPU DCT 改用快速算法"
```

---

### Task 22: P3 — `dihedral.rs` 64-bit 直接位操作

**Files:**
- Modify: `src/tasks/dihedral.rs:19-109`

**依赖:** Task 15

- [ ] **Step 1: 实现 `transpose_8x8` delta-swap 位操作**

```rust
/// 8×8 位矩阵转置（delta-swap 算法）
fn transpose_8x8(hash: u64) -> u64 {
    let mut x = hash;
    // 第 1 步：交换 bit 1↔8, 3↔10, ...（间隔 7）
    x = ((x & 0xAA00AA00AA00AA00) >> 7) | ((x & 0x0055005500550055) << 7) | (x & 0x55AA55AA55AA55AA);
    // 第 2 步：交换间隔 14
    x = ((x & 0xCCCC0000CCCC0000) >> 14) | ((x & 0x3333000033330000) << 14) | (x & 0x0000FFFF0000FFFF);
    // 第 3 步：交换间隔 28
    x = ((x & 0xF0F0F0F000000000) >> 28) | ((x & 0x0F0F0F0F00000000) << 28) | (x & 0x00000000FFFFFFFF);
    x
}
```

- [ ] **Step 2: 实现翻转、旋转等其他变换的位操作版本**

```rust
fn flip_horizontal_8x8(hash: u64) -> u64 {
    // 水平翻转 8×8 位矩阵
}

fn flip_vertical_8x8(hash: u64) -> u64 {
    // 垂直翻转 8×8 位矩阵
}

fn rotate_90_8x8(hash: u64) -> u64 {
    transpose_8x8(flip_horizontal_8x8(hash))
}
```

- [ ] **Step 3: 修改 `DihedralHashes64::all_variants` 返回 `[u64; 8]`**

```rust
impl DihedralHashes64 {
    pub fn all_variants(&self) -> [u64; 8] {
        let h = self.original;
        [
            h,
            rotate_90_8x8(h),
            rotate_180_8x8(h),
            rotate_270_8x8(h),
            flip_horizontal_8x8(h),
            flip_vertical_8x8(h),
            transpose_8x8(h),
            anti_transpose_8x8(h),
        ]
    }
}
```

- [ ] **Step 4: 编写 exhaustive 测试对比 Vec<bool> 结果**

在 `tests/dihedral_test.rs` 中新增测试，对所有 2^16 个 8×8 位矩阵（采样）对比位操作版本与原 Vec<bool> 版本的输出一致性。

- [ ] **Step 5: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 6: 运行测试验证**

Run: `cargo test --test dihedral_test`
Expected: 测试通过

- [ ] **Step 7: 提交**

```bash
git add src/tasks/dihedral.rs tests/dihedral_test.rs
git commit -m "perf(dihedral): 64-bit 直接位操作消除 Vec<bool> 分配"
```

---

### Task 23: P3 — `dihedral.rs` 256-bit 位操作 + 统一抽象

**Files:**
- Modify: `src/tasks/dihedral.rs:114-210`

**依赖:** Task 22

- [ ] **Step 1: 在 `[u64; 4]` 上实现 256-bit 位操作**

```rust
fn transpose_16x16(hash: [u64; 4]) -> [u64; 4] {
    // 在 [u64; 4] 上实现 16×16 位矩阵转置
}
```

- [ ] **Step 2: 抽象 `BitMatrix<N>` 泛型结构（可选）**

若四种尺寸实现风格统一，抽象出泛型结构：

```rust
trait BitMatrixOps {
    fn transpose(&self) -> Self;
    fn flip_horizontal(&self) -> Self;
    fn flip_vertical(&self) -> Self;
    fn rotate_90(&self) -> Self;
    fn all_variants(&self) -> [Self; 8];
}
```

- [ ] **Step 3: 修改 `DihedralHashes256::all_variants` 返回 `[[u64; 4]; 8]`**

- [ ] **Step 4: 编写 exhaustive 测试对比 Vec<bool> 结果**

- [ ] **Step 5: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 6: 运行测试验证**

Run: `cargo test --test dihedral_test`
Expected: 测试通过

- [ ] **Step 7: 提交**

```bash
git add src/tasks/dihedral.rs tests/dihedral_test.rs
git commit -m "perf(dihedral): 256-bit 直接位操作 + 统一抽象风格"
```

---

### Task 24: P3 — `gpu_matcher.rs` 距离矩阵扁平化

**Files:**
- Modify: `src/tasks/gpu_matcher.rs:144-196`

**依赖:** Task 15

- [ ] **Step 1: 新增 `DistanceMatrix` 扁平化结构**

```rust
/// 扁平化距离矩阵
pub struct DistanceMatrix {
    /// 扁平 n*m 数据
    pub data: Vec<u32>,
    /// 列数（database 大小）
    pub cols: usize,
}

impl DistanceMatrix {
    /// 获取第 i 行的切片
    pub fn row(&self, i: usize) -> &[u32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// 获取行数
    pub fn rows(&self) -> usize {
        self.data.len() / self.cols
    }
}
```

- [ ] **Step 2: 修改 `compute_distance_matrix` 返回 `DistanceMatrix`**

```rust
pub fn compute_distance_matrix(
    &self,
    ctx: &GpuContext,
    queries: &[u64],
    database: &[u64],
) -> Result<DistanceMatrix, GpuError> {
    let cols = database.len();
    let mut data = vec![0u32; queries.len() * cols];
    // 分块路径直接写入预分配 Vec 的对应区间
    // ... 实现略 ...
    Ok(DistanceMatrix { data, cols })
}
```

- [ ] **Step 3: 修复所有 `compute_distance_matrix` 调用点**

搜索所有调用点，将 `Vec<Vec<u32>>` 改为 `DistanceMatrix`，使用 `.row(i)` 访问行。

- [ ] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 5: 运行测试验证**

Run: `cargo test --test gpu_matcher_test`
Expected: 测试通过

- [ ] **Step 6: 提交**

```bash
git add src/tasks/gpu_matcher.rs
git commit -m "perf(gpu_matcher): 距离矩阵扁平化消除 N 次堆分配"
```

---

### Task 25: P3 — `matcher.rs` Arc 共享 + 先 filter 后 clone

**Files:**
- Modify: `src/tasks/matcher.rs:152-158`
- Modify: `src/tasks/matcher_bytes.rs`

**依赖:** Task 15

- [ ] **Step 1: 修改 `bk_tree_plus_linear` 使用 `Arc<Vec<u64>>` 共享**

```rust
pub fn bk_tree_plus_linear(hashes: Vec<u64>) -> Self {
    let hashes = Arc::new(hashes);
    let matchers: Vec<Box<dyn HashMatcher>> = vec![
        Box::new(BkTreeMatcher::new(Arc::clone(&hashes))),
        Box::new(LinearScanMatcher::new(Arc::clone(&hashes))),
    ];
    Self::new(matchers, ChainStrategy::FirstHit)
}
```

- [ ] **Step 2: 修改 `LinearScanMatcher` 和 `BkTreeMatcher` 接受 `Arc<Vec<u64>>`**

```rust
pub struct LinearScanMatcher {
    hashes: Arc<Vec<u64>>,
}

impl LinearScanMatcher {
    pub fn new(hashes: Arc<Vec<u64>>) -> Self {
        Self { hashes }
    }
}
```

- [ ] **Step 3: 同步修改 `matcher_bytes.rs` 的 `bk_tree_plus_linear` 使用 `Arc<Vec<HashBytes>>`**

- [ ] **Step 4: 修改 `LinearScanMatcherBytes::find_similar` 先 filter 后 clone**

```rust
fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes> {
    self.hashes.iter().enumerate()
        .filter_map(|(i, h)| {
            let dist = hamming_distance(query, h);
            if dist <= threshold {
                Some((i, dist))  // 先持有索引，不 clone
            } else {
                None
            }
        })
        .map(|(i, dist)| MatchResultBytes {
            hash: Arc::clone(&self.hashes[i]),  // 仅匹配项 clone
            distance: dist,
        })
        .collect()
}
```

- [ ] **Step 5: 修改 `MatchResultBytes` 持有 `Arc<HashBytes>` 替代 `HashBytes`**

- [ ] **Step 6: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 7: 运行测试验证**

Run: `cargo test --test matcher_test --test matcher_bytes_test`
Expected: 测试通过

- [ ] **Step 8: 提交**

```bash
git add src/tasks/matcher.rs src/tasks/matcher_bytes.rs
git commit -m "perf(matcher): Arc 共享哈希数据 + 先 filter 后 clone"
```

---

### Task 26: P3 — `hash_bytes_to_u32` 缓存

**Files:**
- Modify: `src/tasks/gpu_matcher.rs`

**依赖:** Task 24

- [ ] **Step 1: 为 `GpuHashMatcherBytes` 增加 u32 转换缓存**

```rust
pub struct GpuHashMatcherBytes {
    // ... 现有字段 ...
    /// 缓存最近一次 database 的 u32 转换结果
    cached_db_u32: RefCell<Option<(Vec<HashBytes>, Vec<u32>)>>,
}
```

- [ ] **Step 2: 在 `find_nearest_neighbors` 中检查缓存**

```rust
fn find_nearest_neighbors(&self, queries: &[HashBytes], database: &[HashBytes], threshold: u32) {
    let db_u32 = {
        let mut cache = self.cached_db_u32.borrow_mut();
        if let Some((ref cached_db, ref cached_u32)) = cache.as_ref() {
            if cached_db.as_slice() == database {
                cached_u32.clone()
            } else {
                let u32_data = hash_bytes_to_u32(database);
                *cache = Some((database.to_vec(), u32_data.clone()));
                u32_data
            }
        } else {
            let u32_data = hash_bytes_to_u32(database);
            *cache = Some((database.to_vec(), u32_data.clone()));
            u32_data
        }
    };
    // ... 使用 db_u32 ...
}
```

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行测试验证**

Run: `cargo test --test gpu_matcher_test`
Expected: 测试通过

- [ ] **Step 5: 提交**

```bash
git add src/tasks/gpu_matcher.rs
git commit -m "perf(gpu_matcher): hash_bytes_to_u32 缓存避免重复转换"
```

---

### Task 27: P4 业务层 — `GpuImageMatcher::index_database`

**Files:**
- Modify: `src/tasks/gpu_image_matcher.rs`

**依赖:** Task 15

- [ ] **Step 1: 新增 `GpuImageMatcherIndex` 结构体**

```rust
/// 预计算的 database 索引
pub struct GpuImageMatcherIndex {
    /// database 哈希列表
    hashes: Vec<HashBytes>,
    /// 可选的 GPU 缓冲区，缓存 database 哈希的 u32 表示
    gpu_buffer: Option<GpuBuffer>,
    /// 每个 hash 的 u32 数量
    u32_per_hash: usize,
}
```

- [ ] **Step 2: 实现 `index_database` 方法**

```rust
impl GpuImageMatcher {
    pub fn index_database(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dims: &[(u32, u32)],
    ) -> Result<GpuImageMatcherIndex, GpuError> {
        let hashes = self.compute_hashes(ctx, images, dims)?;
        Ok(GpuImageMatcherIndex {
            hashes,
            gpu_buffer: None,
            u32_per_hash: self.hash_size.u32s_per_hash() as usize,
        })
    }
}
```

- [ ] **Step 3: 实现 `find_similar_with_index` 方法**

```rust
pub fn find_similar_with_index(
    &self,
    ctx: &GpuContext,
    query_images: &[Vec<u8>],
    query_dims: &[(u32, u32)],
    index: &GpuImageMatcherIndex,
    threshold: u32,
) -> Result<Vec<Vec<MatchResultBytes>>, GpuError> {
    let query_hashes = self.compute_hashes(ctx, query_images, query_dims)?;
    // 复用 index.hashes，无需重新计算 database
    // ... 距离矩阵计算 ...
}
```

- [ ] **Step 4: 合并 query + database 哈希计算为一次 GPU submit**

在 `index_database` 和 `find_similar_with_index` 中，将 query 和 database 的哈希计算合并到单次 `GpuBatchSubmitter` 提交。

- [ ] **Step 5: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 6: 运行测试验证**

Run: `cargo test --test gpu_image_matcher_test`
Expected: 测试通过

- [ ] **Step 7: 提交**

```bash
git add src/tasks/gpu_image_matcher.rs
git commit -m "feat(gpu_image_matcher): 新增 index_database 预计算 database 哈希"
```

---

### Task 28: P4 业务层 — dihedral 批量查询

**Files:**
- Modify: `src/tasks/matcher.rs:297-312`

**依赖:** Task 22, Task 25

- [ ] **Step 1: 修改 `find_similar_dihedral` 批量提交 8 变体**

```rust
fn find_similar_dihedral(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
    let variants = DihedralHashes64::from_u64(query).all_variants();  // [u64; 8]
    let batch_results = self.matcher.find_similar_batch(&variants, threshold);

    // 合并 8 个变体的结果，去重
    let mut seen = HashSet::new();
    batch_results.into_iter().flatten()
        .filter(|r| seen.insert(r.hash))
        .collect()
}
```

- [ ] **Step 2: 在 `HashMatcher` trait 新增 `find_similar_batch` 方法**

```rust
pub trait HashMatcher {
    // ... 现有方法 ...

    /// 批量查询多个 query
    fn find_similar_batch(&self, queries: &[u64], threshold: u32) -> Vec<Vec<MatchResult>> {
        queries.iter().map(|q| self.find_similar(*q, threshold)).collect()
    }
}
```

- [ ] **Step 3: GPU 路径下 8 变体合并为单次 GPU dispatch**

在 `GpuHashMatcherFacade` 中重写 `find_similar_batch`，将 8 个 query 合并为单次 GPU dispatch，8× 加速。

- [ ] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 5: 运行测试验证**

Run: `cargo test --test matcher_test`
Expected: 测试通过

- [ ] **Step 6: 提交**

```bash
git add src/tasks/matcher.rs
git commit -m "perf(matcher): dihedral 8 变体批量查询合并为单次 GPU dispatch"
```

---

### Task 29: P5 — `phasher.rs` 尺寸分组批量处理

**Files:**
- Modify: `src/tasks/phasher.rs`（`compute_gpu_per_image_pipeline`）

**依赖:** Task 9, Task 15

- [ ] **Step 1: 实现 `compute_gpu_per_image_pipeline` 按尺寸分桶**

```rust
fn compute_gpu_per_image_pipeline(
    &self,
    ctx: &GpuContext,
    images: &[Vec<u8>],
    dimensions: &[(u32, u32)],
) -> Result<Vec<u64>, GpuError> {
    // 按尺寸分桶
    let mut groups: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, &(w, h)) in dimensions.iter().enumerate() {
        groups.entry((w, h)).or_default().push(i);
    }

    let mut all_hashes = vec![0u64; images.len() * self.computer.u64s_per_image()];
    for ((w, h), indices) in groups {
        let group_images: Vec<&Vec<u8>> = indices.iter().map(|&i| &images[i]).collect();
        let group_dims = vec![(w, h); indices.len()];
        let group_hashes = self.compute_gpu_batch_pipeline(ctx, &group_images, &group_dims)?;
        // 按 indices 写回 all_hashes
        for (i, &idx) in indices.iter().enumerate() {
            let start = idx * self.computer.u64s_per_image();
            let end = start + self.computer.u64s_per_image();
            all_hashes[start..end].copy_from_slice(&group_hashes[i * self.computer.u64s_per_image()..(i + 1) * self.computer.u64s_per_image()]);
        }
    }
    Ok(all_hashes)
}
```

- [ ] **Step 2: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 3: 运行测试验证**

Run: `cargo test --test phasher_test`
Expected: 测试通过

- [ ] **Step 4: 提交**

```bash
git add src/tasks/phasher.rs
git commit -m "perf(phasher): 逐图路径按尺寸分桶批量处理"
```

---

### Task 30: P5 — `preprocess_gpu` 合并到 `GpuBatchSubmitter`

**Files:**
- Modify: `src/tasks/phasher.rs`（`preprocess_gpu`）
- Modify: `src/tasks/phasher_util.rs`（`merge_gpu_buffers`）

**依赖:** Task 29

- [ ] **Step 1: 修改 `preprocess_gpu` 接受 `&mut GpuBatchSubmitter`**

```rust
fn preprocess_gpu(
    &self,
    ctx: &GpuContext,
    batch: &mut GpuBatchSubmitter,
    images: &[Vec<u8>],
    dims: &[(u32, u32)],
) -> Result<Vec<GpuBuffer>, GpuError> {
    // 合并所有图像模糊操作到单次 submit
}
```

- [ ] **Step 2: 标记 `merge_gpu_buffers` 非 batch 版本为 `#[deprecated]`**

```rust
#[deprecated(note = "使用 batch 版本替代，将 copy 命令编码到后续 encoder")]
pub fn merge_gpu_buffers(/* ... */) -> Result<GpuBuffer, GpuError> { /* ... */ }
```

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行测试验证**

Run: `cargo test --test phasher_test`
Expected: 测试通过

- [ ] **Step 5: 提交**

```bash
git add src/tasks/phasher.rs src/tasks/phasher_util.rs
git commit -m "perf(phasher): preprocess_gpu 合并到 GpuBatchSubmitter"
```

---

### Task 31: P5 — Mean/Median GPU 阈值计算

**Files:**
- Modify: `src/tasks/phasher.rs`（`compute_hash_gpu` Mean/Median 路径）
- 新增: `src/tasks/threshold_reduce.wgsl`

**依赖:** Task 15

- [ ] **Step 1: 新增 `threshold_reduce.wgsl` 实现 reduce kernel**

```wgsl
@compute @workgroup_size(256)
fn reduce_mean(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // 标准 reduce sum + 除法计算均值
    var<workgroup> shared: array<f32, 256>;
    // ... reduce 逻辑 ...
}
```

- [ ] **Step 2: 在 `phasher.rs` Mean 路径调用 GPU 阈值计算**

```rust
fn compute_mean_threshold_gpu(
    ctx: &GpuContext,
    resized_buffer: &GpuBuffer,
) -> Result<f32, GpuError> {
    // 调用 reduce_mean 着色器
    // 输出 uniform buffer，消除 GPU↔CPU 往返
}
```

- [ ] **Step 3: Median 路径保持 CPU 预计算或实现 bitonic sort**

中位数 GPU 实现复杂，保持 CPU 预计算（通过 `f32::to_bits()` 传入），并在注释中说明。

- [ ] **Step 4: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 5: 运行测试验证**

Run: `cargo test --test mean_hash_test --test median_hash_test`
Expected: 测试通过

- [ ] **Step 6: 提交**

```bash
git add src/tasks/phasher.rs src/tasks/threshold_reduce.wgsl
git commit -m "perf(phasher): Mean 阈值改 GPU reduce kernel 消除 GPU↔CPU 往返"
```

---

### Task 32: P5 — `buffer_pool.rs` 配置优化

**Files:**
- Modify: `src/buffer_pool.rs`（`release_staging` + `BufferPoolConfig`）

**依赖:** Task 15

- [ ] **Step 1: 统一 `release_staging` 与 `release` 的大缓冲区策略**

修改 `release_staging`，使 `large_buffer_cache` 配置对 staging pool 同样生效。

- [ ] **Step 2: 在 `BufferPoolConfig` 增加 `image_friendly_classes` 选项**

```rust
pub struct BufferPoolConfig {
    // ... 现有字段 ...
    /// 图像场景的非 2 倍档位（如 1920×1080×4 = 8294400 字节）
    pub image_friendly_classes: Vec<u64>,
}
```

- [ ] **Step 3: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 4: 运行测试验证**

Run: `cargo test`
Expected: 测试通过

- [ ] **Step 5: 提交**

```bash
git add src/buffer_pool.rs
git commit -m "perf(buffer_pool): 统一 release_staging 策略 + image_friendly_classes"
```

---

### Task 33: P5 — `gaussian_blur.rs` 核缓存

**Files:**
- Modify: `src/tasks/gaussian_blur.rs`

**依赖:** Task 15

- [ ] **Step 1: 缓存 `(kernel_size, sigma) → kernel_1d` 映射**

```rust
pub struct GpuGaussianBlur {
    // ... 现有字段 ...
    kernel_cache: RefCell<HashMap<(u32, f32), Vec<f32>>>,
}

impl GpuGaussianBlur {
    fn get_kernel(&self, kernel_size: u32, sigma: f32) -> Vec<f32> {
        let mut cache = self.kernel_cache.borrow_mut();
        cache.entry((kernel_size, sigma)).or_insert_with(|| {
            generate_gaussian_kernel(kernel_size, sigma)
        }).clone()
    }
}
```

- [ ] **Step 2: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 3: 运行测试验证**

Run: `cargo test --test gaussian_blur_test`
Expected: 测试通过

- [ ] **Step 4: 提交**

```bash
git add src/tasks/gaussian_blur.rs
git commit -m "perf(gaussian_blur): 缓存 kernel_1d 避免重复计算"
```

---

### Task 34: P6 — 新增 `bincode` + `serde` dev-dependencies

**Files:**
- Modify: `Cargo.toml`

**依赖:** Task 15

- [ ] **Step 1: 在 `Cargo.toml` `[dev-dependencies]` 新增依赖**

```toml
[dev-dependencies]
bincode = "1.3"
serde = { version = "1.0", features = ["derive"] }
```

- [ ] **Step 2: 编译验证**

Run: `cargo build`
Expected: 编译通过

- [ ] **Step 3: 提交**

```bash
git add Cargo.toml
git commit -m "chore: 新增 bincode + serde dev-dependencies 用于缓存数据测试"
```

---

### Task 35: P6 — 创建 `tests/common/cache_loader.rs`

**Files:**
- Create: `tests/common/cache_loader.rs`

**依赖:** Task 34

- [ ] **Step 1: 实现 Czkawka 缓存文件解析**

```rust
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[derive(Deserialize)]
pub struct ImagesEntry {
    pub path: String,
    pub hash: Vec<u8>,
    pub dimensions: (u32, u32),
}

/// 从 Czkawka bincode 缓存文件加载哈希
pub fn load_hashes_from_cache(path: &str) -> Result<Vec<ImagesEntry>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    let entries: Vec<ImagesEntry> = bincode::deserialize(&data)?;
    Ok(entries)
}
```

- [ ] **Step 2: 编译验证**

Run: `cargo build --tests`
Expected: 编译通过

- [ ] **Step 3: 提交**

```bash
git add tests/common/cache_loader.rs
git commit -m "test: 新增 cache_loader 解析 Czkawka bincode 缓存"
```

---

### Task 36: P6 — 创建 `tests/cache_perf_test.rs`

**Files:**
- Create: `tests/cache_perf_test.rs`

**依赖:** Task 35

- [ ] **Step 1: 实现缓存数据性能验证测试**

```rust
mod common;

use common::cache_loader::load_hashes_from_cache;

#[test]
fn test_cache_gpu_nearest_neighbor_perf() {
    let entries = load_hashes_from_cache("data/cache_similar_images_32_Gradient_Gaussian_100.bin").unwrap();
    let hashes: Vec<HashBytes> = entries.iter().map(|e| HashBytes::from_vec(e.hash.clone())).collect();

    // 构建 BkTreeBytes，验证搜索性能
    let bktree = BkTreeBytes::new(&hashes);
    // ... 性能断言 ...

    // 上传缓存哈希到 GPU，验证 GPU 汉明距离矩阵计算性能
    let ctx = GpuContext::new_sync(ComputeBackend::Gpu).unwrap();
    let matcher = GpuHashMatcherBytes::new(&ctx, 1024).unwrap();
    // ... 性能断言 ...

    // 比较缓存哈希与 GPU 重新计算的哈希，验证正确性
    // ... 正确性断言 ...
}

#[test]
fn test_oom_degradation() {
    // 验证 OOM 降级路径（大图像批量场景）
}
```

- [ ] **Step 2: 编译验证**

Run: `cargo build --tests`
Expected: 编译通过

- [ ] **Step 3: 运行测试验证**

Run: `cargo test --test cache_perf_test`
Expected: 测试通过

- [ ] **Step 4: 提交**

```bash
git add tests/cache_perf_test.rs
git commit -m "test: 新增 cache_perf_test 缓存数据性能验证"
```

---

### Task 37: P6 — 创建 `benches/cache_matcher_bench.rs`

**Files:**
- Create: `benches/cache_matcher_bench.rs`

**依赖:** Task 35

- [ ] **Step 1: 实现缓存数据驱动的基准测试**

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use gpgpu_tool::*;

fn bench_bktree_vs_linear(c: &mut Criterion) {
    let entries = load_hashes_from_cache("data/cache_similar_images_32_Gradient_Gaussian_100.bin").unwrap();
    let hashes: Vec<HashBytes> = entries.iter().map(|e| HashBytes::from_vec(e.hash.clone())).collect();

    let bktree = BkTreeBytes::new(&hashes);
    let linear = LinearScanMatcherBytes::new(hashes.clone());

    c.bench_function("bktree_search", |b| {
        b.iter(|| bktree.find_similar(&hashes[0], 20))
    });

    c.bench_function("linear_search", |b| {
        b.iter(|| linear.find_similar(&hashes[0], 20))
    });
}

fn bench_gpu_vs_cpu(c: &mut Criterion) {
    // GPU 汉明距离矩阵 vs CPU 线性扫描性能对比
}

criterion_group!(benches, bench_bktree_vs_linear, bench_gpu_vs_cpu);
criterion_main!(benches);
```

- [ ] **Step 2: 在 `Cargo.toml` 注册 benchmark**

```toml
[[bench]]
name = "cache_matcher_bench"
harness = false
```

- [ ] **Step 3: 编译验证**

Run: `cargo build --benches`
Expected: 编译通过

- [ ] **Step 4: 运行基准测试建立基线**

Run: `cargo bench --bench cache_matcher_bench`
Expected: 基准测试运行完成，输出性能数据

- [ ] **Step 5: 提交**

```bash
git add benches/cache_matcher_bench.rs Cargo.toml
git commit -m "bench: 新增 cache_matcher_bench 缓存数据驱动基准测试"
```

---

### Task 38: 阶段 2 集成验证 + 最终回归

**Files:** 无（仅验证）

**依赖:** Task 16-37 全部完成

- [ ] **Step 1: 运行完整测试套件**

Run: `cargo test`
Expected: 所有测试通过

- [ ] **Step 2: 运行 PDQ feature 测试**

Run: `cargo test --features pdq`
Expected: 所有测试通过

- [ ] **Step 3: 运行 clippy**

Run: `cargo clippy`
Expected: 无错误

- [ ] **Step 4: 运行 clippy with image feature**

Run: `cargo clippy --features image`
Expected: 无错误

- [ ] **Step 5: 运行 clippy with pdq feature**

Run: `cargo clippy --features pdq`
Expected: 无错误

- [ ] **Step 6: 运行基准测试对比优化前后性能**

Run: `cargo bench`
Expected: 性能数据符合预期（GPU 最近邻 20000×20000 保持 2.2s 或更优，距离矩阵 500×500 从 14.6ms 提升）

- [ ] **Step 7: 手动验证 DX12 后端 OOM 降级**

在大图像批量场景下手动测试，验证 OOM 后正确降级到 CPU 而非崩溃。

- [ ] **Step 8: 提交阶段 2 完成标记**

```bash
git commit --allow-empty -m "chore: 阶段 2（业务层优化 P2+P3+P5+P6）完成"
```

---

## 自检清单

### Spec 覆盖率

- ✅ P0 显存安全（Task 1-6）：四层防御 + OOM 降级
- ✅ P1 GPU 同步（Task 7-9, 14）：wait_all 保留引用 + 解放 DualEncoderSubmitter + 跨 chunk 复用 + download 推广
- ✅ P2 着色器（Task 16-21）：resize SAT + block_hash 预计算 + PDQ cos 表 + convolution LDS + hamming 并行归约 + PDQ CPU 快速 DCT
- ✅ P3 内存管理（Task 22-26）：dihedral 64/256-bit 位操作 + 距离矩阵扁平化 + Arc 共享 + 先 filter 后 clone + hash_bytes_to_u32 缓存
- ✅ P4 抽象边界（Task 10-13, 27-28）：UniformPool 下沉 + PipelineCache RefCell + LRU O(1) + index_database + dihedral 批量查询
- ✅ P5 逐图路径（Task 29-33）：尺寸分组 + 预处理合并 + Mean/Median GPU 阈值 + buffer_pool 配置 + gaussian_blur 核缓存
- ✅ P6 验证（Task 34-38）：bincode 依赖 + cache_loader + cache_perf_test + cache_matcher_bench + 集成验证

### 关键决策点验证

- ✅ P0 `BufferPool::acquire` 直接改签名为 `Result<Buffer, GpuError>`（Task 2-3）
- ✅ P2 着色器全部 5 项实施（Task 16-20）
- ✅ P4 Uniform 管理下沉到 `GpuContext`，新增 `UniformPool`（Task 10-11）

### 任务依赖关系

- 阶段 1（Task 1-15）按 P0 → P1 → P4 顺序，能力层先稳定
- 阶段 2（Task 16-38）依赖阶段 1 完成，按 P2 → P3 → P5 → P6 顺序
- 单个任务不超过 200 行代码变更（除 Task 3 调用点同步、Task 12 PipelineCache 重构等结构性变更）

### 风险缓解

- `acquire` breaking change：minor 版本升级，编译期发现遗漏调用点（Task 3）
- OOM 后 GPU 状态损坏：降级为终态，进程重启恢复（Task 6）
- `UniformPool` 增加抽象：符合长期目标，消除业务层绕过封装（Task 10-11）
- `DualEncoderSubmitter` 默认启用：保留 feature flag 回退（Task 8）
- resize 积分图显存占用：仅大比例下采样启用，结合 P0 分批保护（Task 16）
- hamming 并行归约后端差异：保留串行路径 fallback（Task 20）
- dihedral 位操作正确性：exhaustive 测试对比 Vec<bool> 结果（Task 22-23）
