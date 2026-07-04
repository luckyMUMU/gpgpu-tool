---
comet_change: performance-optimization
role: technical-design
canonical_spec: openspec
---

# 性能优化与显存安全技术设计

## 1. 概述

本设计文档基于 2026-06-16 全代码库静态分析，覆盖两大目标：

1. **显存安全**：解决 DX12 后端 OOM 时进程被直接终止的问题，实现四层防御 + 运行时 CPU 降级
2. **性能优化**：解决 GPU 同步开销、着色器算法效率、内存管理、抽象泄漏四类瓶颈

采用**能力层优先 + 业务层跟进**的两阶段策略，确保能力层（长期目标核心）先稳定，业务层（短期目标核心）基于新 API 优化。

## 2. 实施策略：两阶段方案

### 阶段 1：能力层重构（P0 + P1 + P4）

聚焦 `context.rs`/`pipeline.rs`/`batch.rs`/`buffer*.rs` 的重构，让能力层成为"GPU 类型无关抽象"的稳定基座。

**包含**：
- P0 显存安全（四层防御 + OOM 降级）
- P1 GPU 同步（wait_all 保留引用 + 解放 DualEncoderSubmitter + 跨 chunk 复用）
- P4 抽象边界（UniformPool 下沉 GpuContext + PipelineCache RefCell 化 + LRU O(1)）

**理由**：P4 的 Uniform 下沉到 GpuContext 是架构性变更，若在 P1/P3 之后做会导致业务层代码返工。能力层先稳定符合 ADR-006 的"能力层封装质量维护红线"。

### 阶段 2：业务层优化（P2 + P3 + P5 + P6）

基于阶段 1 稳定的能力层 API，优化业务层算法效率、内存管理和逐图路径。

**包含**：
- P2 着色器效率（resize SAT + block_hash 预计算 + PDQ cos 表 + convolution LDS + hamming 并行归约）
- P3 内存管理（dihedral 位操作 + 距离矩阵扁平化 + Arc 共享 + 先 filter 后 clone）
- P5 逐图路径（尺寸分组 + 预处理合并 + Mean/Median GPU 阈值计算）
- P4 业务层部分（index_database + dihedral 批量查询）
- P6 验证（Czkawka 缓存数据 + DX12 OOM 降级）

## 3. 阶段 1 详细设计

### 3.1 P0 显存安全 — 四层防御

#### 层 1：UncapturedErrorHandler 注册

**位置**：`src/context.rs`

在模块级定义原子标志：

```rust
use std::sync::atomic::{AtomicBool, Ordering};

static GPU_OOM_FLAG: AtomicBool = AtomicBool::new(false);
static GPU_DEVICE_LOST_FLAG: AtomicBool = AtomicBool::new(false);
```

在 `init_gpu`（`context.rs:202-280`）和 `init_software_adapter`（`context.rs:284-334`）中，Device 创建后立即注册：

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

**约束**：handler 内不可 panic，否则触发 abort。

#### 层 2：BufferPool::acquire 预检查（breaking change）

**位置**：`src/buffer_pool.rs:157-177`

`acquire` 返回类型从 `Buffer` 改为 `Result<Buffer, GpuError>`：

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
    // ... 现有池化逻辑 ...
}
```

**调用点修改**：约 15 处调用点改用 `?` 传播错误。涉及文件：
- `src/buffer.rs`（download_with_pool 等）
- `src/batch.rs`（ensure_initialized）
- `src/tasks/hash_common.rs`（compute_phash）
- `src/tasks/phasher_util.rs`（upload_image_to_gpu）
- `src/tasks/gpu_matcher.rs`（compute_distance_matrix 等）
- `src/tasks/convolution.rs`、`gpu_resize.rs`、`pdq_hash.rs`、`sha256.rs`

#### 层 3：compute_phash 内部分批

**位置**：`src/tasks/hash_common.rs:267-322`

```rust
pub fn compute_phash(...) -> Result<Vec<u64>, GpuError> {
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

    // 按显存预算分批
    let max_batch = (DEFAULT_MAX_BATCH_SIZE / per_image_bytes).max(1) as usize;

    let mut all_hashes = Vec::with_capacity(images.len());
    for chunk in images.chunks(max_batch) {
        let chunk_hashes = compute_phash_single_batch(pipeline, ctx, chunk, ...)?;
        all_hashes.extend(chunk_hashes);
    }
    Ok(all_hashes)
}
```

**批次大小估算**：
- 单张图显存占用 = `src_u32_per_image × 4` (输入) + `u32s_per_image × 4` (输出) + staging
- 安全系数 = 0.5（留 50% 余量）
- `max_batch = max_batch_size × 安全系数 / 单张图显存占用`
- `max_batch_size` 默认 128 MB（`phasher.rs:142` `DEFAULT_MAX_BATCH_SIZE`）

`compute_phash_from_gpu_buffer`（`hash_common.rs:497-535`）同步增加分批逻辑。

#### 层 4：运行时 OOM → CPU 降级

**位置**：`src/context.rs` `backend()` 方法

```rust
impl GpuContext {
    pub fn backend(&self) -> ComputeBackend {
        if self.backend == ComputeBackend::Gpu && GPU_OOM_FLAG.load(Ordering::SeqCst) {
            log::warn!("检测到 GPU OOM，切换到 CPU 降级模式");
            return ComputeBackend::Cpu;
        }
        self.backend
    }
}
```

`DefaultBackendDispatcher::dispatch_gpu`（`backend_dispatcher.rs:47-66`）自动响应 `backend()` 动态切换，业务层无感知。

**风险缓解**：OOM 后 GPU 状态可能已损坏。降级为终态，所有后续操作走 CPU，不再访问 GPU。进程重启恢复。

### 3.2 P1 GPU 同步 — 连续批量提交

#### wait_all 保留引用

**位置**：`src/batch.rs:268-270, 324-326, 341-343`

`GpuBatchSubmitter::wait_all()` 修改为仅清空 `encoder` 和 `pending`，保留 `device/queue/pool`：

```rust
fn wait_all(&mut self) -> Result<Vec<Vec<u8>>, GpuError> {
    // ... 现有 submit + poll + 读取逻辑 ...
    self.encoder = None;      // 仅清空 encoder
    self.pending.clear();     // 仅清空 pending
    // 保留 device/queue/pool，与 flush() 语义一致
}
```

`DualEncoderSubmitter::wait_all()`（`batch.rs:532-627`）同步修改。

#### 解放 DualEncoderSubmitter

**位置**：`src/batch.rs:389, 414, 653, 660`

移除 `#[cfg(feature = "dual-encoder")]`，作为默认实现。`dual-encoder` feature 改为默认启用，保留作为回退选项。

#### 跨 chunk 复用 submitter

**位置**：`src/tasks/phasher_pipeline.rs:110, 217`

将 `batch` 提升到循环外：

```rust
let mut batch = GpuBatchSubmitter::new();
for &(start, end) in &chunks {
    batch.ensure_initialized(ctx)?;  // 首次初始化，后续复用
    // ... 编码 dispatch ...
    let results = batch.wait_all()?;  // 仅清空 encoder/pending，保留引用
}
```

### 3.3 P4 抽象边界 — Uniform 下沉到 GpuContext

#### 新增 UniformPool

**位置**：`src/context.rs` 新增 `UniformPool` 结构

```rust
struct UniformPool {
    /// 按 size tier 缓存的可复用 Uniform 缓冲区
    buffers: RefCell<HashMap<u64, Vec<GpuBuffer>>>,
    /// 每 tier 最大缓存数
    max_per_tier: usize,
}

impl UniformPool {
    /// 获取指定大小的 Uniform 缓冲区
    fn acquire(&self, device: &Device, size: u64) -> GpuBuffer {
        let mut buffers = self.buffers.borrow_mut();
        let tier = Self::size_tier(size);
        if let Some(pool) = buffers.get_mut(&tier) {
            if let Some(buffer) = pool.pop() {
                return buffer;
            }
        }
        GpuBuffer::create(device, tier, BufferUsage::Uniform)
    }

    /// 归还 Uniform 缓冲区
    fn release(&self, buffer: GpuBuffer) {
        let mut buffers = self.buffers.borrow_mut();
        let tier = Self::size_tier(buffer.size());
        let pool = buffers.entry(tier).or_insert_with(Vec::new);
        if pool.len() < self.max_per_tier {
            pool.push(buffer);
        }
    }
}
```

`GpuContext` 新增字段：

```rust
struct GpuContext {
    // ... 现有字段 ...
    uniform_pool: UniformPool,
}
```

API：

```rust
impl GpuContext {
    pub fn acquire_uniform(&self, size: u64) -> GpuBuffer {
        self.uniform_pool.acquire(self.device()?, size)
    }

    pub fn release_uniform(&self, buffer: GpuBuffer) {
        self.uniform_pool.release(buffer);
    }
}
```

#### ComputePipeline 改用 UniformPool

**位置**：`src/pipeline.rs:544-609`

`dispatch_with_params` 和 `encode_dispatch_with_params_into` 都通过 `ctx.acquire_uniform` 获取 Uniform 缓冲区：

```rust
pub fn dispatch_with_params(
    &self,
    ctx: &GpuContext,
    params: &[u8],
    bindings: &[&GpuBuffer],
    workgroup_count: [u32; 3],
) -> Result<(), GpuError> {
    let uniform_buffer = ctx.acquire_uniform(params.len() as u64);
    ctx.queue()?.write_buffer(uniform_buffer.raw(), 0, params);
    // ... dispatch ...
    ctx.release_uniform(uniform_buffer);
    Ok(())
}
```

消除 `pipeline.rs` 内部的 `reusable_uniform: RefCell<Option<GpuBuffer>>`，统一到 `GpuContext`。消除 `hash_common.rs:599` 业务层绕过封装。

#### PipelineCache RefCell 化

**位置**：`src/context.rs:34-67, 428-438`

`PipelineCache` 的 `entries`、`generations`、`next_gen` 改为 `RefCell`：

```rust
struct PipelineCache {
    entries: RefCell<HashMap<u64, Arc<ComputePipeline>>>,
    generations: RefCell<HashMap<u64, u64>>,
    next_gen: RefCell<u64>,
    max_entries: usize,
}

impl PipelineCache {
    pub fn get_or_create(
        &self,  // ← 改为 &self
        device: &Device,
        descriptor: &PipelineDescriptor,
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        // ... 使用 .borrow_mut() ...
    }
}
```

`GpuContext::get_or_create_pipeline` 接受 `&self`，与 `BindGroupCache` 设计一致。

#### PipelineCache LRU O(1)

**位置**：`src/context.rs:76-87`

LRU 淘汰改为 arena + 双向链表，与 `BindGroupCache`（`pipeline.rs:174-220`）一致：

```rust
struct PipelineCache {
    entries: RefCell<HashMap<u64, Entry>>,
    arena: RefCell<Vec<Node>>,
    head: RefCell<Option<usize>>,  // LRU 链表头
    tail: RefCell<Option<usize>>,  // LRU 链表尾
    max_entries: usize,
}
```

淘汰时 O(1) 移除 tail 节点。

## 4. 阶段 2 详细设计

### 4.1 P2 着色器效率优化

#### resize.wgsl 积分图（SAT）

**位置**：`src/tasks/resize.wgsl` + `gpu_resize.rs`

新增积分图构建 pass：

```wgsl
@compute @workgroup_size(16, 16)
fn compute_sat(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @group(0) @binding(0) src: ptr<storage, array<u32>, read>,
    @group(0) @binding(1) sat: ptr<storage, array<u32>, read_write>,
    @group(0) @binding(2) params: ptr<uniform, SatParams>,
) {
    // SAT[y * width + x] = src[y][x] + SAT[y-1][x] + SAT[y][x-1] - SAT[y-1][x-1]
    // 需两趟：水平前缀和 + 垂直前缀和
}
```

查询 pass：

```wgsl
@compute @workgroup_size(8, 8)
fn resize_sat(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @group(0) @binding(0) sat: ptr<storage, array<u32>, read>,
    @group(0) @binding(1) dst: ptr<storage, array<u32>, read_write>,
    @group(0) @binding(2) params: ptr<uniform, ResizeParams>,
) {
    // O(1) 查询：sum = SAT[x1,y1] - SAT[x0,y1] - SAT[x1,y0] + SAT[x0,y0]
    // mean = sum / ((x1-x0) * (y1-y0))
}
```

**启用条件**：仅在大比例下采样场景（src/dst ≥ 4×）启用，小图保持区域平均。显存开销：增加 `src_w × src_h × 4` 字节。

#### block_hash.wgsl 预计算

**位置**：`src/tasks/block_hash.wgsl`

新增预处理 pass `compute_block_means`：将 64×64 图像下采样为 8×8 block 均值图（64 个 u32）。

主 pass `block_hash_with_means`：从 64 个 block 均值读取，消除 O(block_area) 循环。提取 `m_current = block_mean(bx, by)` 一次，复用于 left/top 分支。

#### pdq_hash.wgsl cos 表 + 系数裁剪

**位置**：`src/tasks/pdq_hash.wgsl` + `pdq_hash.rs`

- 预计算 64×64 余弦查找表作为 storage buffer 传入（16KB）
- workgroup 改为 2D (8,8,1)，利用 LDS 共享 cos 表
- 新增第 3 个 pass `extract_low_freq`：提取 16×16 低频系数，减少 93.75% 下载量

#### convolution.wgsl Full2D LDS

**位置**：`src/tasks/convolution.wgsl:120-130`

新增 `lds_2d: array<f32, (8+2r)*(8+2r)>` 共享内存。协作加载 halo 区域像素到 LDS。卷积计算从 LDS 读取，消除全局内存重复读取。

#### hamming.wgsl 并行归约

**位置**：`src/tasks/hamming.wgsl:183-200`

`find_nearest_neighbor` 改为 256 线程协作扫描 256 db：

```wgsl
@compute @workgroup_size(256)
fn find_nearest_neighbor_parallel(...) {
    // 每线程计算 1 个 db 条目的距离
    let dist = compute_distance(query, db[local_id]);
    
    // warp shuffle 归约最小距离
    var best_dist = dist;
    var best_idx = local_id;
    for (var offset = 16u; offset > 0u; offset >>= 1u) {
        let other_dist = subgroupShuffle(best_dist, local_id ^ offset);
        let other_idx = subgroupShuffle(best_idx, local_id ^ offset);
        if (other_dist < best_dist) {
            best_dist = other_dist;
            best_idx = other_idx;
        }
    }
    
    // workgroup 内归约（跨 warp）
    // ... shared memory 归约 ...
}
```

`hamming_distance_matrix` 加载阶段：所有 256 线程协作加载 query_tile 和 db_tile_matrix。

### 4.2 P3 内存管理优化

#### dihedral.rs 位操作

**位置**：`src/tasks/dihedral.rs:19-109, 114-210`

64-bit：参考 32×32 的 `transpose_32x32` delta-swap，直接在 u64 上实现转置/翻转/旋转：

```rust
fn transpose_8x8(hash: u64) -> u64 {
    // delta-swap 转置 8×8 位矩阵
    let mut x = hash;
    x = ((x & 0xAA00AA00AA00AA00) >> 7) | ((x & 0x0055005500550055) << 7) | (x & 0x55AA55AA55AA55AA);
    x = ((x & 0xCCCC0000CCCC0000) >> 14) | ((x & 0x3333000033330000) << 14) | (x & 0x0000FFFF0000FFFF);
    // ... 更多 delta-swap 步骤 ...
    x
}
```

256-bit：在 `[u64; 4]` 上实现位操作。统一四种尺寸实现风格，抽象 `BitMatrix<N>` 泛型结构。

`all_variants` 返回 `[u64; 8]` 替代 `Vec<u64>`。

#### gpu_matcher.rs 扁平化

**位置**：`src/tasks/gpu_matcher.rs:144-196`

```rust
pub struct DistanceMatrix {
    pub data: Vec<u32>,  // 扁平 n*m
    pub cols: usize,
}

impl DistanceMatrix {
    pub fn row(&self, i: usize) -> &[u32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}
```

`compute_distance_matrix` 返回 `DistanceMatrix` 替代 `Vec<Vec<u32>>`。分块路径直接写入预分配 Vec 的对应区间。

#### matcher.rs Arc 共享

**位置**：`src/tasks/matcher.rs:152-158`

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

`LinearScanMatcherBytes::find_similar` 改为先计算距离再决定是否 clone（先 filter 后 clone）。

`MatchResultBytes` 改为持有索引 `(usize, u32)` 或 `Arc<HashBytes>`，避免每匹配 clone。

### 4.3 P5 逐图路径优化

#### 尺寸分组

**位置**：`src/tasks/phasher.rs` `compute_gpu_per_image_pipeline`

按 `(w, h)` 分桶，每组调用 `compute_gpu_batch_pipeline` 批量处理：

```rust
fn compute_gpu_per_image_pipeline(&self, ctx, images, dimensions) -> Result<Vec<u64>, GpuError> {
    // 按尺寸分桶
    let mut groups: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, &(w, h)) in dimensions.iter().enumerate() {
        groups.entry((w, h)).or_default().push(i);
    }
    
    let mut all_hashes = vec![0u64; images.len() * self.computer.u64s_per_image()];
    for ((w, h), indices) in groups {
        let group_images: Vec<&Vec<u8>> = indices.iter().map(|&i| &images[i]).collect();
        let group_hashes = self.compute_gpu_batch_pipeline(ctx, &group_images, &[(w, h); indices.len()])?;
        // 按 indices 写回 all_hashes
    }
    Ok(all_hashes)
}
```

#### 预处理合并

`preprocess_gpu` 改为接受 `&mut GpuBatchSubmitter`，合并所有图像模糊操作到单次 submit。`merge_gpu_buffers` 的 copy 命令编码到后续 encoder。

#### Mean/Median GPU 阈值计算

编写独立 GPU 阈值计算着色器（reduce kernel）：
- 均值：标准 reduce sum + 除法
- 中位数：排序网络（bitonic sort）或保持 CPU 预计算（中位数 GPU 实现复杂）

在 GPU 端计算每图阈值，输出到 uniform buffer，消除 GPU↔CPU 往返。

### 4.4 P4 业务层抽象边界

#### GpuImageMatcher::index_database

**位置**：`src/tasks/gpu_image_matcher.rs`

```rust
pub struct GpuImageMatcherIndex {
    hashes: Vec<HashBytes>,
    /// 可选的 GPU 缓冲区，缓存 database 哈希的 u32 表示
    gpu_buffer: Option<GpuBuffer>,
    u32_per_hash: usize,
}

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
}
```

合并 query + database 哈希计算为一次 GPU submit。

#### dihedral 批量查询

**位置**：`src/tasks/matcher.rs:297-312`

```rust
fn find_similar_dihedral(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
    let variants = DihedralHashes64::from_u64(query).all();  // [u64; 8]
    let batch_results = self.matcher.find_similar_batch(&variants, threshold);
    
    // 合并 8 个变体的结果，去重
    let mut seen = HashSet::new();
    batch_results.into_iter().flatten()
        .filter(|r| seen.insert(r.hash))
        .collect()
}
```

GPU 路径下 8 变体合并为单次 GPU dispatch，8× 加速。

## 5. 数据流

### 优化前（逐图路径 + 无显存保护）

```
图像1 → [submit] → blur → [submit] → resize → [submit] → hash → [download]
图像2 → [submit] → blur → [submit] → resize → [submit] → hash → [download]
...
N 张图 → 全量上传 → OOM → 进程终止（DX12）
```

### 优化后（分组批量 + 显存安全）

```
显存预算计算 → 单批 max_batch 图像数
同尺寸图像组 → [单次 submit] → blur_all → resize_all → hash_all → [单次 download]
OOM 检测 → UncapturedErrorHandler → CPU 降级
```

## 6. 测试策略

### 单元测试

- **P0 每层防御**：handler 注册、acquire 预检查、分批逻辑、backend 切换
- **P2 每个着色器**：对比优化前后输出一致性
- **P3 内存管理**：dihedral 位操作 exhaustive 对比 Vec<bool> 结果，距离矩阵扁平化对比

### 集成测试

- **DX12 后端 OOM 降级**：大图像批量场景，验证 OOM 后正确降级到 CPU
- **端到端**：Czkawka 缓存数据 20000 条 1024-bit 哈希，验证 GPU 最近邻性能

### 回归测试

- 现有 23 个测试文件全部通过（含 CPU 降级路径）
- `cargo test --features pdq` 验证 PDQ 路径

### 基准测试

- `benches/cache_matcher_bench.rs`：优化前后性能对比
- GPU 最近邻（20000×20000）目标：保持 2.2s 或更优
- 距离矩阵（500×500）目标：从 14.6ms 提升

## 7. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| `acquire` 签名变更（breaking） | minor 版本升级，编译期发现遗漏调用点 |
| OOM 后 GPU 状态损坏 | 降级为终态，进程重启恢复 |
| `UniformPool` 增加抽象 | 符合长期目标，消除业务层绕过封装 |
| `DualEncoderSubmitter` 默认启用 | 保留 feature flag 回退 |
| resize 积分图显存占用 | 仅大比例下采样启用，结合 P0 分批保护 |
| hamming 并行归约后端差异 | 保留串行路径 fallback |
| dihedral 位操作正确性 | exhaustive 测试对比 Vec<bool> 结果 |
| Mean/Median GPU 中位数 | 均值用 reduce，中位数用排序网络或保持 CPU 预计算 |

## 8. 对长期/短期目标的对齐

### 长期目标（GPU 类型无关抽象）

- **P4 抽象边界守护**：消除 `&mut self` 泄漏、Uniform 管理不一致，强化能力层封装
- **P0 UncapturedErrorHandler**：增强能力层对后端差异（DX12 OOM）的容错
- **UniformPool 下沉 GpuContext**：业务层不再感知 Uniform 缓冲区管理细节
- 所有优化不破坏 `GpuContext`/`GpuBuffer`/`ComputePipeline` 抽象边界

### 短期目标（图像感知哈希 + 距离比较 GPU 批量并行）

- **P0 显存安全**：解决大规模图像批量处理的崩溃问题，是"批量并行"的前提
- **P1 GPU 同步**：直接降低端到端延迟
- **P2 着色器**：提升核心算法计算效率（resize 10-60×、block_hash 10-30×、hamming 8-32×）
- **P3 内存管理**：减少 GC 压力，提升吞吐量
- **P4 `index_database` + dihedral 批量**：端到端 API 可用性核心
