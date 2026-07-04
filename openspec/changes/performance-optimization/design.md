# 性能优化设计文档

## 高层架构决策

### 决策 1：GPU 显存安全策略（新增 P0，最高优先级）

**背景**：wgpu v24 在 DX12 后端遇到显存不足时，通过 `UncapturedErrorHandler` 报告错误，但项目从未注册 handler，导致进程被直接终止，`log_panics` 无法捕获。

**方案选型**：四层防御 + 运行时降级

#### 层 1：注册 `UncapturedErrorHandler`

在 `context.rs` 的 `init_gpu` 和 `init_software_adapter` 中，Device 创建后立即注册：

```rust
use std::sync::atomic::{AtomicBool, Ordering};

static GPU_OOM_FLAG: AtomicBool = AtomicBool::new(false);
static GPU_DEVICE_LOST_FLAG: AtomicBool = AtomicBool::new(false);

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

#### 层 2：`BufferPool::acquire` 预检查

```rust
pub fn acquire(&self, device: &Device, size: u64, usage: BufferUsage) -> Result<Buffer, GpuError> {
    let max_binding = device.limits().max_storage_buffer_binding_size as u64;
    if size > max_binding {
        return Err(GpuError::Oom { requested: size, limit: max_binding });
    }
    // ... 现有逻辑（返回类型改为 Result）
}
```

**Breaking change**：`acquire` 返回类型从 `Buffer` 改为 `Result<Buffer, GpuError>`，需同步修改约 15 处调用点。

#### 层 3：`compute_phash` 内部分批

```rust
pub fn compute_phash(...) -> Result<Vec<u64>, GpuError> {
    let pixels_per_image = images[0].len();
    let per_image_bytes = (pixels_per_image * 4) as u64;
    let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;

    if per_image_bytes > max_binding {
        return Err(GpuError::InvalidInput(...));
    }

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
- `max_batch_size` 默认 128 MB，可通过 `GpuResizeConfig` 配置

#### 层 4：运行时 OOM → CPU 降级

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

`DefaultBackendDispatcher::dispatch_gpu` 自动生效，无需修改业务层。

**风险**：OOM 后 GPU 状态可能已损坏。缓解：降级后所有后续操作走 CPU，不再访问 GPU。

### 决策 2：GPU 同步优化策略

**方案选型**：合并 submit + 减少 poll 次数 + 连续批量提交

- `wait_all()` 后保留 `device/queue/pool` 引用，仅清空 `encoder` 和 `pending`，与 `flush()` 语义一致
- 解放 `DualEncoderSubmitter`（移除 `#[cfg(feature = "dual-encoder")]`），作为 `phasher_pipeline` 默认提交器
- `phasher_pipeline.rs` 跨 chunk 复用 `GpuBatchSubmitter`，避免每 chunk 新建

```rust
// 优化后：wait_all 保留引用
fn wait_all(&mut self) -> Result<Vec<Vec<u8>>, GpuError> {
    // ... 现有逻辑 ...
    self.encoder = None;      // 仅清空 encoder
    self.pending.clear();     // 仅清空 pending
    // 保留 device/queue/pool
}
```

### 决策 3：着色器算法优化

| 模块 | 当前 | 优化方案 | 预期收益 |
|------|------|---------|---------|
| `resize.wgsl` | O(src_area) 区域平均 | 积分图(SAT) O(1) 查询 | 10-60× |
| `block_hash.wgsl` | O(block_area) 循环 + 重复计算 | 预计算 block 均值图 | 10-30× |
| `pdq_hash.wgsl` | O(N²) DCT + 循环内 cos | 预计算 cos 表 + GPU 端裁剪 16×16 | 30-50% |
| `convolution.wgsl` Full2D | 全局内存读取 | LDS 共享内存 + halo 加载 | 5-10× |
| `hamming.wgsl` nearest | 串行扫描 256 db | 并行归约（256 线程协作） | 8-32× |

### 决策 4：内存管理优化

- **dihedral.rs**：64/256-bit 改直接 u64 位操作（参考 32×32 的 `transpose_32x32` delta-swap），消除 `Vec<bool>` 中间表示
- **gpu_matcher.rs**：返回扁平距离矩阵 `(Vec<u32>, usize)` 替代 `Vec<Vec<u32>>`，提供 `row(i) -> &[u32]` 辅助方法
- **matcher.rs**：`ChainedMatcher::bk_tree_plus_linear` 使用 `Arc<Vec<u64>>` 共享哈希数据
- **matcher_bytes.rs**：`LinearScanMatcherBytes::find_similar` 改为先计算距离再决定是否 clone（先 filter 后 clone）
- **pipeline.rs**：`encode_dispatch_with_params_into` 增加 `queue: &Queue` 参数，复用 `reusable_uniform`，消除 `hash_common.rs:599` 绕过封装

### 决策 5：抽象边界守护（长期目标红线）

- **PipelineCache RefCell 化**：`entries`、`generations`、`next_gen` 改为 `RefCell`，`get_or_create_pipeline` 接受 `&self`，与 `BindGroupCache` 设计一致
- **GpuImageMatcher::index_database**：新增 `GpuImageMatcherIndex` 结构体，持有 `Vec<HashBytes>` 和可选 GPU 缓冲区，`find_similar_with_index` 复用预计算哈希
- **HashMatcherFacade::find_similar_dihedral 批量查询**：8 种二面体变体作为 batch queries 一次性提交 `find_similar_batch(&variants, threshold)`，GPU 路径下 8× 加速

### 决策 6：逐图路径优化

- 按尺寸 `(w, h)` 分桶，每组调用 `compute_gpu_batch_pipeline` 批量处理
- `preprocess_gpu` 改为接受 `&mut GpuBatchSubmitter`，合并所有图像模糊操作
- `merge_gpu_buffers` 的 copy 命令编码到后续 encoder
- 小数据量（像素数 < 阈值）自动选择 CPU 路径

## 数据流

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

## 优化优先级（更新）

1. **P0（立即，显存安全）**：注册 error handler + `compute_phash` 分批 + `acquire` 预检查 + OOM 降级
2. **P1（高，GPU 同步）**：`wait_all` 保留引用 + 解放 `DualEncoderSubmitter` + 跨 chunk 复用
3. **P2（高，着色器）**：resize 积分图 + block_hash 预计算 + PDQ cos 表 + convolution LDS + hamming 并行归约
4. **P3（中，内存管理）**：dihedral 位操作 + 距离矩阵扁平化 + Arc 共享 + Uniform 统一
5. **P4（中，抽象边界）**：PipelineCache RefCell + `index_database` + dihedral 批量查询
6. **P5（低，逐图路径）**：尺寸分组 + 预处理合并 + PipelineCache O(1) LRU + `release_staging` 统一
7. **P6（验证）**：缓存数据驱动的性能验证

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| `acquire` 签名变更（breaking） | minor 版本升级，提供迁移指南 |
| OOM 后 GPU 状态损坏 | 降级后所有操作走 CPU，不再访问 GPU |
| `DualEncoderSubmitter` 默认启用可能暴露 bug | 保留 feature flag 作为回退选项 |
| 积分图增加显存占用（src_w × src_h × 4 字节） | 仅在大比例下采样场景启用，小图保持区域平均 |
| PDQ cos 表增加 binding | 使用 storage buffer，与 kernel 数据合并 |
| `index_database` 引入新 API | 作为可选方法，不破坏现有 `find_similar` |
