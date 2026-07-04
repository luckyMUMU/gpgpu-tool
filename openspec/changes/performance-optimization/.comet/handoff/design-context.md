# Comet Design Handoff

- Change: performance-optimization
- Phase: design
- Mode: compact
- Context hash: 2630118d8aadfaa19f9fbb61a48fcf2026756b53af7854ec8c29c4fe03aa0ca4

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/performance-optimization/proposal.md

- Source: openspec/changes/performance-optimization/proposal.md
- Lines: 1-145
- SHA256: f9455ccdd0467b4117d8dccce5f2a755cac05275c9475e52e322b895cc3e0161

[TRUNCATED]

```md
# 性能优化变更提案

## 问题背景

基于 2026-06-16 对全代码库的重新静态分析，识别出五类核心性能与稳定性瓶颈：

1. **GPU 显存安全（新增高优先级）**：DX12 后端 OOM 时进程被直接终止
   - 项目从未注册 `UncapturedErrorHandler`（`context.rs:202-280`）
   - `GpuError::Oom`（`error.rs:24-25`）从未被触发，形同虚设
   - `compute_phash`（`hash_common.rs:267-322`）等核心路径无分批保护
   - `BufferPool::acquire`（`buffer_pool.rs:157-177`）无显存预算
   - GPU 运行时 OOM 无法触发 CPU 降级（`backend_dispatcher.rs:47-66`）

2. **GPU 同步开销**：非批量路径每 dispatch ~1.6ms，逐图处理累积严重
   - `phasher_pipeline.rs:110, 217` 每个 chunk 新建 `GpuBatchSubmitter`
   - `batch.rs:268-270, 324-326, 341-343` `wait_all()` 清空引用，无法跨 chunk 复用
   - `DualEncoderSubmitter` 被 feature gate 限制，默认未启用

3. **着色器/算法效率**：
   - `resize.wgsl:55-62` 区域平均 O(src_area)，大比例下采样 10-60× 冗余
   - `block_hash.wgsl:14-25` `block_mean` O(block_area) 循环 + 重复计算
   - `pdq_hash.wgsl:34-58` O(N²) DCT + 循环内 `cos()` + 下载完整 64×64 系数
   - `convolution.wgsl:120-130` Full2D 模式无 LDS 共享内存
   - `hamming.wgsl:183-200` `find_nearest_neighbor` 串行扫描 256 db

4. **内存/数据管理**：
   - `dihedral.rs:19-109, 114-210` 64/256-bit 使用 `Vec<bool>` 中间表示（9 次堆分配/变换）
   - `gpu_matcher.rs:144-196` 距离矩阵返回 `Vec<Vec<u32>>`（N 次堆分配）
   - `matcher.rs:152-158` `bk_tree_plus_linear` 数据复制未共享
   - `pipeline.rs:576-609` `encode_dispatch_with_params_into` 每次创建新 Uniform 缓冲区
   - `hash_common.rs:599` 业务层绕过封装直接创建 Uniform 缓冲区

5. **抽象泄漏（长期目标红线）**：
   - `context.rs:428-438` `get_or_create_pipeline` 需 `&mut self`，与 `BindGroupCache` 的 `RefCell` 设计不一致
   - `pipeline.rs:576-609` encode 路径缺 `queue` 参数，迫使业务层绕过封装
   - `phasher_pipeline.rs` 业务层被迫感知"分块边界即同步点"

## 目标

### P0 显存安全（新增，最高优先级）
- 注册 `UncapturedErrorHandler`，DX12 OOM 时记录日志而非 abort
- `compute_phash` 内部分批，按 `max_storage_buffer_binding_size` 自动切片
- `BufferPool::acquire` 预检查并触发 `GpuError::Oom`
- 运行时 OOM 触发 CPU 降级（通过原子标志 + `backend()` 动态返回）

### P1 GPU 同步开销
- 非批量路径 dispatch 开销降低 50%+
- `wait_all()` 保留 device/queue/pool 引用，支持连续批量提交
- 解放 `DualEncoderSubmitter`（移除 feature gate），作为 `phasher_pipeline` 默认提交器

### P2 着色器算法效率
- `resize.wgsl` 改用积分图(SAT)，O(1) 查询替代 O(src_area)
- `block_hash.wgsl` 预计算 block 均值图，消除 O(block_area) 循环
- PDQ DCT 预计算 cos 表 + GPU 端裁剪 16×16 系数
- `convolution.wgsl` Full2D 模式增加 LDS 共享内存
- `hamming.wgsl` `find_nearest_neighbor` 改并行归约

### P3 内存/数据管理
- `dihedral.rs` 64/256-bit 改直接位操作，零堆分配
- `gpu_matcher.rs` 距离矩阵扁平化为 `(Vec<u32>, usize)`
- `matcher.rs` 使用 `Arc<Vec<u64>>` 共享哈希数据
- 统一 Uniform 缓冲区管理，消除业务层绕过封装

### P4 抽象边界守护（长期目标）
- `PipelineCache` 改用 `RefCell`，`get_or_create_pipeline` 接受 `&self`
- `encode_dispatch_with_params_into` 增加 `queue` 参数或下沉到 `GpuContext`
- `GpuImageMatcher` 新增 `index_database` 预计算 database 哈希
- `HashMatcherFacade::find_similar_dihedral` 批量查询优化（8 变体合并为单次 GPU dispatch）

### P5 逐图路径与细节优化
- `phasher.rs` 逐图路径按尺寸分组批量处理
- `phasher.rs` `preprocess_gpu` 合并所有图像模糊到 `GpuBatchSubmitter`
- `context.rs` PipelineCache LRU 改 O(1) arena + 双向链表
- `buffer_pool.rs` 统一 `release_staging` 与 `release` 的大缓冲区策略

### P6 缓存数据驱动的性能验证
- 基于 Czkawka 缓存数据（20000 条 1024-bit 哈希）量化优化效果
- 对比优化前后 GPU/CPU 性能数据

## 范围
```

Full source: openspec/changes/performance-optimization/proposal.md

## openspec/changes/performance-optimization/design.md

- Source: openspec/changes/performance-optimization/design.md
- Lines: 1-184
- SHA256: 79d94e23d692655ca73a38f9a17860d5d64ee8fb2ee42de9ed3b12c39a42dbd5

[TRUNCATED]

```md
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

```

Full source: openspec/changes/performance-optimization/design.md

## openspec/changes/performance-optimization/tasks.md

- Source: openspec/changes/performance-optimization/tasks.md
- Lines: 1-106
- SHA256: 491243e8f7a343393b8574ea275dd9f73cb5fbe39be7996b9108bdf8d348851f

[TRUNCATED]

```md
## P0: GPU 显存安全（新增，最高优先级）

- [ ] 0.1 在 `src/context.rs` `init_gpu` 和 `init_software_adapter` 中注册 `UncapturedErrorHandler`，设置 `GPU_OOM_FLAG` 和 `GPU_DEVICE_LOST_FLAG` 原子标志
- [ ] 0.2 在 `src/buffer_pool.rs` `acquire` 方法增加 `max_storage_buffer_binding_size` 预检查，超限返回 `GpuError::Oom`（返回类型改为 `Result<Buffer, GpuError>`）
- [ ] 0.3 同步修改 `acquire` 的约 15 处调用点，处理 `Result` 返回值
- [ ] 0.4 在 `src/tasks/hash_common.rs` `compute_phash` 内部增加分批逻辑：按 `DEFAULT_MAX_BATCH_SIZE / per_image_bytes` 计算单批最大图像数，循环处理 chunks
- [ ] 0.5 在 `src/tasks/hash_common.rs` `compute_phash_from_gpu_buffer` 增加分批逻辑（同 0.4）
- [ ] 0.6 在 `src/tasks/phasher_pipeline.rs` `compute_gpu_per_image_pipeline` 循环内增加单图尺寸检查，超限返回 `GpuError::InvalidInput`
- [ ] 0.7 在 `src/tasks/gpu_image_matcher.rs` `compute_hashes` 入口增加显存预算分批加固
- [ ] 0.8 在 `src/context.rs` `backend()` 方法增加 OOM 标志检查，动态返回 `ComputeBackend::Cpu`
- [ ] 0.9 在 `src/backend_dispatcher.rs` 验证 `dispatch_gpu` 能正确响应 `backend()` 动态切换
- [ ] 0.10 运行 `cargo test` 验证编译和测试通过
- [ ] 0.11 运行 `cargo test --features pdq` 验证 PDQ 路径
- [ ] 0.12 手动测试 DX12 后端大图像批量场景，验证 OOM 降级而非崩溃

## P1: GPU 同步开销优化

- [ ] 1.1 优化 `src/batch.rs` `GpuBatchSubmitter::wait_all()`：保留 `device/queue/pool` 引用，仅清空 `encoder` 和 `pending`
- [ ] 1.2 优化 `src/batch.rs` `DualEncoderSubmitter::wait_all()`：同 1.1，保留引用
- [ ] 1.3 解放 `src/batch.rs` `DualEncoderSubmitter`：移除 `#[cfg(feature = "dual-encoder")]`，作为默认实现
- [ ] 1.4 优化 `src/tasks/phasher_pipeline.rs`：跨 chunk 复用 `GpuBatchSubmitter`，避免每 chunk 新建
- [ ] 1.5 优化 `src/pipeline.rs` 非 Push Constant 路径：缓存可复用 Uniform 缓冲区，避免每次 `dispatch_with_params` 创建新缓冲区
- [ ] 1.6 优化 `src/pipeline.rs` `encode_dispatch_with_params_into`：增加 `queue: &Queue` 参数，复用 `reusable_uniform`
- [ ] 1.7 修复 `src/tasks/hash_common.rs:599` 业务层绕过封装：统一调用 `pipeline.encode_dispatch_with_params_into`
- [ ] 1.8 标记 `src/buffer.rs` `download()` 和 `download_batch()` 为 `#[deprecated]`，引导使用 `_with_pool` 版本
- [ ] 1.9 在 `src/tasks/gpu_matcher.rs` 等核心模块推广 `download_batch_with_pool`，合并多个独立 download 为单次 poll
- [ ] 1.10 运行 `cargo test` 验证编译和测试通过

## P2: 着色器算法效率优化

- [ ] 2.1 重写 `src/tasks/resize.wgsl` 缩放着色器：使用积分图(SAT)方法替代区域平均，将 O(src_area) 降为 O(1)（仅大比例下采样场景启用）
- [ ] 2.2 优化 `src/tasks/gpu_resize.rs`：增加积分图构建 pass + 查询 pass 的两阶段调度
- [ ] 2.3 重写 `src/tasks/block_hash.wgsl` `block_mean`：增加预处理 pass 预计算所有 block 均值图，消除 O(block_area) 循环
- [ ] 2.4 优化 `src/tasks/block_hash.wgsl`：提取 `m_current = block_mean(bx, by)` 一次，复用于 left/top 分支
- [ ] 2.5 重写 `src/tasks/pdq_hash.wgsl` DCT 着色器：预计算 64×64 余弦查找表作为 storage buffer 传入
- [ ] 2.6 优化 `src/tasks/pdq_hash.rs` GPU 路径：增加第 3 个 pass 提取 16×16 低频系数，减少 93.75% 下载量
- [ ] 2.7 优化 `src/tasks/pdq_hash.wgsl` workgroup 布局：从 1D (256,1,1) 改为 2D (8,8,1)，利用 LDS 共享 cos 表
- [ ] 2.8 为 `src/tasks/convolution.wgsl` Full2D 模式实现 LDS 共享内存优化：协作加载 (WG_X+2r)×(WG_Y+2r) 像素到 LDS
- [ ] 2.9 优化 `src/tasks/convolution.wgsl` separable_fused 水平 pass：将输入区域加载到 LDS，减少全局内存重复读取
- [ ] 2.10 重写 `src/tasks/hamming.wgsl` `find_nearest_neighbor`：改为并行归约（256 线程协作扫描 256 db，warp shuffle 归约最小值）
- [ ] 2.11 优化 `src/tasks/hamming.wgsl` `hamming_distance_matrix` 加载阶段：所有 256 线程协作加载 query_tile 和 db_tile_matrix
- [ ] 2.12 优化 `src/tasks/pdq_hash.rs` CPU DCT：使用快速 DCT 算法替代朴素实现
- [ ] 2.13 运行 `cargo test` 验证编译和测试通过
- [ ] 2.14 运行 `cargo test --features pdq` 验证 PDQ 路径

## P3: 内存管理优化

- [ ] 3.1 重写 `src/tasks/dihedral.rs` 64-bit 变换：直接在 u64 位上实现 8 种二面体变换（参考 32×32 的 `transpose_32x32` delta-swap），消除所有 `Vec<bool>` 分配
- [ ] 3.2 重写 `src/tasks/dihedral.rs` 256-bit 变换：直接在 `[u64; 4]` 上实现位操作，消除 `Vec<bool>` 分配
- [ ] 3.3 统一 `src/tasks/dihedral.rs` 四种尺寸实现风格：抽象出 `BitMatrix<N>` 泛型结构或统一位操作模式
- [ ] 3.4 优化 `src/tasks/dihedral.rs` `all_variants` 返回类型：改为 `[u64; 8]` 或迭代器，避免 `.to_vec()` 堆分配
- [ ] 3.5 优化 `src/tasks/gpu_matcher.rs` `compute_distance_matrix`：返回扁平 `(Vec<u32>, usize)` 替代 `Vec<Vec<u32>>`，提供 `row(i) -> &[u32]` 辅助方法
- [ ] 3.6 优化 `src/tasks/gpu_matcher.rs` 分块路径：扁平化后直接写入预分配 Vec 的对应区间，消除 chunk 内 Vec 分配
- [ ] 3.7 优化 `src/tasks/matcher.rs` `ChainedMatcher::bk_tree_plus_linear`：使用 `Arc<Vec<u64>>` 共享哈希数据
- [ ] 3.8 优化 `src/tasks/matcher_bytes.rs` `ChainedMatcherBytes::bk_tree_plus_linear`：使用 `Arc<Vec<HashBytes>>` 共享
- [ ] 3.9 优化 `src/tasks/matcher_bytes.rs` `LinearScanMatcherBytes::find_similar`：改为先计算距离再决定是否 clone（先 filter 后 clone）
- [ ] 3.10 优化 `src/tasks/matcher.rs` `HashMatcherFacade::find_similar_dihedral`：8 种变体作为 batch queries 一次性提交 `find_similar_batch(&variants, threshold)`
- [ ] 3.11 优化 `src/tasks/gpu_image_matcher.rs` 过滤阶段：`MatchResultBytes` 改为持有索引 `(usize, u32)` 或 `Arc<HashBytes>`，避免每匹配 clone
- [ ] 3.12 优化 `src/tasks/gpu_matcher.rs` `hash_bytes_to_u32`：缓存最近一次转换结果，避免重复查询场景的重复转换
- [ ] 3.13 运行 `cargo test` 验证编译和测试通过

## P4: 抽象边界守护（长期目标红线）

- [ ] 4.1 优化 `src/context.rs` `PipelineCache`：`entries`、`generations`、`next_gen` 改为 `RefCell`，`get_or_create` 接受 `&self`
- [ ] 4.2 优化 `src/context.rs` `get_or_create_pipeline`：接受 `&self`，与 `BindGroupCache` 设计一致
- [ ] 4.3 优化 `src/context.rs` `PipelineCache` LRU：改为 O(1) arena + 双向链表，与 `BindGroupCache` 一致
- [ ] 4.4 新增 `src/tasks/gpu_image_matcher.rs` `GpuImageMatcherIndex` 结构体：持有 `Vec<HashBytes>` 和可选 GPU 缓冲区
- [ ] 4.5 新增 `src/tasks/gpu_image_matcher.rs` `index_database` 方法：预计算 database 哈希并缓存
- [ ] 4.6 新增 `src/tasks/gpu_image_matcher.rs` `find_similar_with_index` 方法：复用预计算哈希
- [ ] 4.7 优化 `src/tasks/gpu_image_matcher.rs`：合并 query + database 哈希计算为一次 GPU submit
- [ ] 4.8 优化 `src/tasks/gpu_matcher.rs` `find_nearest_neighbors`：增加分块逻辑，参考 `compute_distance_matrix` 的分块模式
- [ ] 4.9 运行 `cargo test` 验证编译和测试通过

## P5: 逐图路径与细节优化

- [ ] 5.1 优化 `src/tasks/phasher.rs` `compute_gpu_per_image_pipeline`：按尺寸 `(w, h)` 分桶，每组调用 `compute_gpu_batch_pipeline` 批量处理
- [ ] 5.2 优化 `src/tasks/phasher.rs` `preprocess_gpu`：改为接受 `&mut GpuBatchSubmitter`，合并所有图像模糊操作
- [ ] 5.3 优化 `src/tasks/phasher.rs` `process_chunks_serial` Mean/Median 分支：统一使用 `GpuBatchSubmitter` + `merge_gpu_buffers_batch`
- [ ] 5.4 优化 `src/tasks/phasher_util.rs` `merge_gpu_buffers`：将 copy 命令编码到后续 encoder（标记非 batch 版本为 `#[deprecated]`）
- [ ] 5.5 优化 `src/tasks/phasher.rs` `compute_hash_gpu` Mean/Median 路径：编写独立 GPU 阈值计算着色器（reduce kernel），消除 GPU↔CPU 往返
```

Full source: openspec/changes/performance-optimization/tasks.md

