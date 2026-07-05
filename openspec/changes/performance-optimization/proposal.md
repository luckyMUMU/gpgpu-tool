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

### 涉及模块

**显存安全（P0）**：
- `src/context.rs` — 注册 `UncapturedErrorHandler`，OOM 原子标志
- `src/buffer_pool.rs` — `acquire` 预检查 + `GpuError::Oom`
- `src/error.rs` — 完善 `Oom` 错误传播
- `src/backend_dispatcher.rs` — 运行时 OOM → CPU 降级
- `src/tasks/hash_common.rs` — `compute_phash` 内部分批
- `src/tasks/gpu_image_matcher.rs` — `compute_hashes` 入口分批加固
- `src/tasks/phasher_pipeline.rs` — 逐图路径单图尺寸检查

**GPU 同步（P1）**：
- `src/batch.rs` — `wait_all()` 保留引用，解放 `DualEncoderSubmitter`
- `src/tasks/phasher_pipeline.rs` — 跨 chunk 复用 submitter

**着色器（P2）**：
- `src/tasks/resize.wgsl` + `gpu_resize.rs` — 积分图
- `src/tasks/block_hash.wgsl` — block 均值预计算
- `src/tasks/pdq_hash.wgsl` + `pdq_hash.rs` — cos 表 + 系数裁剪
- `src/tasks/convolution.wgsl` — Full2D LDS
- `src/tasks/hamming.wgsl` — 并行归约最近邻

**内存管理（P3）**：
- `src/tasks/dihedral.rs` — 64/256-bit 位操作
- `src/tasks/gpu_matcher.rs` — 扁平化距离矩阵
- `src/tasks/matcher.rs` + `matcher_bytes.rs` — `Arc` 共享 + 先 filter 后 clone
- `src/pipeline.rs` — Uniform 缓冲区复用统一

**抽象边界（P4）**：
- `src/context.rs` — `PipelineCache` RefCell 化
- `src/pipeline.rs` — `encode_dispatch_with_params_into` 签名优化
- `src/tasks/gpu_image_matcher.rs` — `index_database` 预计算
- `src/tasks/matcher.rs` — dihedral 批量查询

**逐图路径（P5）**：
- `src/tasks/phasher.rs` — 尺寸分组 + 预处理合并
- `src/context.rs` — PipelineCache O(1) LRU
- `src/buffer_pool.rs` — `release_staging` 策略统一

**验证（P6）**：
- `tests/cache_perf_test.rs` — 缓存数据性能测试
- `benches/cache_matcher_bench.rs` — 基准测试

### 非目标
- 不涉及功能新增（除 `index_database` 等 API 扩展）
- 不涉及公共 API 签名重构（仅内部优化，`BufferPool::acquire` 返回 `Result` 除外）
- 不引入新的外部依赖（除可能的 `rustdct` 用于 CPU DCT）
- 不修改公共接口语义（`acquire` 签名变更为 breaking change，需 minor 版本升级）

## 决策原则对齐

本提案严格遵循 ADR-006 双问题检验：

1. **是否服务短期目标（图像感知哈希 + 距离比较 GPU 批量并行）？**
   - P0 显存安全：解决大规模图像批量处理的崩溃问题，是"批量并行"的前提
   - P1 GPU 同步：直接降低端到端延迟
   - P2 着色器：提升核心算法计算效率
   - P3 内存管理：减少 GC 压力，提升吞吐量
   - P4 `index_database` + dihedral 批量：端到端 API 可用性核心

2. **是否保持或推进长期目标（GPU 类型无关的抽象）？**
   - P4 抽象边界守护：消除 `&mut self` 泄漏、Uniform 管理不一致等抽象缺陷
   - P0 `UncapturedErrorHandler`：增强能力层对后端差异的容错
   - 所有优化不破坏 `GpuContext`/`GpuBuffer`/`ComputePipeline` 抽象边界
