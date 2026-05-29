# 第二轮审查修复规范

## Why

第一轮 22 个任务完成后，对代码库进行了全面五轴审查（正确性、可读性、架构、安全、性能），发现 9 个 Critical、22 个 Important、18 个 Suggestion 级别问题。最紧迫的问题是 `packed_dst` u32 溢出（大图缩放静默错误）、Push Constant 大小未验证（GPU 驱动崩溃风险）、SHA-256 批量提交缓冲区提前 Drop（计算结果可能错误）、以及多处缓存无限增长（内存泄露）。此外，Push Constant/Uniform 回退分支在 5 处重复，`compute_phash` 系列函数 ~80% 逻辑重复，需要架构层面统一。

## What Changes

- 修复 `packed_dst` u32 溢出：添加输入验证或改用独立 u32 字段
- 修复 Push Constant 大小验证：添加 `≤128` 且 `4 字节对齐` 的断言
- 修复 `upload_image_to_gpu` 输入验证：验证 `image.len() == w*h`
- 修复 SHA-256 批量提交缓冲区提前 Drop：加入 `input_buffers_to_release`
- 修复 PipelineCache / SHA-256 缓存无限增长：添加上限 + LRU 淘汰
- 统一 Push Constant/Uniform 分支：在 `ComputePipeline` 层封装 `dispatch_with_params`
- 消除 `compute_phash` / `compute_phash_from_gpu_buffer` 重复：抽取公共函数
- 消除 `resize_batch_inner` / `resize_batch_gpu_inner` 重复：抽取 `dispatch_resize`
- 消除 `wgsl_push_constant_to_uniform` 重复：抽取通用函数
- 修复 staging buffer 泄漏：错误路径和 clear() 归还池
- 修复零尺寸缓冲区验证：download/resize 入口检查
- 修复 `width * height` u32 溢出：改用 u64 中间计算
- 移除不必要的输出缓冲区零初始化
- 统一错误消息语言为中文
- GpuBuffer 按用途区分对齐策略（Uniform 16 字节 vs Storage 256 字节）

## Impact

- Affected specs: gpu-pipeline-optimization（后续修复）
- Affected code:
  - `src/tasks/gpu_resize.rs` — packed_dst 溢出 + resize 内部方法重复
  - `src/tasks/hash_common.rs` — compute_phash 重复 + push constant 分支
  - `src/tasks/sha256.rs` — 缓冲区提前 Drop + 缓存无限增长
  - `src/tasks/phasher.rs` — upload_image_to_gpu 输入验证
  - `src/pipeline.rs` — Push Constant 验证 + dispatch_with_params + BindGroup LRU
  - `src/context.rs` — PipelineCache LRU
  - `src/buffer.rs` — 零尺寸检查 + 按用途对齐
  - `src/batch.rs` — staging buffer 归还
  - `src/buffer_pool.rs` — staging 分档复用

## ADDED Requirements

### Requirement: 输入验证防护

系统 SHALL 在所有接受外部输入的公开 API 入口处验证参数合法性：

#### Scenario: packed_dst 溢出防护
- **WHEN** `target_width > 65535` 或 `target_height > 65535`
- **THEN** 返回 `GpuError::InvalidInput`，包含描述性错误消息

#### Scenario: Push Constant 大小验证
- **WHEN** `push_constant_size > 128` 或 `push_constant_size % 4 != 0`
- **THEN** panic 并附带清晰错误消息（此为编程错误，不应静默）

#### Scenario: 图像尺寸一致性验证
- **WHEN** `image.len() != (width * height) as usize`
- **THEN** 返回 `GpuError::InvalidInput`

#### Scenario: 零尺寸缓冲区防护
- **WHEN** `target_width == 0` 或 `target_height == 0` 或 `self.size == 0`
- **THEN** 返回 `GpuError::InvalidInput`（resize）或 `Ok(vec![])`（download）

### Requirement: 缓存增长上限

系统 SHALL 对所有内部缓存设置最大条目数限制：

#### Scenario: PipelineCache 上限
- **WHEN** PipelineCache 条目数超过 64
- **THEN** 淘汰最久未使用的条目（LRU）

#### Scenario: SHA-256 params 缓存上限
- **WHEN** `cached_single_block_params` 条目数超过 16
- **THEN** 淘汰最久未使用的条目

### Requirement: 资源生命周期安全

系统 SHALL 确保 GPU 缓冲区在 GPU 命令提交完成前不被回收：

#### Scenario: SHA-256 批量提交缓冲区保持
- **WHEN** `submit_multi_block_message` 创建 `params_buf`/`prefix_sum_buf`
- **THEN** 这些缓冲区被加入 `input_buffers_to_release`，在 `wait_all` 后释放

#### Scenario: staging buffer 归还
- **WHEN** `GpuBatchSubmitter::wait_all` 遇到映射错误
- **THEN** 所有 pending staging buffer 被 unmap 并归还 BufferPool
- **WHEN** `GpuBatchSubmitter::clear()` 被调用
- **THEN** 所有 pending staging buffer 被归还 BufferPool

### Requirement: 代码重复消除

系统 SHALL 将 Push Constant/Uniform 回退分支统一到 `ComputePipeline::dispatch_with_params` 方法：

#### Scenario: dispatch_with_params 调用
- **WHEN** 调用方需要 dispatch 并传递参数
- **THEN** 调用 `pipeline.dispatch_with_params(ctx, &params_bytes, &buffers, compute_units)` 自动选择 Push Constant 或 Uniform 路径

### Requirement: u32 溢出防护

系统 SHALL 在所有 `width * height` 计算中使用 u64 中间类型：

#### Scenario: 大图像像素数计算
- **WHEN** 计算 `width * height` 像素总数
- **THEN** 使用 `(width as u64 * height as u64) as usize` 避免溢出

## MODIFIED Requirements

### Requirement: GpuBuffer 对齐策略

GpuBuffer 创建时 SHALL 根据用途使用不同对齐策略：
- `BufferUsage::Uniform`：16 字节对齐（满足 std140 要求）
- `BufferUsage::Storage` / 其他：256 字节对齐

### Requirement: 输出缓冲区初始化

感知哈希输出缓冲区 SHALL NOT 使用 `queue.write_buffer` 零初始化。wgpu 保证新创建的 Storage buffer 内容为零。

## REMOVED Requirements

### Requirement: wgsl_push_constant_to_uniform 独立实现
**Reason**: 合并为通用函数 `wgsl_push_constant_to_uniform(wgsl: &str, struct_name: &str) -> String`
**Migration**: hash_common.rs 和 gpu_resize.rs 统一调用通用函数
