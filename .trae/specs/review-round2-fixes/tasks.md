# Tasks

## P0: Critical 正确性修复

- [x] Task 1: `packed_dst` u32 溢出防护
  - [x] 1.1-1.3: 全部完成（pack_dst_dimensions 辅助函数 + 3 处调用 + WGSL 注释）

- [x] Task 2: Push Constant 大小验证
  - [x] 2.1-2.3: 全部完成（PUSH_CONSTANT_MAX_SIZE 常量 + 构造时断言 + dispatch 数据长度验证）

- [x] Task 3: `upload_image_to_gpu` 输入验证
  - [x] 3.1-3.2: 全部完成（移除 _ 前缀 + image.len() 验证）

- [x] Task 4: SHA-256 批量提交缓冲区提前 Drop 修复
  - [x] 4.1-4.2: 全部完成（buffers_to_keep_alive 字段 + into_raw() 保持存活）

- [x] Task 5: PipelineCache LRU 淘汰
  - [x] 5.1-5.4: 全部完成（access_order Vec + touch/evict_if_needed + max_entries=64）

- [x] Task 6: SHA-256 `cached_single_block_params` 增长上限
  - [x] 6.1-6.2: 全部完成（MAX_CACHED_PARAMS=16 + cached_params_order LRU）

## P1: 代码重复消除

- [x] Task 7: 统一 Push Constant/Uniform dispatch 分支
  - [x] 7.1-7.7: 全部完成（dispatch_with_params + wgsl_push_constant_to_uniform + dispatch_and_parse_hash + dispatch_resize）

## P1: 资源管理修复

- [x] Task 8: staging buffer 归还池修复
  - [x] 8.1-8.5: 全部完成（BufferPool Mutex 化 + wait_all/clear 错误路径修复 + SHA-256 acquire_staging + download_with_pool）

- [x] Task 9: 零尺寸缓冲区防护
  - [x] 9.1-9.3: 全部完成（download 零尺寸快速返回 + resize 零尺寸验证）

## P2: 性能与健壮性改进

- [x] Task 10: u32 溢出防护
  - [x] 10.1-10.2: 全部完成（27 处 width*height 改为 u64 中间计算）

- [x] Task 11: 移除不必要的输出缓冲区零初始化
  - [x] 11.1-11.2: 全部完成（移除 2 处 queue.write_buffer 零初始化）

- [x] Task 12: GpuBuffer 按用途区分对齐策略
  - [x] 12.1-12.3: 全部完成（align_for_usage: Uniform 16B / Storage 256B + Cow 零拷贝）

- [x] Task 13: BindGroup 缓存 LRU 淘汰
  - [x] 13.1-13.2: 全部完成（access_order + touch/evict_if_needed，满时仅淘汰 1 条）

- [x] Task 14: 错误消息统一中文
  - [x] 14.1-14.3: 全部完成（device/queue 英文错误消息改为中文）

- [x] Task 15: LDS 大小与 Rust 常量同步断言
  - [x] 15.1-15.3: 全部完成（const assert LDS_H_SIZE==144 + WGSL 注释）

# Task Dependencies

- [Task 7] depends on [Task 2]（dispatch_with_params 需要 Push Constant 验证逻辑稳定）
- [Task 8] depends on nothing（可并行）
- [Task 12] depends on nothing（可并行）
- [Task 13] depends on nothing（可并行）
