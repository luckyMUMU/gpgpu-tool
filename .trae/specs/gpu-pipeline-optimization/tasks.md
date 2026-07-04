# Tasks

## P0: 正确性修复（必须优先完成）

- [x] Task 1: 修复 unpack_u32_to_u8 越界 panic（审查 C1）
  - [x] 1.1: 修改 `unpack_u32_to_u8`，使用 `count.min(data.len())` 替代 `data[..count]`
  - [x] 1.2: 添加 `debug_assert!(count <= data.len())` 在开发期捕获异常调用
  - [x] 1.3: 更新相关测试验证边界保护
  - [x] 1.4: `cargo test` 验证

- [x] Task 2: 修复可分离卷积中间结果精度丢失（审查 C2）
  - [x] 2.1: 修改 convolution.wgsl 水平 1D 输出，使用 `bitcast<u32>(sum)` 存储 f32 中间结果
  - [x] 2.2: 修改 convolution.wgsl 垂直 1D 输入，使用 `bitcast<f32>(pixels[idx])` 还原 f32 中间值
  - [x] 2.3: 确保最终输出仍使用 `u32(clamp(sum, 0.0, 255.0))` 量化
  - [x] 2.4: 添加可分离 vs Full2D 精度对比测试
  - [x] 2.5: `cargo test` 验证

- [x] Task 3: 修复 GpuBatchSubmitter 资源泄漏（审查 C3）
  - [x] 3.1: 为 GpuBatchSubmitter 实现 Drop trait，drop 时清理 pending jobs 和 staging buffers
  - [x] 3.2: 将 BatchJob 的 `input/output/params` 硬编码 3-buffer 改为 `buffers: Vec<GpuBuffer>` 动态绑定
  - [x] 3.3: 修改 submit() 中 create_bind_group 使用动态绑定列表
  - [x] 3.4: 更新所有 BatchJob 使用方适配新 API
  - [x] 3.5: `cargo test` + `cargo clippy` 验证

## P1: 线程模型与 API 安全性

- [x] Task 4: 重构 resize.wgsl 线程模型（审查 C4）
  - [x] 4.1: 修改 resize.wgsl 为每目标像素一个线程模型
  - [x] 4.2: 着色器从 global_invocation_id.x 解码 img_idx、dy、dx
  - [x] 4.3: 更新 Rust 端 dispatch 计算逻辑
  - [x] 4.4: 添加批量 2 张图像缩放测试，验证线程数 > 2
  - [x] 4.5: `cargo test` 验证

- [x] Task 5: device()/queue() 安全化（审查 C5）
  - [x] 5.1: 修改 device()/queue() 在 CPU 降级模式下返回 Result 而非 panic
  - [x] 5.2: 标记 device()/queue() 为 `#[deprecated]`，推荐 try_device()/try_queue()
  - [x] 5.3: 更新所有内部调用方使用 try_ 变体
  - [x] 5.4: `cargo test` + `cargo clippy` 验证

## P1: 基础设施统一

- [x] Task 6: 统一 BufferPool 到 GpuContext
  - [x] 6.1-6.10: 全部完成

- [x] Task 7: 统一 to_wgpu_usage（审查 I3）
  - [x] 7.1-7.4: 全部完成（BufferUsage::to_wgpu_usage(include_copy_src) 统一方法）

- [x] Task 8: GpuBuffer COPY_SRC 按需添加（审查 I10）
  - [x] 8.1-8.4: 全部完成（from_data_readable/from_bytes_readable/empty_readable 变体）

- [x] Task 9: fxhash 安全网（审查 I8）
  - [x] 9.1-9.3: 全部完成（缓存命中时 WGSL 源码 == 比较）

- [x] Task 10: BindGroup 缓存（审查 I9）
  - [x] 10.1-10.4: 全部完成（ComputePipeline 内部 BindGroupCache，上限 64 条）

## P1: 架构重构

- [x] Task 11: BackendDispatcher 统一降级策略
  - [x] 11.1-11.7: 全部完成（BackendDispatcher trait + DefaultBackendDispatcher + PHasherCpu 缓存）

- [x] Task 12: 零拷贝管线完整串联
  - [x] 12.1-12.7: 全部完成（blur → resize → hash 零拷贝 GPU 流水线 + 7 个集成测试）

- [x] Task 13: PerceptualHasher 职责分离
  - [x] 13.1-13.6: 全部完成（preprocess/resize/compute_hash 三阶段独立公开方法）

- [x] Task 14: 声明式管线 API
  - [x] 14.1-14.7: 全部完成（GpuPipelineBuilder + 14 个集成测试）

## P2: WGSL 着色器硬件优化

- [x] Task 15: WGSL 着色器 Workgroup Size 优化（基于 RX5700 报告）
  - [x] 15.1-15.6: 全部完成（2D 图像着色器 @workgroup_size(8,8,1)，SHA-256/PDQ 保持 256）

- [x] Task 16: 卷积着色器 LDS 共享内存优化（基于 RX5700 报告）
  - [x] 16.1-16.6: 全部完成（separable_fused 入口点 + LDS halo 预加载 + use_lds 配置）

- [x] Task 17: Push Constant 支持（基于 RX5700 报告）
  - [x] 17.1-17.6: 全部完成（基础设施 + 哈希/缩放着色器 PhashParams/ResizeParams 迁移到 push constant）

- [x] Task 18: Storage Buffer 256 字节对齐 + Dispatch 粒度优化
  - [x] 18.1-18.5: 全部完成（SIZE_CLASSES 256 对齐 + GpuBuffer 填充 + CU 估算 + 低 occupancy 警告）

- [x] Task 19: SHA-256 链式着色器前缀和优化（审查 I7）
  - [x] 19.1-19.3: 全部完成（prefix_sums buffer + O(1) 偏移查找）

## P2: 其他审查修复

- [x] Task 20: ChainedMatcher 语义修正（审查 I12）
  - [x] 20.1-20.3: 全部完成（Union → FirstHit 策略）

- [x] Task 21: GpuResize API 一致性修复（审查 I11）
  - [x] 21.1-21.2: 全部完成（空输入返回 Ok + 提取 validate_batch_input）

## P2: 性能基准

- [x] Task 22: 性能基准建立
  - [x] 22.1-22.10: 全部完成（7 个基准组 + criterion 配置）

# Task Dependencies

- [Task 4] depends on [Task 1, Task 2] (线程模型重构需先确保基础正确性)
- [Task 6] depends on [Task 3] (BatchJob 动态绑定是共享池的前置)
- [Task 7] depends on [Task 8] (to_wgpu_usage 统一需先确定 COPY_SRC 策略)
- [Task 10] depends on [Task 6] (BindGroup 缓存需共享池 API 稳定)
- [Task 11] depends on [Task 5, Task 6] (BackendDispatcher 需 device/queue 安全化和共享池)
- [Task 12] depends on [Task 2, Task 4, Task 6] (零拷贝管线需精度修复、线程模型、共享池)
- [Task 13] depends on [Task 11, Task 12] (职责分离需降级策略和零拷贝管线)
- [Task 14] depends on [Task 13] (声明式 API 需各阶段可独立调用)
- [Task 15] depends on [Task 4, Task 6] (workgroup size 需线程模型和共享池)
- [Task 16] depends on [Task 2, Task 15] (LDS 需精度修复和 2D workgroup)
- [Task 17] depends on [Task 6] (Push Constant 需管线基础设施稳定)
- [Task 18] depends on [Task 6, Task 7] (对齐优化需共享池和统一 to_wgpu_usage)
- [Task 19] depends on [Task 6] (前缀和优化需共享池)
- [Task 22] depends on [Task 12, Task 14, Task 15, Task 16, Task 17] (基准需优化完成)
