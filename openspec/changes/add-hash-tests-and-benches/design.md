## Context

当前已有 6 种感知哈希算法的 GPU 实现（`src/tasks/*_hash.rs` + `*_hash.wgsl`），但测试覆盖仅限于 SHA-256。现有测试结构：
- `tests/sha256_test.rs` — 功能测试
- `benches/sha256_bench.rs` — 性能基准测试

本次需要为 6 种新哈希算法建立同样的测试体系。

**重要发现**：代码审查中发现以下问题：

1. **WGSL bit 映射缺陷**：Gradient Hash 和 Double Gradient Hash 使用像素索引作为 bit 位置，导致信息丢失
   - Gradient Hash：8x9 图像的最后一行（row=8）的 7 个梯度比较被丢弃
   - Double Gradient Hash：72 个水平/垂直梯度比较各只保留 32 个，信息丢失 56%

2. **buffer.rs 错误传播缺陷**：`download` 方法中 `map_async` 的错误仅记录日志但未返回给调用者（[buffer.rs#L90-L96](file:///d:/Code/AI/wgpu-tool/src/buffer.rs#L90-L96)）
   - `staging.slice(..).map_async()` 的回调中 `Err(e)` 仅调用 `log::error!`
   - 映射失败时 `get_mapped_range()` 可能 panic，错误未通过 `GpuError` 传播

3. **sha256.rs 死代码**：`compute_multi_block_batch` 中遗留了已分配但未使用的 `input_data`（[sha256.rs#L218-L251](file:///d:/Code/AI/wgpu-tool/src/tasks/sha256.rs#L218-L251)）
   - 原并行策略放弃后，`msg_blocks` 等变量仍被构造但无实际用途
   - 造成不必要的内存分配和计算

这些问题将在测试实施前一并修复。

## Goals / Non-Goals

**Goals:**
- 修复 Gradient Hash 和 Double Gradient Hash 的 WGSL bit 映射缺陷
- 修复 buffer.rs 中 `map_async` 错误未正确传播的问题
- 清理 sha256.rs 中 `compute_multi_block_batch` 的死代码
- 每种算法至少 3 个功能测试（单图像、批量图像、空输入）
- 每种算法至少 1 个性能基准测试（GPU vs CPU）
- 提供 CPU 参考实现用于结果比对
- 测试数据可复现（固定随机种子或固定像素值）

**Non-Goals:**
- 不测试图像预处理（缩放、灰度化）
- 不引入真实图像文件（使用合成数据）
- 不测试汉明距离计算

## Decisions

### 1. WGSL bit 映射策略：统一使用独立计数器
- **问题**：原实现使用像素索引/块索引作为 bit 位置，当图像尺寸与哈希位数不匹配时导致信息丢失
- **修复方案**：所有算法统一使用独立的 `bit_pos` 计数器（0-63），按处理顺序连续编号
- **已修复文件**：
  - `gradient_hash.wgsl`：引入 `bit_pos` 替代 `idx`
  - `double_gradient_hash.wgsl`：引入 `h_bit_pos` / `v_bit_pos` 替代 `bit_idx`

### 2. buffer.rs 错误传播：使用 channel 同步获取 map_async 结果
- **问题**：`map_async` 是异步 API，错误在回调中仅记录日志，调用者无法感知
- **修复方案**：使用 `std::sync::mpsc::channel` 同步等待回调结果
  ```rust
  let (sender, receiver) = std::sync::mpsc::channel();
  staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
      sender.send(result).ok();
  });
  device.poll(wgpu::Maintain::Wait);
  match receiver.recv().unwrap() {
      Ok(()) => { /* 继续读取数据 */ },
      Err(e) => Err(GpuError::MapFailed(e.to_string())),
  }
  ```
- **新增错误类型**：`GpuError::MapFailed(String)`

### 3. sha256.rs 死代码清理：删除未使用的变量分配
- **问题**：`compute_multi_block_batch` 中 `msg_blocks` 等变量被构造但后续未使用
- **修复方案**：删除 `MsgBlocks` 结构体及相关 `msg_blocks` 构造逻辑
- **影响范围**：仅删除死代码，不影响实际计算逻辑（后续走逐条处理路径）

### 4. CPU 参考实现放在 `tests/common/` 目录
- **理由**：测试专用的 CPU 参考实现，不污染主库代码。
- **结构**：`tests/common/hash_reference.rs` 包含 6 种算法的 CPU 实现。
- **关键约束**：CPU 实现必须与 WGSL 的 bit 映射策略完全一致（修复后的独立计数器方案）

### 5. 测试数据使用固定合成图像
- **理由**：避免引入外部图像文件，保证测试可复现。
- **生成方式**：
  - 全 0 / 全 255（边界测试）
  - 渐变模式（0, 1, 2, ... 255 循环）
  - 固定随机种子生成的伪随机数据

### 6. 基准测试使用 Criterion
- **理由**：与现有 `sha256_bench.rs` 保持一致，利用 Criterion 的统计分析和 HTML 报告。
- **对比维度**：
  - GPU 批量处理 vs CPU