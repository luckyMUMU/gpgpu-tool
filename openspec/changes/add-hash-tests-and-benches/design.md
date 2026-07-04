## Context

当前已有 6 种感知哈希算法的 GPU 实现（`src/tasks/*_hash.rs` + `*_hash.wgsl`），但测试覆盖仅限于 SHA-256。现有测试结构：
- `tests/sha256_test.rs` — 功能测试
- `benches/sha256_bench.rs` — 性能基准测试

本次需要为 6 种新哈希算法建立同样的测试体系，并扩展支持多尺寸图像输入和测试报告生成。

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
- **支持多尺寸图像输入**：每种算法支持至少 3 种不同图像尺寸
- 每种算法至少 3 个功能测试（单图像、批量图像、空输入）
- 每种算法至少 1 个性能基准测试（GPU vs CPU）
- **测试运行后生成结构化分析报告**：Markdown 格式，包含测试通过率、性能对比、尺寸影响分析
- 提供 CPU 参考实现用于结果比对
- 测试数据可复现（固定随机种子或固定像素值）

**Non-Goals:**
- 不测试图像预处理（缩放、灰度化）
- 不引入真实图像文件（使用合成数据）
- 不测试汉明距离计算
- 不实现动态图像尺寸（调用方须确保同批次图像尺寸一致）

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

### 4. 多尺寸图像支持策略

**支持矩阵：**

| 算法 | 推荐尺寸 | 额外支持尺寸 | 输出 |
|------|---------|------------|------|
| Mean Hash | 8x8 | 16x16, 32x32 | 64bit |
| Median Hash | 8x8 | 16x16, 32x32 | 64bit |
| Gradient Hash | 8x9 | 16x17, 32x33 | 64bit |
| VertGradient Hash | 9x8 | 17x16, 33x32 | 64bit |
| Block Hash | 16x16 | 32x32 | 64bit |
| DoubleGradient Hash | 9x9 | 17x17, 33x33 | 64bit |

**实现方式：**
- WGSL 着色器已支持任意尺寸（通过 `params.width` / `params.height`）
- Rust 侧 `compute()` 方法已支持任意尺寸（从像素数据推导 width/height）
- **约束**：同批次所有图像须尺寸一致（由 `pixels_per_image` 推导）
- **测试覆盖**：每种算法至少测试 2 种不同尺寸

**大尺寸处理策略：**
- 当图像像素数 > 64 时，Mean/Median Hash 仍只取前 64 个像素生成哈希
- Gradient Hash 在 width-1 或 height-1 > 64 时，只保留前 64 个梯度比较
- Block Hash 始终将图像分为 8x8 块（块大小 = width/8 × height/8）

### 5. 测试报告生成策略

**报告内容：**
```markdown
# 感知哈希测试报告

## 1. 测试摘要
- 测试时间: 2026-05-18 14:30:00
- GPU 信息: NVIDIA GeForce RTX 4090
- 总测试数: 36
- 通过率: 100% (36/36)

## 2. 功能测试结果
| 算法 | 8x8 | 16x16 | 32x32 | 批量 | 空输入 |
|------|-----|-------|-------|------|--------|
| Mean | ✅ | ✅ | ✅ | ✅ | ✅ |
| ... | | | | | |

## 3. 性能基准测试
| 算法 | 尺寸 | Batch=1 | Batch=100 | Batch=1000 | 加速比 |
|------|------|---------|-----------|------------|--------|
| Mean | 8x8 | 0.5ms | 2ms | 15ms | 45x |
| ... | | | | | |

## 4. 尺寸影响分析
| 算法 | 8x8 | 16x16 | 32x32 | 趋势 |
|------|-----|-------|-------|------|
| Mean | 0.5ms | 0.6ms | 0.8ms | 线性增长 |
| ... | | | | |
```

**实现方式：**
- `tests/common/report.rs`：报告生成器
- 测试运行时收集结果，输出到 `target/test-reports/` 目录
- 报告格式：Markdown（便于阅读和版本控制）

### 6. CPU 参考实现放在 `tests/common/` 目录
- **理由**：测试专用的 CPU 参考实现，不污染主库代码。
- **结构**：`tests/common/hash_reference.rs` 包含 6 种算法的 CPU 实现。
- **关键约束**：CPU 实现必须与 WGSL 的 bit 映射策略完全一致（修复后的独立计数器方案）
- **多尺寸支持**：CPU 实现须支持可变 width/height 参数

### 7. 测试数据使用固定合成图像
- **理由**：避免引入外部图像文件，保证测试可复现。
- **生成方式**：
  - 全 0 / 全 255（边界测试）
  - 渐变模式（0, 1, 2, ... 255 循环）
  - 固定随机种子生成的伪随机数据
  - **多尺寸**：为每种尺寸生成对应大小的测试数据

### 8. 基准测试使用 Criterion
- **理由**：与现有 `sha256_bench.rs` 保持一致，利用 Criterion 的统计分析和 HTML 报告。
- **对比维度**：
  - GPU 批量处理 vs CPU 串行处理
  - 不同 batch size（1, 10, 100, 1000）
  - 不同 workgroup_size（64, 128, 256, 512）
  - **不同图像尺寸**（8x8、16x16、32x32）

### 9. 每种算法独立测试文件
- **理由**：便于单独运行和定位问题。
- **文件命名**：`tests/mean_hash_test.rs`、`tests/gradient_hash_test.rs` 等。

### 10. 基准测试按算法分组
- **理由**：便于选择性运行。
- **文件命名**：`benches/mean_hash_bench.rs`、`benches/gradient_hash_bench.rs` 等。
- **Cargo.toml**：为每个 bench 添加 `[[bench]]` 条目。

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| GPU 测试在 CI 环境中失败（无 GPU） | 使用 `#[ignore]` 标记 GPU 测试，本地运行 `cargo test -- --ignored` |
| CPU 参考实现与 GPU 实现结果不一致 | 仔细核对算法逻辑，确保 WGSL 和 Rust 语义一致；已修复 bit 映射缺陷 |
| 基准测试运行时间过长 | 减少迭代次数，使用较小的 batch size 范围；大尺寸测试减少样本数 |
| WGSL 修复引入新的回归问题 | 修复后立即运行 cargo test 验证现有测试不受影响 |
| buffer.rs channel 引入性能开销 | channel 仅用于错误路径，正常路径无额外开销 |
| sha256.rs 死代码删除误伤有效逻辑 | 仔细核对删除范围，确保只删除未使用变量 |
| 多尺寸支持导致 WGSL 复杂度增加 | 保持现有参数传递方式（width/height 已通过 uniform 传入），无需修改着色器逻辑 |
| 大尺寸图像测试占用过多 GPU 内存 | 控制 batch size，大尺寸图像使用较小的 batch（如 10 而非 1000） |

## Migration Plan

- 先修复 WGSL bit 映射缺陷（已完成）
- 修复 buffer.rs 错误传播（进行中）
- 清理 sha256.rs 死代码（进行中）
- 验证修复不破坏现有功能（cargo test 通过）
- 实现多尺寸图像支持（WGSL 已支持，仅需测试覆盖）
- 添加测试和基准测试代码
- 实现测试报告生成器
- 纯新增测试代码，无破坏性变更

## Open Questions

1. ~~是否需要为每种算法提供多组测试图像尺寸？~~ —— **已决定**：支持 8x8/16x16/32x32 三种尺寸
2. 基准测试是否需要在报告中包含 GPU 型号信息？—— 当前设计：通过 `ctx.adapter_info()` 打印
3. 测试报告是否需要 JSON 格式供 CI 解析？—— 当前设计：仅 Markdown，后续可扩展
4. 大尺寸图像（如 128x128）是否需要支持？—— 当前设计：最大 32x32，更大尺寸后续评估
