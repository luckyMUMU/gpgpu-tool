## 0. WGSL 修复（前置任务）

- [x] 0.1 修复 `src/tasks/gradient_hash.wgsl`：改用独立 `bit_pos` 计数器，避免最后一行梯度信息丢失
- [x] 0.2 修复 `src/tasks/double_gradient_hash.wgsl`：水平/垂直梯度分别使用独立计数器，避免信息截断
- [x] 0.3 运行 `cargo check` 验证编译通过
- [x] 0.4 运行 `cargo test` 验证现有测试不受影响

## 0.5 代码质量修复（前置任务）

- [x] 0.5.1 修复 `src/buffer.rs` `download` 方法：使用 channel 同步获取 `map_async` 结果，错误通过 `GpuError::MapFailed` 返回
- [x] 0.5.2 在 `src/error.rs` 新增 `GpuError::MapFailed(String)` 错误类型
- [x] 0.5.3 清理 `src/tasks/sha256.rs` `compute_multi_block_batch` 死代码：删除未使用的 `MsgBlocks` 结构体及 `msg_blocks` 变量
- [x] 0.5.4 运行 `cargo check` 验证编译通过
- [x] 0.5.5 运行 `cargo test` 验证现有测试不受影响

## 1. CPU 参考实现（支持多尺寸）

- [x] 1.1 创建 `tests/common/mod.rs`，定义测试公共模块
- [x] 1.2 创建 `tests/common/hash_reference.rs`，实现 6 种算法的 CPU 参考版本
  - 支持可变 width/height 参数
  - Mean/Median：像素 > 64 时取前 64 个
  - Gradient：梯度比较 > 64 时取前 64 个
  - Block：始终分 8x8 块
  - 使用独立 bit_pos 计数器，与修复后的 WGSL 一致
- [x] 1.3 创建 `tests/common/test_data.rs`，实现合成图像生成器
  - 支持多尺寸：8x8、16x16、32x32 及算法特定尺寸
  - 生成模式：全0、全255、渐变、随机

## 2. 测试报告生成器

- [x] 2.1 创建 `tests/common/report.rs`，实现 Markdown 报告生成器
  - 测试摘要（时间、GPU信息、通过率）
  - 功能测试结果表格
  - 性能基准测试表格
  - 尺寸影响分析表格
- [x] 2.2 报告输出到 `target/test-reports/` 目录
- [x] 2.3 集成到各测试文件的 `#[test]` 中，测试完成后自动生成报告

## 3. Mean Hash 测试（多尺寸）

- [x] 3.1 创建 `tests/mean_hash_test.rs`
  - 单图像测试：8x8、16x16、32x32
  - 批量图像测试：10 幅 16x16
  - 空输入测试
  - GPU 输出与 CPU 参考实现比对
- [ ] 3.2 创建 `benches/mean_hash_bench.rs`
  - 不同尺寸对比：8x8、16x16、32x32
  - 不同 batch size：1、10、100、1000
  - GPU vs CPU 性能对比
- [ ] 3.3 更新 `Cargo.toml` 添加 `mean_hash_bench` bench 条目

## 4. Gradient Hash 测试（多尺寸）

- [x] 4.1 创建 `tests/gradient_hash_test.rs`
  - 单图像测试：8x9、16x17、32x33
  - 批量图像测试
  - 空输入测试
- [ ] 4.2 创建 `benches/gradient_hash_bench.rs`
  - 不同尺寸对比：8x9、16x17、32x33
  - 不同 batch size
- [ ] 4.3 更新 `Cargo.toml` 添加 `gradient_hash_bench` bench 条目

## 5. Block Hash 测试（多尺寸）

- [x] 5.1 创建 `tests/block_hash_test.rs`
  - 单图像测试：16x16、32x32
  - 批量图像测试
  - 空输入测试
- [ ] 5.2 创建 `benches/block_hash_bench.rs`
  - 不同尺寸对比：16x16、32x32
  - 不同 batch size
- [ ] 5.3 更新 `Cargo.toml` 添加 `block_hash_bench` bench 条目

## 6. Vertical Gradient Hash 测试（多尺寸）

- [x] 6.1 创建 `tests/vert_gradient_hash_test.rs`
  - 单图像测试：9x8、17x16、33x32
  - 批量图像测试
  - 空输入测试
- [ ] 6.2 创建 `benches/vert_gradient_hash_bench.rs`
  - 不同尺寸对比：9x8、17x16、33x32
  - 不同 batch size
- [ ] 6.3 更新 `Cargo.toml` 添加 `vert_gradient_hash_bench` bench 条目

## 7. Double Gradient Hash 测试（多尺寸）

- [x] 7.1 创建 `tests/double_gradient_hash_test.rs`
  - 单图像测试：9x9、17x17、33x33
  - 批量图像测试
  - 空输入测试
- [ ] 7.2 创建 `benches/double_gradient_hash_bench.rs`
  - 不同尺寸对比：9x9、17x17、33x33
  - 不同 batch size
- [ ] 7.3 更新 `Cargo.toml` 添加 `double_gradient_hash_bench` bench 条目

## 8. Median Hash 测试（多尺寸）

- [x] 8.1 创建 `tests/median_hash_test.rs`
  - 单图像测试：8x8、16x16、32x32
  - 批量图像测试
  - 空输入测试
- [ ] 8.2 创建 `benches/median_hash_bench.rs`
  - 不同尺寸对比：8x8、16x16、32x32
  - 不同 batch size
- [ ] 8.3 更新 `Cargo.toml` 添加 `median_hash_bench` bench 条目

## 9. 验证与清理

- [x] 9.1 运行 `cargo test` 确保所有新测试通过
- [x] 9.2 运行 `cargo clippy` 检查代码质量
- [ ] 9.3 运行 `cargo bench` 验证基准测试可正常执行
- [ ] 9.4 检查 `target/test-reports/` 目录是否正确生成 Markdown 报告
- [ ] 9.5 验证报告内容完整性：测试摘要、功能测试、性能基准、尺寸影响分析
