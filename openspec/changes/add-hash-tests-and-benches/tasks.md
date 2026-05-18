## 0. WGSL 修复（前置任务）

- [x] 0.1 修复 `src/tasks/gradient_hash.wgsl`：改用独立 `bit_pos` 计数器，避免最后一行梯度信息丢失
- [x] 0.2 修复 `src/tasks/double_gradient_hash.wgsl`：水平/垂直梯度分别使用独立计数器，避免信息截断
- [x] 0.3 运行 `cargo check` 验证编译通过
- [x] 0.4 运行 `cargo test` 验证现有测试不受影响

## 1. CPU 参考实现

- [ ] 1.1 创建 `tests/common/mod.rs`，定义测试公共模块
- [ ] 1.2 创建 `tests/common/hash_reference.rs`，实现 6 种算法的 CPU 参考版本（使用独立 bit_pos 计数器，与修复后的 WGSL 一致）
- [ ] 1.3 创建 `tests/common/test_data.rs`，实现合成图像生成器（全0、全255、渐变、随机）

## 2. Mean Hash 测试

- [ ] 2.1 创建 `tests/mean_hash_test.rs`，实现单图像、批量图像、空输入测试
- [ ] 2.2 创建 `benches/mean_hash_bench.rs`，实现 GPU vs CPU 性能对比
- [ ] 2.3 更新 `Cargo.toml` 添加 `mean_hash_bench` bench 条目

## 3. Gradient Hash 测试

- [ ] 3.1 创建 `tests/gradient_hash_test.rs`，实现单图像、批量图像、空输入测试
- [ ] 3.2 创建 `benches/gradient_hash_bench.rs`，实现 GPU vs CPU 性能对比
- [ ] 3.3 更新 `Cargo.toml` 添加 `gradient_hash_bench` bench 条目

## 4. Block Hash 测试

- [ ] 4.1 创建 `tests/block_hash_test.rs`，实现单图像、批量图像、空输入测试
- [ ] 4.2 创建 `benches/block_hash_bench.rs`，实现 GPU vs CPU 性能对比
- [ ] 4.3 更新 `Cargo.toml` 添加 `block_hash_bench` bench 条目

## 5. Vertical Gradient Hash 测试

- [ ] 5.1 创建 `tests/vert_gradient_hash_test.rs`，实现单图像、批量图像、空输入测试
- [ ] 5.2 创建 `benches/vert_gradient_hash_bench.rs`，实现 GPU vs CPU 性能对比
- [ ] 5.3 更新 `Cargo.toml` 添加 `vert_gradient_hash_bench` bench 条目

## 6. Double Gradient Hash 测试

- [ ] 6.1 创建 `tests/double_gradient_hash_test.rs`，实现单图像、批量图像、空输入测试
- [ ] 6.2 创建 `benches/double_gradient_hash_bench.rs`，实现 GPU vs CPU 性能对比
- [ ] 6.3 更新 `Cargo.toml` 添加 `double_gradient_hash_bench` bench 条目

## 7. Median Hash 测试

- [ ] 7.1 创建 `tests/median_hash_test.rs`，实现单图像、批量图像、空输入测试
- [ ] 7.2 创建 `benches/median_hash_bench.rs`，实现 GPU vs CPU 性能对比
- [ ] 7.3 更新 `Cargo.toml` 添加 `median_hash_bench` bench 条目

## 8. 验证与清理

- [ ] 8.1 运行 `cargo test` 确保所有新测试通过
- [ ] 8.2 运行 `cargo clippy` 检查代码质量
- [ ] 8.3 运行 `cargo bench` 验证基准测试可正常执行
