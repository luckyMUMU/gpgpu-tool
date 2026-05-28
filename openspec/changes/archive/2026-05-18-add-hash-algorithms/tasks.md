## 1. 公共基础设施

- [x] 1.1 在 `src/tasks/` 下创建 `hash_common.rs`，定义感知哈希公共参数结构体 `PhashParams`（图像宽度、高度、图像数量）
- [x] 1.2 在 `src/tasks/hash_common.rs` 中定义 `PerceptualHashComputer` trait，统一 `compute` 接口
- [x] 1.3 在 `src/tasks/mod.rs` 中导出 `hash_common` 模块

## 2. Mean Hash 实现

- [x] 2.1 创建 `src/tasks/mean_hash.rs`，实现 `MeanHashComputer` 结构体及 `new`/`with_workgroup_size`/`compute` 方法
- [x] 2.2 创建 `src/tasks/mean_hash.wgsl`，实现 8x8 图像均值哈希计算着色器
- [x] 2.3 在 `src/tasks/mod.rs` 中导出 `mean_hash` 模块

## 3. Gradient Hash 实现

- [x] 3.1 创建 `src/tasks/gradient_hash.rs`，实现 `GradientHashComputer` 结构体
- [x] 3.2 创建 `src/tasks/gradient_hash.wgsl`，实现 8x9 图像水平梯度哈希计算着色器
- [x] 3.3 在 `src/tasks/mod.rs` 中导出 `gradient_hash` 模块

## 4. Block Hash 实现

- [x] 4.1 创建 `src/tasks/block_hash.rs`，实现 `BlockHashComputer` 结构体
- [x] 4.2 创建 `src/tasks/block_hash.wgsl`，实现 16x16 图像分块哈希计算着色器
- [x] 4.3 在 `src/tasks/mod.rs` 中导出 `block_hash` 模块

## 5. Vertical Gradient Hash 实现

- [x] 5.1 创建 `src/tasks/vert_gradient_hash.rs`，实现 `VertGradientHashComputer` 结构体
- [x] 5.2 创建 `src/tasks/vert_gradient_hash.wgsl`，实现 9x8 图像垂直梯度哈希计算着色器
- [x] 5.3 在 `src/tasks/mod.rs` 中导出 `vert_gradient_hash` 模块

## 6. Double Gradient Hash 实现

- [x] 6.1 创建 `src/tasks/double_gradient_hash.rs`，实现 `DoubleGradientHashComputer` 结构体
- [x] 6.2 创建 `src/tasks/double_gradient_hash.wgsl`，实现 9x9 图像双梯度哈希计算着色器
- [x] 6.3 在 `src/tasks/mod.rs` 中导出 `double_gradient_hash` 模块

## 7. Median Hash 实现

- [x] 7.1 创建 `src/tasks/median_hash.rs`，实现 `MedianHashComputer` 结构体
- [x] 7.2 创建 `src/tasks/median_hash.wgsl`，实现 8x8 图像中值哈希计算着色器（使用简化中值算法）
- [x] 7.3 在 `src/tasks/mod.rs` 中导出 `median_hash` 模块

## 8. 测试与验证

- [x] 8.1 为 Mean Hash 编写单元测试，验证已知输入的输出哈希值
- [x] 8.2 为 Gradient Hash 编写单元测试
- [x] 8.3 为 Block Hash 编写单元测试
- [x] 8.4 为 Vertical Gradient Hash 编写单元测试
- [x] 8.5 为 Double Gradient Hash 编写单元测试
- [x] 8.6 为 Median Hash 编写单元测试
- [x] 8.7 运行 `cargo test` 确保所有测试通过
- [x] 8.8 运行 `cargo clippy` 检查代码质量
