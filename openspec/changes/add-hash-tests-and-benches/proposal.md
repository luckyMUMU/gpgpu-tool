## Why

当前 `wgpu-compute-engine` 已为 6 种感知哈希算法（Mean、Gradient、Block、Vertical Gradient、Double Gradient、Median）提供了 GPU 实现，但缺少功能测试和性能基准测试。缺乏测试会导致以下问题：
- 无法验证算法正确性（哈希值是否符合预期）
- 无法发现回归问题（后续修改是否破坏现有功能）
- 无法量化 GPU 加速效果（与 CPU 实现对比）
- 无法评估不同 workgroup_size 对性能的影响

此外，代码审查发现以下问题需要一并修复：

1. **WGSL bit 映射缺陷**：Gradient Hash 和 Double Gradient Hash 使用像素索引作为 bit 位置，导致部分梯度信息丢失
2. **buffer.rs 错误传播缺陷**：`download` 方法中 `map_async` 的错误仅记录日志但未返回给调用者，映射失败时可能导致 panic
3. **sha256.rs 死代码**：`compute_multi_block_batch` 方法中遗留了已分配但未使用的 `input_data`，造成不必要的内存分配

## What Changes

- **修复 WGSL bit 映射缺陷**：
  - Gradient Hash：改用独立 `bit_pos` 计数器，避免最后一行梯度信息丢失
  - Double Gradient Hash：水平/垂直梯度分别使用独立计数器，避免信息截断
- **修复 buffer.rs 错误传播**：使用 channel 同步获取 `map_async` 结果，错误通过 `GpuError` 返回
- **清理 sha256.rs 死代码**：删除 `compute_multi_block_batch` 中未使用的 `input_data` 分配逻辑
- **支持多尺寸图像输入**：
  - Mean/Median Hash：支持 8x8、16x16、32x32 等多种尺寸（输出始终 64bit）
  - Gradient/VertGradient Hash：支持 8x9、16x17、32x33 等多种尺寸
  - Block Hash：支持 16x16、32x32 等多种尺寸（分块大小自适应）
  - Double Gradient Hash：支持 9x9、17x17、33x33 等多种尺寸
- 为 6 种感知哈希算法各添加功能测试（正确性验证）
- 为 6 种感知哈希算法各添加性能基准测试（GPU vs CPU 对比）
- **测试运行后生成分析报告**：包含测试通过率、GPU vs CPU 加速比、不同尺寸/批次性能对比
- 提供 CPU 参考实现用于结果比对
- 测试覆盖单图像、批量图像、空输入、多尺寸等场景
- 性能测试覆盖不同 batch size、workgroup_size 和图像尺寸

## Capabilities

### New Capabilities
- `hash-functional-tests`: 6 种哈希算法的功能测试套件（支持多尺寸图像）
- `hash-performance-benches`: 6 种哈希算法的性能基准测试套件（支持多尺寸图像）
- `hash-test-report`: 测试运行后的自动化分析报告生成
- `multi-size-image-support`: 感知哈希算法支持多种图像尺寸输入

### Modified Capabilities
- `gradient-hash`: WGSL 中 bit 映射逻辑修正（独立计数器替代像素索引）
- `double-gradient-hash`: WGSL 中 bit 映射逻辑修正（独立计数器替代像素索引）
- `buffer-download`: 错误传播逻辑修正（map_async 错误通过 GpuError 返回）
- `sha256-multi-block`: 清理死代码（删除未使用的 input_data 分配）

## Impact

- 修改 WGSL 文件：`src/tasks/gradient_hash.wgsl`、`src/tasks/double_gradient_hash.wgsl`
- 修改 Rust 文件：`src/buffer.rs`、`src/tasks/sha256.rs`
- 可能修改 Rust 文件：`src/tasks/*_hash.rs`（支持多尺寸参数传递）
- 新增测试文件：`tests/hash_*.rs`、`tests/common/*.rs`
- 新增基准测试文件：`benches/hash_*.rs`
- 新增报告生成模块：`tests/common/report.rs`
- 可能新增 `dev-dependencies`（如 `image` crate 用于测试数据生成）
- 更新 `Cargo.toml` 添加 bench 条目
