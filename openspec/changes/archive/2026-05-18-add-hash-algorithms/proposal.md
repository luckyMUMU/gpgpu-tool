## Why

当前 `wgpu-compute-engine` 仅支持 SHA-256 哈希计算。为满足图像感知哈希（Perceptual Hash）等场景需求，需要扩展 GPU 哈希算法库，支持多种经典图像/数据哈希算法，利用 GPU 并行能力加速批量处理。

## What Changes

- 新增 6 种 GPU 并行哈希算法支持：
  - **Mean Hash**（均值哈希）
  - **Gradient Hash**（梯度哈希）
  - **Block Hash**（分块哈希）
  - **Vertical Gradient Hash**（垂直梯度哈希）
  - **Double Gradient Hash**（双梯度哈希）
  - **Median Hash**（中值哈希）
- 每种算法提供独立的 `*Computer` Rust 结构体 + WGSL 计算着色器
- 统一算法注册接口，支持通过枚举选择哈希算法
- 保持与现有 `Sha256Computer` 一致的 API 风格（`new()` / `compute()`）

## Capabilities

### New Capabilities
- `mean-hash`: 基于像素均值生成二值哈希，适用于图像相似度快速比对
- `gradient-hash`: 基于水平方向梯度变化生成哈希，对亮度变化鲁棒
- `block-hash`: 基于分块区域均值比较生成哈希，兼顾局部与全局特征
- `vert-gradient-hash`: 基于垂直方向梯度变化生成哈希，与 Gradient Hash 正交互补
- `double-gradient-hash`: 同时利用水平和垂直梯度生成更高维哈希
- `median-hash`: 基于像素中值生成哈希，对异常值更鲁棒

### Modified Capabilities
- （无现有 spec 变更）

## Impact

- 新增 6 个 Rust 模块（`src/tasks/` 下）
- 新增 6 个 WGSL 着色器文件（`src/tasks/` 下）
- `src/tasks/mod.rs` 导出新增模块
- 可能新增公共枚举 `HashAlgorithm` 用于算法选择
- 无外部依赖变更，继续使用 wgpu + bytemuck
