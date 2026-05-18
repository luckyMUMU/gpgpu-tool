## Context

当前 `wgpu-compute-engine` 的 `tasks` 模块仅实现了 `sha256` 一种哈希算法。该算法通过 `Sha256Computer` 结构体 + WGSL 计算着色器完成，API 风格为：`new(ctx)` 创建实例，`compute(ctx, inputs)` 批量计算。

本次需要新增 6 种图像感知哈希算法，均基于像素级操作，天然适合 GPU 并行。设计目标是保持与现有 SHA-256 模块一致的架构风格，同时提取公共模式减少重复代码。

## Goals / Non-Goals

**Goals:**
- 为 6 种哈希算法各提供 GPU 并行实现
- 保持与 `Sha256Computer` 一致的 API 风格
- 支持批量图像数据输入（如 `&[Vec<u8>]` 或 `&[ImageData]`）
- 每种算法输出固定长度的二进制哈希（如 `Vec<u8>` 或 `[u8; N]`）

**Non-Goals:**
- 图像解码（假设输入已是灰度/缩放的原始像素数据）
- 汉明距离计算（仅生成哈希，距离计算由调用方完成）
- 与 SHA-256 共用着色器逻辑（各算法独立 WGSL 文件）

## Decisions

### 1. 每种算法独立模块 + 独立 WGSL
- **理由**：各算法的计算逻辑差异大（均值 vs 梯度 vs 中值），难以抽象为统一着色器。独立文件便于单独维护和调优 workgroup_size。
- **替代方案**：统一着色器通过参数分支 ——  rejected，会导致着色器臃肿且难以优化。

### 2. 输入格式统一为 `&[Vec<u8>]`（每幅图像的像素字节序列）
- **理由**：感知哈希通常前置有图像预处理（缩放、灰度化），不在本库职责内。原始字节输入最通用。
- **替代方案**：引入 image crate 处理解码 —— rejected，增加不必要依赖。

### 3. 输出格式为 `Vec<u64>`（每位图像一个 u64 哈希值）
- **理由**：感知哈希通常为 64bit（8x8 比较结果），u64 是 Rust 最自然的表达。WGSL 中输出为 `array<u32>`，Rust 侧将两个 u32 组合为 u64。
- **替代方案**：输出 `[u8; 8]` —— rejected，u64 更便于后续汉明距离计算。

### 4. 提取公共 `PerceptualHashComputer` trait（可选）
- **理由**：6 种算法结构高度相似（`new` / `with_workgroup_size` / `compute`），可通过 trait 统一接口，方便后续动态分发。
- **实现方式**：
  ```rust
  pub trait PerceptualHashComputer {
      fn compute(&self, ctx: &GpuContext, images: &[Vec<u8>]) -> Result<Vec<u64>, GpuError>;
  }
  ```

### 5. 不修改现有 `ComputePipeline` 绑定布局
- **理由**：现有绑定组布局（storage read + storage read_write + uniform）足够承载感知哈希的参数和数据传输。
- **参数结构**：通过 uniform buffer 传递图像宽度、高度、算法特定参数。

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| WGSL 中排序/中值计算性能差 | Median Hash 使用简化算法（如局部中值近似），或分阶段计算 |
| 图像尺寸不统一导致 batch 复杂 | 要求调用方预处理为统一尺寸（如 8x8、16x16），compute 接口文档明确约束 |
| 6 个新模块导致 tasks 目录膨胀 | 后续可考虑按 `tasks/hash/` 子目录组织，但本次保持扁平以遵循现有风格 |
| 不同 GPU workgroup_size 最优值不同 | 提供 `with_workgroup_size` 构造函数，与 Sha256Computer 保持一致 |

## Migration Plan

- 纯新增功能，无破坏性变更
- 现有 `Sha256Computer` 不受影响
- 逐步添加：先实现 Mean / Gradient（最简单），再扩展至 Median（最复杂）

## Open Questions

1. 输入图像是否需要包含宽度/高度元数据？—— 当前设计：通过 uniform 参数传入，要求调用方已知尺寸
2. 是否需要支持多精度（如 16x16 输出 256bit）？—— 当前设计：统一 64bit 输出，后续可扩展
