## ADDED Requirements

### Requirement: Block Hash 计算
系统 SHALL 提供基于分块区域均值比较的感知哈希计算能力，对每幅输入图像输出 64bit 哈希值。

#### Scenario: 单幅图像分块哈希
- **WHEN** 调用方传入一幅 16x16 灰度图像的 256 字节像素数据
- **THEN** 系统将图像分为 8x8 个 2x2 块，计算每块均值，相邻块均值比较生成 1bit，输出 1 个 u64 哈希值

#### Scenario: 批量图像分块哈希
- **WHEN** 调用方传入多幅 16x16 灰度图像的像素数据
- **THEN** 系统通过单次 GPU dispatch 并行计算所有图像的分块哈希，返回与输入数量相同的 u64 数组
