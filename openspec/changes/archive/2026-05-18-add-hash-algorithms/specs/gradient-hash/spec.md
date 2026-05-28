## ADDED Requirements

### Requirement: Gradient Hash 计算
系统 SHALL 提供基于水平方向梯度变化的感知哈希计算能力，对每幅输入图像输出 64bit 哈希值。

#### Scenario: 单幅图像梯度哈希
- **WHEN** 调用方传入一幅 8x9 灰度图像的 72 字节像素数据
- **THEN** 系统计算每行相邻像素的水平差值，差值大于 0 生成 1bit，输出 1 个 u64 哈希值

#### Scenario: 批量图像梯度哈希
- **WHEN** 调用方传入多幅 8x9 灰度图像的像素数据
- **THEN** 系统通过单次 GPU dispatch 并行计算所有图像的梯度哈希，返回与输入数量相同的 u64 数组
