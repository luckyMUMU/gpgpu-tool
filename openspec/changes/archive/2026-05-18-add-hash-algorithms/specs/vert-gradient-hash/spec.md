## ADDED Requirements

### Requirement: Vertical Gradient Hash 计算
系统 SHALL 提供基于垂直方向梯度变化的感知哈希计算能力，对每幅输入图像输出 64bit 哈希值。

#### Scenario: 单幅图像垂直梯度哈希
- **WHEN** 调用方传入一幅 9x8 灰度图像的 72 字节像素数据
- **THEN** 系统计算每列相邻像素的垂直差值，差值大于 0 生成 1bit，输出 1 个 u64 哈希值

#### Scenario: 批量图像垂直梯度哈希
- **WHEN** 调用方传入多幅 9x8 灰度图像的像素数据
- **THEN** 系统通过单次 GPU dispatch 并行计算所有图像的垂直梯度哈希，返回与输入数量相同的 u64 数组
