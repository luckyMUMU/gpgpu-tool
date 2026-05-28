## ADDED Requirements

### Requirement: Mean Hash 计算
系统 SHALL 提供基于像素均值的感知哈希计算能力，对每幅输入图像输出 64bit 哈希值。

#### Scenario: 单幅图像均值哈希
- **WHEN** 调用方传入一幅 8x8 灰度图像的 64 字节像素数据
- **THEN** 系统计算所有像素的平均值，每个像素与均值比较生成 1bit，输出 1 个 u64 哈希值

#### Scenario: 批量图像均值哈希
- **WHEN** 调用方传入多幅 8x8 灰度图像的像素数据
- **THEN** 系统通过单次 GPU dispatch 并行计算所有图像的均值哈希，返回与输入数量相同的 u64 数组
