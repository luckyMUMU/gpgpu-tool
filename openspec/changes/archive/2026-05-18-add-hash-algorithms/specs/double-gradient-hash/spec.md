## ADDED Requirements

### Requirement: Double Gradient Hash 计算
系统 SHALL 提供同时利用水平和垂直梯度变化的感知哈希计算能力，对每幅输入图像输出 64bit 哈希值。

#### Scenario: 单幅图像双梯度哈希
- **WHEN** 调用方传入一幅 9x9 灰度图像的 81 字节像素数据
- **THEN** 系统分别计算水平方向和垂直方向的相邻像素差值，组合生成 1 个 u64 哈希值

#### Scenario: 批量图像双梯度哈希
- **WHEN** 调用方传入多幅 9x9 灰度图像的像素数据
- **THEN** 系统通过单次 GPU dispatch 并行计算所有图像的双梯度哈希，返回与输入数量相同的 u64 数组
