# Changelog

本项目的所有重要变更均会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.2.0] - 2026-05-24

### 新增

- `HashSize` 结构体替代已废弃的 `HashBits` 枚举，支持任意网格尺寸（8/16/32/64）
- 所有 6 种感知哈希算法的灵活哈希尺寸支持
- `PerceptualHashComputer` trait，提供 `compute_sized()` 方法
- `compute_phash_from_gpu_buffer()` 零拷贝 GPU 管线接口
- GPU 二维卷积模块（`GpuConvolution`），支持 Full2D 与 Separable 两种模式
- GPU 高斯模糊模块（`GpuGaussianBlur`），基于可分离卷积实现
- 二面体变换模块（`DihedralHashes64`/`DihedralHashes256`），支持旋转/翻转不变匹配
- 哈希匹配器策略模式：`LinearScanMatcher`、`BkTreeMatcher`、`ChainedMatcher`、`HashMatcherFacade`
- PDQ 哈希算法（通过 `pdq` feature gate 启用）：GPU DCT-II + CPU 量化
- `pixel_pack` 模块，支持 u8↔u32 像素格式转换
- `BorderMode` 枚举（Zero/Clamp/Reflect），用于卷积边界处理
- `ConvMode` 枚举（Full2D/Separable），用于卷积模式选择
- `GpuResizeConfig` 可配置 GPU 图像缩放行为
- `BufferPoolConfig` 构建器模式，用于缓冲池自定义配置
- SHA-256 链式着色器，支持多块消息（>55 字节）
- `Sha256BatchSubmitter` 真正的批量提交器
- CPU 降级实现：`Sha256Cpu`、`PHasherCpu`
- `cpu-fallback` feature flag（默认启用）
- `pdq` feature flag
- `ComputeBackend` 枚举，用于 GPU/CPU 后端检测
- `GpuContext::try_device()`/`try_queue()` 安全 CPU 降级访问方法
- 软件渲染适配器降级路径

### 变更

- 项目从 "wgpu-compute-engine" 重命名为 "GPGPU-tool"
- `BufferPool::release()` 现在需要 `BufferUsage` 参数（破坏性变更，修复类型标注缺陷）
- `GpuBuffer::from_raw()`/`into_raw()`/`raw()` 可见性改为 `pub(crate)`
- `GpuBuffer::download()` 现在返回 `Result` 而非通道关闭时 panic
- 管线缓存键包含 `Vec<BindingType>`，确保缓存正确区分
- `PerceptualHasher` 新增 `with_hash_size()`、`with_config()`、`with_full_config()`、`with_full_config_and_gpu_resize()` 配置方法
- 哈希算法模块改用 `declare_phash_computer!` + `impl_phash_computer_simple!`/`impl_phash_computer_custom_dims!` 宏

### 废弃

- `HashBits` 枚举（由 `HashSize` 替代）

### 修复

- BufferPool release 使用硬编码导致的严重缺陷
- GpuBuffer download 在通道关闭时 panic 的问题
- Sha256BatchSubmitter 缓冲区提前释放的竞态条件
- 基准测试参考实现算法与 GPU WGSL 不一致的问题

## [0.1.0] - 2026-05-18

### 新增

- 核心 GPU 计算引擎：GpuContext、GpuBuffer、BufferPool、ComputePipeline、GpuBatchSubmitter
- SHA-256 并行哈希（单块模式）
- 6 种感知哈希算法：Mean、Median、Gradient、Block、VertGradient、DoubleGradient
- BK-tree 近似最近邻搜索
- GPU 图像缩放（盒式滤波）
- `image` feature，支持 DynamicImage
- 基于 fxhash 的管线缓存
- 异步批量提交模式
