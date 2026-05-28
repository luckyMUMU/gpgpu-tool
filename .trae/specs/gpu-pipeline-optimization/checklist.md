## P0: 正确性修复

- [ ] unpack_u32_to_u8 对 count > data.len() 安全截取，不 panic
- [ ] unpack_u32_to_u8 添加 debug_assert! 在开发期捕获异常调用
- [ ] convolution.wgsl 可分离卷积水平 1D 输出使用 bitcast<u32>(sum) 保留 f32 精度
- [ ] convolution.wgsl 可分离卷积垂直 1D 输入使用 bitcast<f32>() 还原中间值
- [ ] 可分离卷积输出与 Full2D 精度一致（差异 ≤ 1）
- [ ] GpuBatchSubmitter 实现 Drop trait，清理未完成 pending jobs
- [ ] BatchJob 从硬编码 3-buffer 改为 Vec<GpuBuffer> 动态绑定
- [ ] GpuBatchSubmitter submit() 使用动态绑定列表创建 BindGroup

## P1: 线程模型与 API 安全性

- [ ] resize.wgsl 改为每目标像素一个线程模型
- [ ] 批量 2 张图像缩放线程数 > 2
- [ ] device()/queue() 在 CPU 降级模式下返回 Result 而非 panic
- [ ] device()/queue() 标记为 #[deprecated]

## P1: 基础设施统一

- [ ] GpuContext 持有共享 BufferPool，提供 buffer_pool() 方法
- [ ] GpuConvolution 不再持有自有 BufferPool
- [ ] GpuResize 不再持有自有 BufferPool
- [ ] GpuGaussianBlur 通过 GpuContext 获取共享池
- [ ] Sha256Computer 和 Sha256BatchSubmitter 使用 GpuContext 共享池
- [ ] declare_phash_computer! 宏生成的 struct 不含 buffer_pool 字段
- [ ] compute_phash / compute_phash_from_gpu_buffer 使用 GpuContext 共享池
- [ ] GpuBatchSubmitter 接入 BufferPool，复用 staging buffer
- [ ] to_wgpu_usage 统一到 BufferUsage 上的单一方法，三处重复实现移除
- [ ] BufferUsage::Storage 的 wgpu flags 在任何调用位置一致
- [ ] GpuBuffer::from_data/from_bytes 不再无条件添加 COPY_SRC
- [ ] 输出缓冲区显式添加 COPY_SRC，输入缓冲区不添加
- [ ] PipelineCache 缓存命中时做 WGSL 源码 == 比较防碰撞
- [ ] BindGroup 缓存可用，批量场景避免重复创建

## P1: 架构重构

- [ ] BackendDispatcher trait 定义完整
- [ ] Sha256Computer 使用 BackendDispatcher 替代手动降级分支
- [ ] PerceptualHasher 使用 BackendDispatcher 替代手动降级分支
- [ ] PerceptualHasher 缓存 PHasherCpu 实例
- [ ] PerceptualHasher 支持可选高斯模糊预处理（with_blur 配置）
- [ ] blur_gpu() → resize_gpu() → hash_gpu() 完整零拷贝链路可用
- [ ] CPU 降级路径支持高斯模糊预处理
- [ ] PerceptualHasher::compute() 拆分为 preprocess / resize / compute_hash 三阶段
- [ ] 各阶段可独立调用
- [ ] GpuPipelineBuilder 支持 blur() / resize() / hash() 链式声明
- [ ] GpuPipelineBuilder::execute() 自动推导零拷贝传递路径
- [ ] GpuPipelineBuilder 在 GPU 不可用时自动降级到 CPU

## P2: WGSL 着色器硬件优化

- [ ] 2D 图像处理着色器支持 @workgroup_size(8,8,1) 2D 模式
- [ ] PipelineDescriptor 和 ComputePipeline 支持运行时切换 workgroup size
- [ ] convolution.wgsl 使用 var<workgroup> LDS 共享内存优化邻域滤波
- [ ] LDS 数据布局使用 Struct of Arrays + padding 避免 Bank Conflict
- [ ] 保留无 LDS 的回退路径
- [ ] PipelineDescriptor 支持 Push Constant 布局定义
- [ ] ComputePipeline dispatch/encode_dispatch_into 支持 Push Constant 数据
- [ ] 感知哈希着色器 PhashParams 通过 Push Constant 传递
- [ ] 缩放着色器 ResizeParams 通过 Push Constant 传递
- [ ] BufferPool size_class 增加 256 字节对齐约束
- [ ] GpuContext 查询并缓存 GPU 适配器 CU 数量
- [ ] dispatch 计算逻辑增加低 occupancy 警告
- [ ] sha256_chained.wgsl 偏移查找为 O(1)

## P2: 其他审查修复

- [ ] ChainedMatcher bk_tree_plus_linear 策略从 Union 修正为 FirstHit
- [ ] resize_batch_gpu 对空输入返回 Ok(vec![])
- [ ] resize_batch 和 resize_batch_gpu 验证逻辑不重复

## P2: 性能基准

- [ ] 性能基准覆盖单图像延迟、批量吞吐量、零拷贝对比、模糊预处理对比
- [ ] 性能基准覆盖 Workgroup Size 对比、LDS 优化对比、Push Constant 对比
- [ ] 性能基准覆盖可分离卷积精度修复前后对比

## 全局验证

- [ ] cargo build 通过
- [ ] cargo clippy 无新警告
- [ ] cargo test 通过（GPU 环境）
