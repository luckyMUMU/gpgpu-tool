## P0: GPU 显存安全（新增，最高优先级）

- [ ] 0.1 在 `src/context.rs` `init_gpu` 和 `init_software_adapter` 中注册 `UncapturedErrorHandler`，设置 `GPU_OOM_FLAG` 和 `GPU_DEVICE_LOST_FLAG` 原子标志
- [ ] 0.2 在 `src/buffer_pool.rs` `acquire` 方法增加 `max_storage_buffer_binding_size` 预检查，超限返回 `GpuError::Oom`（返回类型改为 `Result<Buffer, GpuError>`）
- [ ] 0.3 同步修改 `acquire` 的约 15 处调用点，处理 `Result` 返回值
- [ ] 0.4 在 `src/tasks/hash_common.rs` `compute_phash` 内部增加分批逻辑：按 `DEFAULT_MAX_BATCH_SIZE / per_image_bytes` 计算单批最大图像数，循环处理 chunks
- [ ] 0.5 在 `src/tasks/hash_common.rs` `compute_phash_from_gpu_buffer` 增加分批逻辑（同 0.4）
- [ ] 0.6 在 `src/tasks/phasher_pipeline.rs` `compute_gpu_per_image_pipeline` 循环内增加单图尺寸检查，超限返回 `GpuError::InvalidInput`
- [ ] 0.7 在 `src/tasks/gpu_image_matcher.rs` `compute_hashes` 入口增加显存预算分批加固
- [ ] 0.8 在 `src/context.rs` `backend()` 方法增加 OOM 标志检查，动态返回 `ComputeBackend::Cpu`
- [ ] 0.9 在 `src/backend_dispatcher.rs` 验证 `dispatch_gpu` 能正确响应 `backend()` 动态切换
- [ ] 0.10 运行 `cargo test` 验证编译和测试通过
- [ ] 0.11 运行 `cargo test --features pdq` 验证 PDQ 路径
- [ ] 0.12 手动测试 DX12 后端大图像批量场景，验证 OOM 降级而非崩溃

## P1: GPU 同步开销优化

- [ ] 1.1 优化 `src/batch.rs` `GpuBatchSubmitter::wait_all()`：保留 `device/queue/pool` 引用，仅清空 `encoder` 和 `pending`
- [ ] 1.2 优化 `src/batch.rs` `DualEncoderSubmitter::wait_all()`：同 1.1，保留引用
- [ ] 1.3 解放 `src/batch.rs` `DualEncoderSubmitter`：移除 `#[cfg(feature = "dual-encoder")]`，作为默认实现
- [ ] 1.4 优化 `src/tasks/phasher_pipeline.rs`：跨 chunk 复用 `GpuBatchSubmitter`，避免每 chunk 新建
- [ ] 1.5 优化 `src/pipeline.rs` 非 Push Constant 路径：缓存可复用 Uniform 缓冲区，避免每次 `dispatch_with_params` 创建新缓冲区
- [ ] 1.6 优化 `src/pipeline.rs` `encode_dispatch_with_params_into`：增加 `queue: &Queue` 参数，复用 `reusable_uniform`
- [ ] 1.7 修复 `src/tasks/hash_common.rs:599` 业务层绕过封装：统一调用 `pipeline.encode_dispatch_with_params_into`
- [ ] 1.8 标记 `src/buffer.rs` `download()` 和 `download_batch()` 为 `#[deprecated]`，引导使用 `_with_pool` 版本
- [ ] 1.9 在 `src/tasks/gpu_matcher.rs` 等核心模块推广 `download_batch_with_pool`，合并多个独立 download 为单次 poll
- [ ] 1.10 运行 `cargo test` 验证编译和测试通过

## P2: 着色器算法效率优化

- [ ] 2.1 重写 `src/tasks/resize.wgsl` 缩放着色器：使用积分图(SAT)方法替代区域平均，将 O(src_area) 降为 O(1)（仅大比例下采样场景启用）
- [ ] 2.2 优化 `src/tasks/gpu_resize.rs`：增加积分图构建 pass + 查询 pass 的两阶段调度
- [ ] 2.3 重写 `src/tasks/block_hash.wgsl` `block_mean`：增加预处理 pass 预计算所有 block 均值图，消除 O(block_area) 循环
- [ ] 2.4 优化 `src/tasks/block_hash.wgsl`：提取 `m_current = block_mean(bx, by)` 一次，复用于 left/top 分支
- [ ] 2.5 重写 `src/tasks/pdq_hash.wgsl` DCT 着色器：预计算 64×64 余弦查找表作为 storage buffer 传入
- [ ] 2.6 优化 `src/tasks/pdq_hash.rs` GPU 路径：增加第 3 个 pass 提取 16×16 低频系数，减少 93.75% 下载量
- [ ] 2.7 优化 `src/tasks/pdq_hash.wgsl` workgroup 布局：从 1D (256,1,1) 改为 2D (8,8,1)，利用 LDS 共享 cos 表
- [ ] 2.8 为 `src/tasks/convolution.wgsl` Full2D 模式实现 LDS 共享内存优化：协作加载 (WG_X+2r)×(WG_Y+2r) 像素到 LDS
- [ ] 2.9 优化 `src/tasks/convolution.wgsl` separable_fused 水平 pass：将输入区域加载到 LDS，减少全局内存重复读取
- [ ] 2.10 重写 `src/tasks/hamming.wgsl` `find_nearest_neighbor`：改为并行归约（256 线程协作扫描 256 db，warp shuffle 归约最小值）
- [ ] 2.11 优化 `src/tasks/hamming.wgsl` `hamming_distance_matrix` 加载阶段：所有 256 线程协作加载 query_tile 和 db_tile_matrix
- [ ] 2.12 优化 `src/tasks/pdq_hash.rs` CPU DCT：使用快速 DCT 算法替代朴素实现
- [ ] 2.13 运行 `cargo test` 验证编译和测试通过
- [ ] 2.14 运行 `cargo test --features pdq` 验证 PDQ 路径

## P3: 内存管理优化

- [ ] 3.1 重写 `src/tasks/dihedral.rs` 64-bit 变换：直接在 u64 位上实现 8 种二面体变换（参考 32×32 的 `transpose_32x32` delta-swap），消除所有 `Vec<bool>` 分配
- [ ] 3.2 重写 `src/tasks/dihedral.rs` 256-bit 变换：直接在 `[u64; 4]` 上实现位操作，消除 `Vec<bool>` 分配
- [ ] 3.3 统一 `src/tasks/dihedral.rs` 四种尺寸实现风格：抽象出 `BitMatrix<N>` 泛型结构或统一位操作模式
- [ ] 3.4 优化 `src/tasks/dihedral.rs` `all_variants` 返回类型：改为 `[u64; 8]` 或迭代器，避免 `.to_vec()` 堆分配
- [ ] 3.5 优化 `src/tasks/gpu_matcher.rs` `compute_distance_matrix`：返回扁平 `(Vec<u32>, usize)` 替代 `Vec<Vec<u32>>`，提供 `row(i) -> &[u32]` 辅助方法
- [ ] 3.6 优化 `src/tasks/gpu_matcher.rs` 分块路径：扁平化后直接写入预分配 Vec 的对应区间，消除 chunk 内 Vec 分配
- [ ] 3.7 优化 `src/tasks/matcher.rs` `ChainedMatcher::bk_tree_plus_linear`：使用 `Arc<Vec<u64>>` 共享哈希数据
- [ ] 3.8 优化 `src/tasks/matcher_bytes.rs` `ChainedMatcherBytes::bk_tree_plus_linear`：使用 `Arc<Vec<HashBytes>>` 共享
- [ ] 3.9 优化 `src/tasks/matcher_bytes.rs` `LinearScanMatcherBytes::find_similar`：改为先计算距离再决定是否 clone（先 filter 后 clone）
- [ ] 3.10 优化 `src/tasks/matcher.rs` `HashMatcherFacade::find_similar_dihedral`：8 种变体作为 batch queries 一次性提交 `find_similar_batch(&variants, threshold)`
- [ ] 3.11 优化 `src/tasks/gpu_image_matcher.rs` 过滤阶段：`MatchResultBytes` 改为持有索引 `(usize, u32)` 或 `Arc<HashBytes>`，避免每匹配 clone
- [ ] 3.12 优化 `src/tasks/gpu_matcher.rs` `hash_bytes_to_u32`：缓存最近一次转换结果，避免重复查询场景的重复转换
- [ ] 3.13 运行 `cargo test` 验证编译和测试通过

## P4: 抽象边界守护（长期目标红线）

- [ ] 4.1 优化 `src/context.rs` `PipelineCache`：`entries`、`generations`、`next_gen` 改为 `RefCell`，`get_or_create` 接受 `&self`
- [ ] 4.2 优化 `src/context.rs` `get_or_create_pipeline`：接受 `&self`，与 `BindGroupCache` 设计一致
- [ ] 4.3 优化 `src/context.rs` `PipelineCache` LRU：改为 O(1) arena + 双向链表，与 `BindGroupCache` 一致
- [ ] 4.4 新增 `src/tasks/gpu_image_matcher.rs` `GpuImageMatcherIndex` 结构体：持有 `Vec<HashBytes>` 和可选 GPU 缓冲区
- [ ] 4.5 新增 `src/tasks/gpu_image_matcher.rs` `index_database` 方法：预计算 database 哈希并缓存
- [ ] 4.6 新增 `src/tasks/gpu_image_matcher.rs` `find_similar_with_index` 方法：复用预计算哈希
- [ ] 4.7 优化 `src/tasks/gpu_image_matcher.rs`：合并 query + database 哈希计算为一次 GPU submit
- [ ] 4.8 优化 `src/tasks/gpu_matcher.rs` `find_nearest_neighbors`：增加分块逻辑，参考 `compute_distance_matrix` 的分块模式
- [ ] 4.9 运行 `cargo test` 验证编译和测试通过

## P5: 逐图路径与细节优化

- [ ] 5.1 优化 `src/tasks/phasher.rs` `compute_gpu_per_image_pipeline`：按尺寸 `(w, h)` 分桶，每组调用 `compute_gpu_batch_pipeline` 批量处理
- [ ] 5.2 优化 `src/tasks/phasher.rs` `preprocess_gpu`：改为接受 `&mut GpuBatchSubmitter`，合并所有图像模糊操作
- [ ] 5.3 优化 `src/tasks/phasher.rs` `process_chunks_serial` Mean/Median 分支：统一使用 `GpuBatchSubmitter` + `merge_gpu_buffers_batch`
- [ ] 5.4 优化 `src/tasks/phasher_util.rs` `merge_gpu_buffers`：将 copy 命令编码到后续 encoder（标记非 batch 版本为 `#[deprecated]`）
- [ ] 5.5 优化 `src/tasks/phasher.rs` `compute_hash_gpu` Mean/Median 路径：编写独立 GPU 阈值计算着色器（reduce kernel），消除 GPU↔CPU 往返
- [ ] 5.6 优化 `src/pipeline_builder.rs` `execute`：缓存 `PerceptualHasher` 避免重复构造
- [ ] 5.7 优化 `src/buffer_pool.rs` `release_staging`：统一与 `release` 的大缓冲区策略，使 `large_buffer_cache` 配置对 staging pool 同样生效
- [ ] 5.8 优化 `src/buffer_pool.rs` `BufferPoolConfig`：增加 `image_friendly_classes` 选项，针对图像场景增加非 2 倍档位
- [ ] 5.9 优化 `src/tasks/gaussian_blur.rs`：缓存 `(kernel_size, sigma) → kernel_1d` 映射
- [ ] 5.10 优化 `src/tasks/gpu_resize.rs` 分块处理：合并所有分块 dispatch 到同一 encoder
- [ ] 5.11 运行 `cargo test` 验证编译和测试通过

## P6: 缓存数据驱动的性能验证

- [ ] 6.1 新增 `bincode` + `serde` 到 `[dev-dependencies]`
- [ ] 6.2 创建 `tests/common/cache_loader.rs`：实现 Czkawka 缓存文件解析（bincode 反序列化 `Vec<ImagesEntry>`）
- [ ] 6.3 创建 `tests/cache_perf_test.rs`：基于缓存数据的性能验证测试
  - 解析 `cache_similar_images_32_Gradient_Gaussian_100.bin` 提取哈希
  - 构建 BkTreeBytes，验证搜索性能（对比 LinearScanMatcherBytes）
  - 上传缓存哈希到 GPU，验证 GPU 汉明距离矩阵计算性能
  - 比较缓存哈希与 GPU 重新计算的哈希，验证正确性
  - 验证 OOM 降级路径（大图像批量场景）
- [ ] 6.4 创建 `benches/cache_matcher_bench.rs`：缓存数据驱动的匹配器基准测试
  - BkTreeBytes vs LinearScanMatcherBytes 搜索性能对比
  - GPU 汉明距离矩阵 vs CPU 线性扫描性能对比
  - 不同阈值下的搜索性能对比
  - 优化前后性能对比（建立基线）
- [ ] 6.5 运行 `cargo bench` 对比优化前后性能数据
- [ ] 6.6 运行 `cargo clippy` 检查代码质量
- [ ] 6.7 验证所有现有测试通过（含 CPU 降级路径）
- [ ] 6.8 验证 DX12 后端 OOM 场景：大图像批量处理不再崩溃，正确降级到 CPU
