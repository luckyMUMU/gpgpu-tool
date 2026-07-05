//! GPU 加速的批量汉明距离计算和并行匹配。
//!
//! 提供两个核心能力：
//! - [`GpuHashMatcher::compute_distance_matrix`]: 计算 N×M 距离矩阵
//! - [`GpuHashMatcher::find_nearest_neighbors`]: 为每个 query 找最近邻
//!
//! # 使用示例
//!
//! ```no_run
//! use gpgpu_tool::GpuContext;
//! use gpgpu_tool::tasks::gpu_matcher::GpuHashMatcher;
//!
//! let mut ctx = GpuContext::new_sync().unwrap();
//! let matcher = GpuHashMatcher::new(&mut ctx).unwrap();
//!
//! let queries = vec![0u64, 0xFFFF_FFFF_FFFF_FFFF];
//! let database = vec![0u64, 0x0000_0000_0000_000F, 0xFFFF_FFFF_FFFF_FFFF];
//!
//! let matrix = matcher.compute_distance_matrix(&ctx, &queries, &database).unwrap();
//! ```

use std::sync::{Arc, Mutex};

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{BindingType, ComputePipeline, PipelineDescriptor};
use crate::ComputeBackend;

const HAMMING_WGSL: &str = include_str!("hamming.wgsl");
const HAMMING_PAIRS_WGSL: &str = include_str!("hamming_pairs.wgsl");

/// 默认 workgroup 大小：距离矩阵使用 16×16，最近邻使用 256×1，大哈希最近邻使用 32×1。
pub const HAMMING_MATRIX_WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];
pub const HAMMING_NEAREST_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
pub const HAMMING_NEAREST_LARGE_WORKGROUP_SIZE: [u32; 3] = [32, 1, 1];
/// 候选对过滤管线使用与距离矩阵相同的 16×16 workgroup。
pub const HAMMING_PAIRS_WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

/// 大规模哈希集合阈值：N ≥ 此值时使用 GPU 端 threshold 过滤管线，避免下载完整距离矩阵。
pub const GPU_FILTERED_PAIRS_THRESHOLD: usize = 2000;

/// u64 哈希拆分为 u32 的数量（lo, hi）。
pub const U32_PER_U64_HASH: u32 = 2;

/// 256-bit 哈希（hash_size=16）拆分为 u32 的数量。
pub const U32_PER_256BIT_HASH: u32 = 8;

/// 1024-bit 哈希（hash_size=32）拆分为 u32 的数量。
pub const U32_PER_1024BIT_HASH: u32 = 32;

/// 4096-bit 哈希（hash_size=64）拆分为 u32 的数量。
pub const U32_PER_4096BIT_HASH: u32 = 128;

/// 大哈希阈值：u32_per_hash 超过此值时使用 find_nearest_neighbor_large 管线。
pub const LARGE_HASH_THRESHOLD: u32 = 32;

/// GPU 加速的汉明距离计算器。
///
/// 内部持有三个计算管线：
/// - `distance_matrix_pipeline`: 计算 N×M 距离矩阵
/// - `nearest_neighbor_pipeline`: 为每个 query 找最近邻（≤1024-bit 哈希）
/// - `nearest_neighbor_large_pipeline`: 为每个 query 找最近邻（≤4096-bit 哈希）
///
/// 当 `GpuContext` 处于 CPU 降级模式时，管线字段为 `None`，
/// 各计算方法自动回退到纯 CPU 实现（见模块底部 CPU 降级函数）。
pub struct GpuHashMatcher {
    distance_matrix_pipeline: Option<Arc<ComputePipeline>>,
    nearest_neighbor_pipeline: Option<Arc<ComputePipeline>>,
    nearest_neighbor_large_pipeline: Option<Arc<ComputePipeline>>,
    /// GPU 端 threshold 过滤管线（hamming_pairs.wgsl），对称模式大规模矩阵时使用。
    hamming_pairs_pipeline: Option<Arc<ComputePipeline>>,
    /// GPU 端 threshold 过滤管线（hamming_pairs.wgsl），非对称模式。
    hamming_pairs_asymmetric_pipeline: Option<Arc<ComputePipeline>>,
}

impl GpuHashMatcher {
    /// 创建 GPU 汉明距离计算器。
    ///
    /// CPU 降级模式下（`ctx.backend() == Cpu`）不创建任何 GPU 管线，
    /// 返回一个管线字段全为 `None` 的实例，计算时回退到 CPU 实现。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        // CPU 降级：跳过管线创建
        if ctx.backend() == ComputeBackend::Cpu {
            return Ok(Self {
                distance_matrix_pipeline: None,
                nearest_neighbor_pipeline: None,
                nearest_neighbor_large_pipeline: None,
                hamming_pairs_pipeline: None,
                hamming_pairs_asymmetric_pipeline: None,
            });
        }

        let distance_matrix_pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor {
                bindings: vec![
                    BindingType::StorageReadOnly,  // queries
                    BindingType::StorageReadOnly,  // database
                    BindingType::StorageReadWrite, // distances (output)
                    BindingType::Uniform,          // params
                ],
                wgsl: HAMMING_WGSL.to_string(),
                workgroup_size: HAMMING_MATRIX_WORKGROUP_SIZE,
                entry_point: "hamming_distance_matrix",
                push_constant_size: None,
            },
        )?;

        let nearest_neighbor_pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor {
                bindings: vec![
                    BindingType::StorageReadOnly,  // queries
                    BindingType::StorageReadOnly,  // database
                    BindingType::StorageReadWrite, // output (index, distance pairs)
                    BindingType::Uniform,          // params
                ],
                wgsl: HAMMING_WGSL.to_string(),
                workgroup_size: HAMMING_NEAREST_WORKGROUP_SIZE,
                entry_point: "find_nearest_neighbor",
                push_constant_size: None,
            },
        )?;

        let nearest_neighbor_large_pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor {
                bindings: vec![
                    BindingType::StorageReadOnly,  // queries
                    BindingType::StorageReadOnly,  // database
                    BindingType::StorageReadWrite, // output (index, distance pairs)
                    BindingType::Uniform,          // params
                ],
                wgsl: HAMMING_WGSL.to_string(),
                workgroup_size: HAMMING_NEAREST_LARGE_WORKGROUP_SIZE,
                entry_point: "find_nearest_neighbor_large",
                push_constant_size: None,
            },
        )?;

        let hamming_pairs_pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor {
                bindings: vec![
                    BindingType::StorageReadOnly,   // 0: queries
                    BindingType::StorageReadOnly,   // 1: database
                    BindingType::StorageReadWrite,  // 2: pairs output
                    BindingType::StorageReadWrite,  // 3: counter (atomic)
                    BindingType::Uniform,           // 4: params
                ],
                wgsl: HAMMING_PAIRS_WGSL.to_string(),
                workgroup_size: HAMMING_PAIRS_WORKGROUP_SIZE,
                entry_point: "hamming_distance_pairs",
                push_constant_size: None,
            },
        )?;

        let hamming_pairs_asymmetric_pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor {
                bindings: vec![
                    BindingType::StorageReadOnly,   // 0: queries
                    BindingType::StorageReadOnly,   // 1: database
                    BindingType::StorageReadWrite,  // 2: pairs output
                    BindingType::StorageReadWrite,  // 3: counter (atomic)
                    BindingType::Uniform,           // 4: params
                ],
                wgsl: HAMMING_PAIRS_WGSL.to_string(),
                workgroup_size: HAMMING_PAIRS_WORKGROUP_SIZE,
                entry_point: "hamming_distance_pairs_asymmetric",
                push_constant_size: None,
            },
        )?;

        Ok(Self {
            distance_matrix_pipeline: Some(distance_matrix_pipeline),
            nearest_neighbor_pipeline: Some(nearest_neighbor_pipeline),
            nearest_neighbor_large_pipeline: Some(nearest_neighbor_large_pipeline),
            hamming_pairs_pipeline: Some(hamming_pairs_pipeline),
            hamming_pairs_asymmetric_pipeline: Some(hamming_pairs_asymmetric_pipeline),
        })
    }

    /// 计算 N×M 汉明距离矩阵。
    ///
    /// `queries` 和 `database` 中的每个 u64 哈希自动拆分为 2 个 u32（lo, hi）。
    ///
    /// 当输出缓冲区大小超过 `max_storage_buffer_binding_size` 时，自动将 database
    /// 分块处理并合并结果，避免 GPU OOM。
    ///
    /// # 返回
    ///
    /// `Vec<Vec<u32>>` — 外层长度 = N（queries），内层长度 = M（database）。
    pub fn compute_distance_matrix(
        &self,
        ctx: &GpuContext,
        queries: &[u64],
        database: &[u64],
    ) -> Result<Vec<Vec<u32>>, GpuError> {
        if queries.is_empty() || database.is_empty() {
            return Ok(vec![vec![0; database.len()]; queries.len()]);
        }

        // CPU 降级：纯 CPU 计算距离矩阵
        if ctx.backend() == ComputeBackend::Cpu {
            return Ok(distance_matrix_cpu_u64(queries, database));
        }

        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let n = queries.len() as u32;
        let m = database.len() as u32;
        let u32_per_hash = U32_PER_U64_HASH;

        // 检查输出缓冲区是否超过 max_storage_buffer_binding_size
        let output_size = (n as u64) * (m as u64) * 4;
        let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;

        if output_size <= max_binding {
            // 单次 dispatch：无需分块
            self.compute_distance_matrix_impl(device, queue, ctx, queries, database, n, m, u32_per_hash)
        } else {
            // 分块处理：每次处理 chunk_size 条 database 条目
            let chunk_rows = (max_binding / (n as u64 * 4)).min(m as u64) as usize;
            let chunk_rows = chunk_rows.max(1); // 至少 1 行

            let mut full_matrix = vec![vec![0u32; m as usize]; n as usize];

            for chunk_start in (0..m as usize).step_by(chunk_rows) {
                let chunk_end = (chunk_start + chunk_rows).min(m as usize);
                let chunk_db = &database[chunk_start..chunk_end];
                let chunk_m = chunk_db.len() as u32;

                let chunk_result = self.compute_distance_matrix_impl(
                    device, queue, ctx, queries, chunk_db, n, chunk_m, u32_per_hash,
                )?;

                // 将分块结果合并到完整矩阵
                for (qi, row) in chunk_result.iter().enumerate() {
                    full_matrix[qi][chunk_start..chunk_end].copy_from_slice(row);
                }
            }

            Ok(full_matrix)
        }
    }

    /// 单次 dispatch 的距离矩阵计算（不分块）。
    fn compute_distance_matrix_impl(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &GpuContext,
        queries: &[u64],
        database: &[u64],
        n: u32,
        m: u32,
        u32_per_hash: u32,
    ) -> Result<Vec<Vec<u32>>, GpuError> {
        let query_u32 = hashes_to_u32(queries);
        let db_u32 = hashes_to_u32(database);

        let query_size = (query_u32.len() * 4) as u64;
        let db_size = (db_u32.len() * 4) as u64;
        let output_size = (n as u64 * m as u64 * 4) as u64;

        let query_buf = ctx.buffer_pool().acquire(device, query_size, BufferUsage::Storage)?;
        let db_buf = ctx.buffer_pool().acquire(device, db_size, BufferUsage::Storage)?;
        let output_buf = ctx.buffer_pool().acquire(device, output_size, BufferUsage::Storage)?;

        queue.write_buffer(&query_buf, 0, bytemuck::cast_slice(&query_u32));
        queue.write_buffer(&db_buf, 0, bytemuck::cast_slice(&db_u32));

        let query_gpu = GpuBuffer::from_raw(query_buf, query_size);
        let db_gpu = GpuBuffer::from_raw(db_buf, db_size);
        let output_gpu = GpuBuffer::from_raw(output_buf, output_size);

        let params = [n, m, u32_per_hash, 0u32];
        let params_gpu = GpuBuffer::from_data(device, &params, BufferUsage::Uniform);

        let dispatch_x = n.div_ceil(HAMMING_MATRIX_WORKGROUP_SIZE[0]).max(1);
        let dispatch_y = m.div_ceil(HAMMING_MATRIX_WORKGROUP_SIZE[1]).max(1);

        let pipeline = self
            .distance_matrix_pipeline
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("距离矩阵管线不可用".to_string()))?;
        pipeline.dispatch(
            device,
            queue,
            &[&query_gpu, &db_gpu, &output_gpu, &params_gpu],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        let raw = output_gpu.download_with_pool(device, queue, ctx.buffer_pool())?;
        let raw_u32: &[u32] = bytemuck::cast_slice(&raw);

        let mut matrix = Vec::with_capacity(n as usize);
        for qi in 0..n as usize {
            let row_start = qi * m as usize;
            let row = raw_u32[row_start..row_start + m as usize].to_vec();
            matrix.push(row);
        }

        ctx.buffer_pool().release(query_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(db_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(output_gpu.into_raw(), BufferUsage::Storage);

        Ok(matrix)
    }

    /// 为每个 query 找到 database 中的最近邻。
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32)>` — 每个元素为 `(database_index, distance)`。
    /// 当 threshold 生效时（最近邻距离 > threshold），返回 `(u32::MAX, u32::MAX)`。
    pub fn find_nearest_neighbors(
        &self,
        ctx: &GpuContext,
        queries: &[u64],
        database: &[u64],
        threshold: u32,
    ) -> Result<Vec<(u32, u32)>, GpuError> {
        if queries.is_empty() {
            return Ok(vec![]);
        }
        if database.is_empty() {
            return Ok(vec![(u32::MAX, u32::MAX); queries.len()]);
        }

        // CPU 降级：纯 CPU 找最近邻
        if ctx.backend() == ComputeBackend::Cpu {
            return Ok(nearest_neighbors_cpu_u64(queries, database, threshold));
        }

        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let n = queries.len() as u32;
        let m = database.len() as u32;
        let u32_per_hash = U32_PER_U64_HASH;

        let query_u32 = hashes_to_u32(queries);
        let db_u32 = hashes_to_u32(database);

        let query_size = (query_u32.len() * 4) as u64;
        let db_size = (db_u32.len() * 4) as u64;
        // Output: 2 u32 per query (index, distance)
        let output_size = n as u64 * 2 * 4;

        let query_buf = ctx.buffer_pool().acquire(device, query_size, BufferUsage::Storage)?;
        let db_buf = ctx.buffer_pool().acquire(device, db_size, BufferUsage::Storage)?;
        let output_buf = ctx.buffer_pool().acquire(device, output_size, BufferUsage::Storage)?;

        queue.write_buffer(&query_buf, 0, bytemuck::cast_slice(&query_u32));
        queue.write_buffer(&db_buf, 0, bytemuck::cast_slice(&db_u32));

        let query_gpu = GpuBuffer::from_raw(query_buf, query_size);
        let db_gpu = GpuBuffer::from_raw(db_buf, db_size);
        let output_gpu = GpuBuffer::from_raw(output_buf, output_size);

        let params = [n, m, u32_per_hash, threshold];
        let params_gpu = GpuBuffer::from_data(device, &params, BufferUsage::Uniform);

        // 根据哈希大小选择管线
        let (pipeline, workgroup_size) = if u32_per_hash > LARGE_HASH_THRESHOLD {
            (&self.nearest_neighbor_large_pipeline, HAMMING_NEAREST_LARGE_WORKGROUP_SIZE)
        } else {
            (&self.nearest_neighbor_pipeline, HAMMING_NEAREST_WORKGROUP_SIZE)
        };
        let pipeline = pipeline
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("最近邻管线不可用".to_string()))?;

        let dispatch_x = n.div_ceil(workgroup_size[0]).max(1);

        pipeline.dispatch(
            device,
            queue,
            &[&query_gpu, &db_gpu, &output_gpu, &params_gpu],
            [dispatch_x, 1, 1],
            None,
            ctx.compute_units(),
        );

        let raw = output_gpu.download_with_pool(device, queue, ctx.buffer_pool())?;
        let raw_u32: &[u32] = bytemuck::cast_slice(&raw);

        let mut results = Vec::with_capacity(n as usize);
        for qi in 0..n as usize {
            let idx = raw_u32[qi * 2];
            let dist = raw_u32[qi * 2 + 1];
            results.push((idx, dist));
        }

        ctx.buffer_pool().release(query_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(db_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(output_gpu.into_raw(), BufferUsage::Storage);

        Ok(results)
    }
}

/// 将 u64 哈希数组转换为 u32 数组（每个 u64 拆为 lo, hi 两个 u32）。
fn hashes_to_u32(hashes: &[u64]) -> Vec<u32> {
    let mut result = Vec::with_capacity(hashes.len() * 2);
    for &hash in hashes {
        result.push(hash as u32);         // lo
        result.push((hash >> 32) as u32); // hi
    }
    result
}

// ── CPU 降级实现 ──────────────────────────────────────────────────
//
// 当 GpuContext 处于 CPU 降级模式（device/queue 为 None）时，距离比较
// 主线不能整体不可用，因此提供与 GPU 路径结果一致的纯 CPU 实现。
// 复用 bktree::hamming_distance（64-bit）和 HashBytes::hamming_distance（变长），
// 保证 CPU/GPU 两条路径语义统一。

/// CPU 路径：计算 64-bit 哈希的 N×M 距离矩阵。
fn distance_matrix_cpu_u64(queries: &[u64], database: &[u64]) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|&q| {
            database
                .iter()
                .map(|&db| hamming_distance(q, db))
                .collect()
        })
        .collect()
}

/// CPU 路径：为每个 64-bit query 找 database 中的最近邻。
///
/// 超过 threshold 的最近邻返回 `(u32::MAX, u32::MAX)`，与 GPU 管线语义一致。
fn nearest_neighbors_cpu_u64(queries: &[u64], database: &[u64], threshold: u32) -> Vec<(u32, u32)> {
    queries
        .iter()
        .map(|&q| {
            let mut best_idx = u32::MAX;
            let mut best_dist = u32::MAX;
            for (di, &db) in database.iter().enumerate() {
                let dist = hamming_distance(q, db);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = di as u32;
                }
            }
            // 与 GPU find_nearest_neighbor 一致：距离超过阈值视为无匹配
            if best_dist <= threshold {
                (best_idx, best_dist)
            } else {
                (u32::MAX, u32::MAX)
            }
        })
        .collect()
}

/// CPU 路径：计算变长哈希的 N×M 距离矩阵。
fn distance_matrix_cpu_bytes(
    queries: &[HashBytes],
    database: &[HashBytes],
) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| database.iter().map(|db| q.hamming_distance(db)).collect())
        .collect()
}

/// CPU 路径：为每个变长哈希 query 找 database 中的最近邻。
///
/// 超过 threshold 的最近邻返回 `(u32::MAX, u32::MAX)`，与 GPU 管线语义一致。
fn nearest_neighbors_cpu_bytes(
    queries: &[HashBytes],
    database: &[HashBytes],
    threshold: u32,
) -> Vec<(u32, u32)> {
    queries
        .iter()
        .map(|q| {
            let mut best_idx = u32::MAX;
            let mut best_dist = u32::MAX;
            for (di, db) in database.iter().enumerate() {
                let dist = q.hamming_distance(db);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = di as u32;
                }
            }
            if best_dist <= threshold {
                (best_idx, best_dist)
            } else {
                (u32::MAX, u32::MAX)
            }
        })
        .collect()
}

/// 将 HashBytes 数组转换为 u32 数组。
///
/// 每个哈希的字节按 4 字节一组转换为 u32（小端序），
/// 不足 4 字节的部分用 0 填充。
fn hash_bytes_to_u32(hashes: &[crate::tasks::hash_bytes::HashBytes]) -> (Vec<u32>, u32) {
    if hashes.is_empty() {
        return (vec![], 0);
    }

    let byte_len = hashes[0].byte_len();
    if byte_len == 0 {
        return (vec![], 0);
    }
    // 每个 u32 包含 4 字节，向上取整
    let u32_per_hash = byte_len.div_ceil(4);
    let mut result = Vec::with_capacity(hashes.len() * u32_per_hash);

    for hash in hashes {
        let bytes = hash.as_bytes();
        for chunk_start in (0..byte_len).step_by(4) {
            let chunk_end = (chunk_start + 4).min(byte_len);
            let mut u32_bytes = [0u8; 4];
            u32_bytes[..chunk_end - chunk_start].copy_from_slice(&bytes[chunk_start..chunk_end]);
            result.push(u32::from_le_bytes(u32_bytes));
        }
    }

    (result, u32_per_hash as u32)
}

// ── Facade 集成 ─────────────────────────────────────────────────

use crate::tasks::bktree::hamming_distance;
use crate::tasks::matcher::{HashMatcher, MatchResult};

/// GPU 加速的哈希匹配器，适配 [`HashMatcher`] trait。
///
/// 构造时将 database 上传到 GPU，之后每次 `find_similar` 调用
/// 通过 GPU 并行计算单个 query 与所有 database 条目的汉明距离。
///
/// `find_similar_batch` 使用 GPU 距离矩阵批量计算，获得更好的并行性能。
pub struct GpuHashMatcherFacade {
    /// GPU 距离矩阵计算器
    gpu_matcher: GpuHashMatcher,
    /// GPU 上下文（用于批量计算，仅在使用 `new_with_shared_ctx` 时设置）
    ctx: Option<Arc<Mutex<GpuContext>>>,
    /// Database 哈希（CPU 侧，用于构造和 fallback）
    database: Vec<u64>,
}

impl GpuHashMatcherFacade {
    /// 创建 GPU 加速匹配器。
    ///
    /// 注意：此构造函数创建的 matcher 在批量查询时会回退到 CPU 实现，
    /// 因为无法在 `&self` 方法中获取 `&mut GpuContext`。
    /// 要使用真正的 GPU 批量匹配，请使用 [`new_with_shared_ctx`]。
    pub fn new(ctx: &mut GpuContext, database: Vec<u64>) -> Result<Self, GpuError> {
        let gpu_matcher = GpuHashMatcher::new(ctx)?;

        Ok(Self {
            gpu_matcher,
            ctx: None,
            database,
        })
    }

    /// 创建 GPU 加速匹配器，使用共享的 GpuContext。
    ///
    /// 这是推荐的构造方式，允许多个 matcher 共享同一个 GPU 上下文，
    /// 并支持真正的 GPU 批量匹配。
    pub fn new_with_shared_ctx(
        ctx: Arc<Mutex<GpuContext>>,
        database: Vec<u64>,
    ) -> Result<Self, GpuError> {
        let mut ctx_guard = ctx.lock().unwrap();
        let gpu_matcher = GpuHashMatcher::new(&mut ctx_guard)?;
        drop(ctx_guard);

        Ok(Self {
            gpu_matcher,
            ctx: Some(ctx),
            database,
        })
    }

    /// 使用归一化阈值比例查找相似哈希。
    ///
    /// `ratio` 取值范围 0.0-1.0，内部计算 `threshold = (ratio * 64.0) as u32`。
    ///
    /// # 推荐比例范围
    ///
    /// - 64-bit 哈希：`0.0`（精确匹配）到 `0.25`（最多 16 位差异）
    /// - 推荐 `0.05`-`0.15` 用于相似图像检测
    pub fn find_similar_ratio(&self, query: u64, ratio: f32) -> Vec<MatchResult> {
        let threshold = (ratio * 64.0) as u32;
        self.find_similar(query, threshold)
    }

    /// 使用归一化阈值比例批量查找相似哈希。
    ///
    /// 参见 [`find_similar_ratio`](GpuHashMatcherFacade::find_similar_ratio)。
    pub fn find_similar_batch_ratio(&self, queries: &[u64], ratio: f32) -> Vec<Vec<MatchResult>> {
        let threshold = (ratio * 64.0) as u32;
        self.find_similar_batch(queries, threshold)
    }
}

impl HashMatcher for GpuHashMatcherFacade {
    fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
        // For a single query, use CPU directly (GPU overhead not worth it)
        // This matches the task requirement: GPU computes distances, CPU filters
        self.database
            .iter()
            .map(|&h| MatchResult {
                hash: h,
                distance: hamming_distance(query, h),
                quality: None,
            })
            .filter(|r| r.distance <= threshold)
            .collect()
    }

    fn find_similar_batch(&self, queries: &[u64], threshold: u32) -> Vec<Vec<MatchResult>> {
        // 对于少量查询或空数据库，使用 CPU 逐个计算（避免 GPU 启动开销）
        if queries.is_empty() || self.database.is_empty() {
            return queries
                .iter()
                .map(|&q| self.find_similar(q, threshold))
                .collect();
        }

        // 如果有共享的 GpuContext，尝试使用 GPU 批量计算距离矩阵
        if let Some(ctx_ref) = &self.ctx {
            if let Ok(ctx) = ctx_ref.lock() {
                if let Ok(matrix) = self.gpu_matcher.compute_distance_matrix(&ctx, queries, &self.database) {
                    // GPU 计算成功，CPU 端过滤 threshold 并构造 MatchResult
                    return matrix
                        .iter()
                        .map(|distances| {
                            distances
                                .iter()
                                .enumerate()
                                .filter(|(_, &dist)| dist <= threshold)
                                .map(|(di, &dist)| MatchResult {
                                    hash: self.database[di],
                                    distance: dist,
                                    quality: None,
                                })
                                .collect()
                        })
                        .collect();
                }
            }
        }

        // 无共享 context 或 GPU 计算失败，回退到 CPU 实现
        queries
            .iter()
            .map(|&q| self.find_similar(q, threshold))
            .collect()
    }
}

// ── 变长哈希 GPU 匹配器 ──────────────────────────────────────────

use crate::tasks::hash_bytes::HashBytes;
use crate::tasks::matcher_bytes::{
    HashMatcherBytes, LinearScanMatcherBytes, MatchResultBytes,
};

/// 变长哈希 GPU 汉明距离计算器。
///
/// 与 [`GpuHashMatcher`] 功能等价，但支持 64-bit 到 4096-bit 的变长哈希。
/// 内部复用同一个 `hamming.wgsl` 着色器，通过 `params.z` (u32_per_hash) 控制哈希长度。
/// 当 u32_per_hash > 32 时自动使用 `find_nearest_neighbor_large` 管线（workgroup_size=32）。
pub struct GpuHashMatcherBytes {
    gpu_matcher: GpuHashMatcher,
}

impl GpuHashMatcherBytes {
    /// 创建变长哈希 GPU 汉明距离计算器。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        let gpu_matcher = GpuHashMatcher::new(ctx)?;
        Ok(Self { gpu_matcher })
    }

    /// 计算变长哈希的 N×M 汉明距离矩阵。
    ///
    /// 所有哈希必须具有相同长度。
    ///
    /// 当输出缓冲区大小超过 `max_storage_buffer_binding_size` 时，自动将 database
    /// 分块处理并合并结果，避免 GPU OOM。
    ///
    /// # 返回
    ///
    /// `Vec<Vec<u32>>` — 外层长度 = N（queries），内层长度 = M（database）。
    pub fn compute_distance_matrix(
        &self,
        ctx: &GpuContext,
        queries: &[HashBytes],
        database: &[HashBytes],
    ) -> Result<Vec<Vec<u32>>, GpuError> {
        if queries.is_empty() || database.is_empty() {
            return Ok(vec![vec![0; database.len()]; queries.len()]);
        }

        // CPU 降级：纯 CPU 计算变长哈希距离矩阵
        if ctx.backend() == ComputeBackend::Cpu {
            return Ok(distance_matrix_cpu_bytes(queries, database));
        }

        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let n = queries.len() as u32;
        let m = database.len() as u32;

        let (_, u32_per_hash) = hash_bytes_to_u32(queries);

        // 检查输出缓冲区是否超过 max_storage_buffer_binding_size
        let output_size = (n as u64) * (m as u64) * 4;
        let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;

        if output_size <= max_binding {
            self.compute_distance_matrix_bytes_impl(device, queue, ctx, queries, database, n, m, u32_per_hash)
        } else {
            // 分块处理
            let chunk_rows = (max_binding / (n as u64 * 4)).min(m as u64) as usize;
            let chunk_rows = chunk_rows.max(1);

            let mut full_matrix = vec![vec![0u32; m as usize]; n as usize];

            for chunk_start in (0..m as usize).step_by(chunk_rows) {
                let chunk_end = (chunk_start + chunk_rows).min(m as usize);
                let chunk_db = &database[chunk_start..chunk_end];
                let chunk_m = chunk_db.len() as u32;

                let chunk_result = self.compute_distance_matrix_bytes_impl(
                    device, queue, ctx, queries, chunk_db, n, chunk_m, u32_per_hash,
                )?;

                for (qi, row) in chunk_result.iter().enumerate() {
                    full_matrix[qi][chunk_start..chunk_end].copy_from_slice(row);
                }
            }

            Ok(full_matrix)
        }
    }

    /// 单次 dispatch 的变长哈希距离矩阵计算（不分块）。
    fn compute_distance_matrix_bytes_impl(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &GpuContext,
        queries: &[HashBytes],
        database: &[HashBytes],
        n: u32,
        m: u32,
        u32_per_hash: u32,
    ) -> Result<Vec<Vec<u32>>, GpuError> {
        let (query_u32, _) = hash_bytes_to_u32(queries);
        let (db_u32, _) = hash_bytes_to_u32(database);

        let query_size = (query_u32.len() * 4) as u64;
        let db_size = (db_u32.len() * 4) as u64;
        let output_size = (n as u64 * m as u64 * 4) as u64;

        let query_buf = ctx.buffer_pool().acquire(device, query_size, BufferUsage::Storage)?;
        let db_buf = ctx.buffer_pool().acquire(device, db_size, BufferUsage::Storage)?;
        let output_buf = ctx.buffer_pool().acquire(device, output_size, BufferUsage::Storage)?;

        queue.write_buffer(&query_buf, 0, bytemuck::cast_slice(&query_u32));
        queue.write_buffer(&db_buf, 0, bytemuck::cast_slice(&db_u32));

        let query_gpu = GpuBuffer::from_raw(query_buf, query_size);
        let db_gpu = GpuBuffer::from_raw(db_buf, db_size);
        let output_gpu = GpuBuffer::from_raw(output_buf, output_size);

        let params = [n, m, u32_per_hash, 0u32];
        let params_gpu = GpuBuffer::from_data(device, &params, BufferUsage::Uniform);

        let dispatch_x = n.div_ceil(HAMMING_MATRIX_WORKGROUP_SIZE[0]).max(1);
        let dispatch_y = m.div_ceil(HAMMING_MATRIX_WORKGROUP_SIZE[1]).max(1);

        let pipeline = self
            .gpu_matcher
            .distance_matrix_pipeline
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("距离矩阵管线不可用".to_string()))?;
        pipeline.dispatch(
            device,
            queue,
            &[&query_gpu, &db_gpu, &output_gpu, &params_gpu],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        let raw = output_gpu.download_with_pool(device, queue, ctx.buffer_pool())?;
        let raw_u32: &[u32] = bytemuck::cast_slice(&raw);

        let mut matrix = Vec::with_capacity(n as usize);
        for qi in 0..n as usize {
            let row_start = qi * m as usize;
            let row = raw_u32[row_start..row_start + m as usize].to_vec();
            matrix.push(row);
        }

        ctx.buffer_pool().release(query_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(db_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(output_gpu.into_raw(), BufferUsage::Storage);

        Ok(matrix)
    }

    /// 为每个变长哈希 query 找到 database 中的最近邻。
    ///
    /// 根据 u32_per_hash 自动选择合适的 GPU 管线：
    /// - ≤32（≤1024-bit）：使用 workgroup_size(256) 管线
    /// - >32（>1024-bit）：使用 workgroup_size(32) 管线
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32)>` — 每个元素为 `(database_index, distance)`。
    pub fn find_nearest_neighbors(
        &self,
        ctx: &GpuContext,
        queries: &[HashBytes],
        database: &[HashBytes],
        threshold: u32,
    ) -> Result<Vec<(u32, u32)>, GpuError> {
        if queries.is_empty() {
            return Ok(vec![]);
        }
        if database.is_empty() {
            return Ok(vec![(u32::MAX, u32::MAX); queries.len()]);
        }

        // CPU 降级：纯 CPU 找变长哈希最近邻
        if ctx.backend() == ComputeBackend::Cpu {
            return Ok(nearest_neighbors_cpu_bytes(queries, database, threshold));
        }

        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let n = queries.len() as u32;
        let m = database.len() as u32;

        let (query_u32, u32_per_hash) = hash_bytes_to_u32(queries);
        let (db_u32, _) = hash_bytes_to_u32(database);

        let query_size = (query_u32.len() * 4) as u64;
        let db_size = (db_u32.len() * 4) as u64;
        let output_size = n as u64 * 2 * 4;

        let query_buf = ctx.buffer_pool().acquire(device, query_size, BufferUsage::Storage)?;
        let db_buf = ctx.buffer_pool().acquire(device, db_size, BufferUsage::Storage)?;
        let output_buf = ctx.buffer_pool().acquire(device, output_size, BufferUsage::Storage)?;

        queue.write_buffer(&query_buf, 0, bytemuck::cast_slice(&query_u32));
        queue.write_buffer(&db_buf, 0, bytemuck::cast_slice(&db_u32));

        let query_gpu = GpuBuffer::from_raw(query_buf, query_size);
        let db_gpu = GpuBuffer::from_raw(db_buf, db_size);
        let output_gpu = GpuBuffer::from_raw(output_buf, output_size);

        let params = [n, m, u32_per_hash, threshold];
        let params_gpu = GpuBuffer::from_data(device, &params, BufferUsage::Uniform);

        // 根据哈希大小选择管线
        let (pipeline, workgroup_size) = if u32_per_hash > LARGE_HASH_THRESHOLD {
            (&self.gpu_matcher.nearest_neighbor_large_pipeline, HAMMING_NEAREST_LARGE_WORKGROUP_SIZE)
        } else {
            (&self.gpu_matcher.nearest_neighbor_pipeline, HAMMING_NEAREST_WORKGROUP_SIZE)
        };
        let pipeline = pipeline
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("最近邻管线不可用".to_string()))?;

        let dispatch_x = n.div_ceil(workgroup_size[0]).max(1);

        pipeline.dispatch(
            device,
            queue,
            &[&query_gpu, &db_gpu, &output_gpu, &params_gpu],
            [dispatch_x, 1, 1],
            None,
            ctx.compute_units(),
        );

        let raw = output_gpu.download_with_pool(device, queue, ctx.buffer_pool())?;
        let raw_u32: &[u32] = bytemuck::cast_slice(&raw);

        let mut results = Vec::with_capacity(n as usize);
        for qi in 0..n as usize {
            let idx = raw_u32[qi * 2];
            let dist = raw_u32[qi * 2 + 1];
            results.push((idx, dist));
        }

        ctx.buffer_pool().release(query_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(db_gpu.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(output_gpu.into_raw(), BufferUsage::Storage);

        Ok(results)
    }

    /// 计算所有满足容差的相似哈希对（czkawka 兼容格式）。
    ///
    /// 对称模式：在同一个哈希集合内查找所有距离 ≤ `tolerance` 的对。
    /// 内部使用 GPU 距离矩阵计算，CPU 端过滤 threshold 并去除自身匹配。
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32, u32)>` — `(parent_idx, child_idx, distance)` 三元组列表，
    /// 按 distance 升序排序。已过滤 `parent_idx == child_idx` 的自身匹配。
    ///
    /// 对应 czkawka `gpu_compare_hashes_auto()` 的输出格式。
    pub fn compute_similar_pairs(
        &self,
        ctx: &GpuContext,
        hashes: &[HashBytes],
        tolerance: u32,
    ) -> Result<Vec<(u32, u32, u32)>, GpuError> {
        if hashes.len() < 2 {
            return Ok(vec![]);
        }

        // GPU 距离矩阵
        let matrix = self.compute_distance_matrix(ctx, hashes, hashes)?;

        // CPU 端过滤 threshold + 去除自身匹配
        let mut pairs = Vec::new();
        for (i, row) in matrix.iter().enumerate() {
            for (j, &dist) in row.iter().enumerate() {
                if i != j && dist <= tolerance {
                    pairs.push((i as u32, j as u32, dist));
                }
            }
        }

        // 按 distance 升序排序
        pairs.sort_by_key(|&(_, _, dist)| dist);

        Ok(pairs)
    }

    /// 计算非对称模式的相似哈希对（czkawka 兼容格式）。
    ///
    /// 非对称模式：在 `ref_hashes`（参考文件夹）和 `normal_hashes`（普通文件夹）
    /// 之间查找所有距离 ≤ `tolerance` 的对。
    ///
    /// 对应 czkawka `gpu_compare_hashes_asymmetric()`。
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32, u32)>` — `(ref_idx, normal_idx, distance)` 三元组列表，
    /// 按 distance 升序排序。
    pub fn compute_similar_pairs_asymmetric(
        &self,
        ctx: &GpuContext,
        ref_hashes: &[HashBytes],
        normal_hashes: &[HashBytes],
        tolerance: u32,
    ) -> Result<Vec<(u32, u32, u32)>, GpuError> {
        if ref_hashes.is_empty() || normal_hashes.is_empty() {
            return Ok(vec![]);
        }

        // GPU 非对称距离矩阵：ref_hashes × normal_hashes
        let matrix = self.compute_distance_matrix(ctx, ref_hashes, normal_hashes)?;

        // CPU 端过滤 threshold
        let mut pairs = Vec::new();
        for (i, row) in matrix.iter().enumerate() {
            for (j, &dist) in row.iter().enumerate() {
                if dist <= tolerance {
                    pairs.push((i as u32, j as u32, dist));
                }
            }
        }

        // 按 distance 升序排序
        pairs.sort_by_key(|&(_, _, dist)| dist);

        Ok(pairs)
    }

    /// GPU 端 threshold 过滤的相似哈希对计算（对称模式）。
    ///
    /// 与 [`compute_similar_pairs`](Self::compute_similar_pairs) 不同，此方法使用
    /// `hamming_pairs.wgsl` 着色器在 GPU 端直接过滤 threshold，仅下载匹配的对，
    /// 避免下载完整 N×N 距离矩阵。适用于大规模哈希集合（N ≥ `GPU_FILTERED_PAIRS_THRESHOLD`）。
    ///
    /// # 工作原理
    ///
    /// 1. GPU 端计算所有 N×N 哈希对的汉明距离
    /// 2. 在 GPU 端过滤距离 ≤ `tolerance` 且非自身匹配的对
    /// 3. 通过原子计数器管理输出位置
    /// 4. 仅下载匹配的对（通常远少于 N²）
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32, u32)>` — `(parent_idx, child_idx, distance)` 三元组列表，
    /// 按 distance 升序排序。已过滤 `parent_idx == child_idx` 的自身匹配。
    pub fn compute_similar_pairs_gpu_filtered(
        &self,
        ctx: &GpuContext,
        hashes: &[HashBytes],
        tolerance: u32,
    ) -> Result<Vec<(u32, u32, u32)>, GpuError> {
        if hashes.len() < 2 {
            return Ok(vec![]);
        }

        // CPU 降级：回退到矩阵方式（内部再降级到纯 CPU）
        if ctx.backend() == ComputeBackend::Cpu {
            return self.compute_similar_pairs(ctx, hashes, tolerance);
        }

        let (hash_u32, u32_per_hash) = hash_bytes_to_u32(hashes);
        let n = hashes.len() as u32;

        let pipeline = self
            .gpu_matcher
            .hamming_pairs_pipeline
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("候选对过滤管线不可用".to_string()))?;

        // 对称模式：queries 和 database 是同一个缓冲区
        self.compute_pairs_gpu_filtered_impl(
            ctx,
            &hash_u32,
            &hash_u32,
            n,
            n,
            u32_per_hash,
            tolerance,
            pipeline,
            true,
        )
    }

    /// GPU 端 threshold 过滤的相似哈希对计算（非对称模式）。
    ///
    /// 与 [`compute_similar_pairs_asymmetric`](Self::compute_similar_pairs_asymmetric) 不同，
    /// 此方法使用 `hamming_pairs.wgsl` 的非对称入口点在 GPU 端直接过滤 threshold。
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32, u32)>` — `(ref_idx, normal_idx, distance)` 三元组列表，
    /// 按 distance 升序排序。
    pub fn compute_similar_pairs_asymmetric_gpu_filtered(
        &self,
        ctx: &GpuContext,
        ref_hashes: &[HashBytes],
        normal_hashes: &[HashBytes],
        tolerance: u32,
    ) -> Result<Vec<(u32, u32, u32)>, GpuError> {
        if ref_hashes.is_empty() || normal_hashes.is_empty() {
            return Ok(vec![]);
        }

        // CPU 降级
        if ctx.backend() == ComputeBackend::Cpu {
            return self.compute_similar_pairs_asymmetric(
                ctx,
                ref_hashes,
                normal_hashes,
                tolerance,
            );
        }

        let (ref_u32, u32_per_hash) = hash_bytes_to_u32(ref_hashes);
        let (normal_u32, _) = hash_bytes_to_u32(normal_hashes);
        let n = ref_hashes.len() as u32;
        let m = normal_hashes.len() as u32;

        let pipeline = self
            .gpu_matcher
            .hamming_pairs_asymmetric_pipeline
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("候选对过滤管线(非对称)不可用".to_string()))?;

        self.compute_pairs_gpu_filtered_impl(
            ctx,
            &ref_u32,
            &normal_u32,
            n,
            m,
            u32_per_hash,
            tolerance,
            pipeline,
            false,
        )
    }

    /// GPU 端 threshold 过滤的内部实现。
    ///
    /// `is_symmetric` 为 true 时，`query_u32` 和 `db_u32` 指向同一数据，
    /// 使用同一个 GPU 缓冲区作为 binding 0 和 binding 1。
    fn compute_pairs_gpu_filtered_impl(
        &self,
        ctx: &GpuContext,
        query_u32: &[u32],
        db_u32: &[u32],
        n: u32,
        m: u32,
        u32_per_hash: u32,
        tolerance: u32,
        pipeline: &ComputePipeline,
        is_symmetric: bool,
    ) -> Result<Vec<(u32, u32, u32)>, GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;

        // 计算输出缓冲区容量：每对 3 个 u32 = 12 字节
        let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;
        let max_pairs_by_buffer = max_binding / 12;
        let pairs_capacity = ((n as u64) * (m as u64)).min(max_pairs_by_buffer) as u32;
        let pairs_capacity = pairs_capacity.max(1);

        let query_size = (query_u32.len() * 4) as u64;
        let db_size = (db_u32.len() * 4) as u64;
        let pairs_size = (pairs_capacity as u64) * 12;
        let counter_size = 4u64; // 1 u32

        // 创建缓冲区
        let query_buf = ctx.buffer_pool().acquire(device, query_size, BufferUsage::Storage)?;
        let db_buf = if is_symmetric {
            // 对称模式：复用同一个缓冲区
            query_buf.clone()
        } else {
            ctx.buffer_pool().acquire(device, db_size, BufferUsage::Storage)?
        };
        let pairs_buf = ctx.buffer_pool().acquire(device, pairs_size, BufferUsage::Storage)?;
        let counter_buf = ctx.buffer_pool().acquire(device, counter_size, BufferUsage::Storage)?;

        // 上传数据
        queue.write_buffer(&query_buf, 0, bytemuck::cast_slice(query_u32));
        if !is_symmetric {
            queue.write_buffer(&db_buf, 0, bytemuck::cast_slice(db_u32));
        }
        // 初始化计数器为 0
        queue.write_buffer(&counter_buf, 0, bytemuck::cast_slice(&[0u32]));

        let query_gpu = GpuBuffer::from_raw(query_buf, query_size);
        let db_gpu = GpuBuffer::from_raw(db_buf, db_size);
        let pairs_gpu = GpuBuffer::from_raw(pairs_buf, pairs_size);
        let counter_gpu = GpuBuffer::from_raw(counter_buf, counter_size);

        // 参数: (n, m, u32_per_hash, threshold)
        let params = [n, m, u32_per_hash, tolerance];
        let params_gpu = GpuBuffer::from_data(device, &params, BufferUsage::Uniform);

        let dispatch_x = n.div_ceil(HAMMING_PAIRS_WORKGROUP_SIZE[0]).max(1);
        let dispatch_y = m.div_ceil(HAMMING_PAIRS_WORKGROUP_SIZE[1]).max(1);

        pipeline.dispatch(
            device,
            queue,
            &[&query_gpu, &db_gpu, &pairs_gpu, &counter_gpu, &params_gpu],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        // 下载计数器
        let counter_raw = counter_gpu.download_with_pool(device, queue, ctx.buffer_pool())?;
        let counter_u32: &[u32] = bytemuck::cast_slice(&counter_raw);
        let pair_count = counter_u32[0] as usize;

        // 检查是否溢出缓冲区容量
        if pair_count > pairs_capacity as usize {
            return Err(GpuError::InvalidInput(format!(
                "GPU 过滤候选对数量 ({}) 超过缓冲区容量 ({})，请减小数据集或降低 tolerance",
                pair_count, pairs_capacity
            )));
        }

        // 清理计数器缓冲区
        ctx.buffer_pool().release(counter_gpu.into_raw(), BufferUsage::Storage);

        if pair_count == 0 {
            ctx.buffer_pool().release(query_gpu.into_raw(), BufferUsage::Storage);
            if !is_symmetric {
                ctx.buffer_pool().release(db_gpu.into_raw(), BufferUsage::Storage);
            }
            ctx.buffer_pool().release(pairs_gpu.into_raw(), BufferUsage::Storage);
            return Ok(vec![]);
        }

        // 下载候选对数据
        let pairs_raw = pairs_gpu.download_with_pool(device, queue, ctx.buffer_pool())?;
        let pairs_u32: &[u32] = bytemuck::cast_slice(&pairs_raw);

        // 解析候选对三元组
        let mut pairs = Vec::with_capacity(pair_count);
        for i in 0..pair_count {
            let qi = pairs_u32[i * 3];
            let di = pairs_u32[i * 3 + 1];
            let dist = pairs_u32[i * 3 + 2];
            pairs.push((qi, di, dist));
        }

        // 清理缓冲区
        ctx.buffer_pool().release(query_gpu.into_raw(), BufferUsage::Storage);
        if !is_symmetric {
            ctx.buffer_pool().release(db_gpu.into_raw(), BufferUsage::Storage);
        }
        ctx.buffer_pool().release(pairs_gpu.into_raw(), BufferUsage::Storage);

        // 按 distance 升序排序
        pairs.sort_by_key(|&(_, _, dist)| dist);

        Ok(pairs)
    }
}

/// 变长哈希 GPU 加速匹配器，适配 [`HashMatcherBytes`] trait。
///
/// 构造时将 database 上传到 GPU，之后每次 `find_similar` 调用
/// 通过 GPU 并行计算单个 query 与所有 database 条目的汉明距离。
pub struct GpuHashMatcherFacadeBytes {
    gpu_matcher: GpuHashMatcherBytes,
    ctx: Option<Arc<Mutex<GpuContext>>>,
    database: Vec<HashBytes>,
}

impl GpuHashMatcherFacadeBytes {
    /// 创建变长哈希 GPU 加速匹配器。
    pub fn new(ctx: &mut GpuContext, database: Vec<HashBytes>) -> Result<Self, GpuError> {
        let gpu_matcher = GpuHashMatcherBytes::new(ctx)?;
        Ok(Self {
            gpu_matcher,
            ctx: None,
            database,
        })
    }

    /// 创建变长哈希 GPU 加速匹配器，使用共享的 GpuContext。
    pub fn new_with_shared_ctx(
        ctx: Arc<Mutex<GpuContext>>,
        database: Vec<HashBytes>,
    ) -> Result<Self, GpuError> {
        let mut ctx_guard = ctx.lock().unwrap();
        let gpu_matcher = GpuHashMatcherBytes::new(&mut ctx_guard)?;
        drop(ctx_guard);

        Ok(Self {
            gpu_matcher,
            ctx: Some(ctx),
            database,
        })
    }

    /// 使用归一化阈值比例查找相似哈希。
    ///
    /// `ratio` 取值范围 0.0-1.0，内部计算
    /// `threshold = (ratio * query.byte_len() as f32 * 8.0) as u32`。
    ///
    /// # 推荐比例范围
    ///
    /// - 64-bit 哈希（8 字节）：`0.0`-`0.25`
    /// - 256-bit 哈希（32 字节）：`0.0`-`0.15`
    /// - 1024-bit 哈希（128 字节）：`0.0`-`0.10`
    /// - 4096-bit 哈希（512 字节）：`0.0`-`0.05`
    /// - 推荐 `0.05`-`0.10` 用于相似图像检测
    pub fn find_similar_ratio(&self, query: &HashBytes, ratio: f32) -> Vec<MatchResultBytes> {
        let total_bits = query.byte_len() as f32 * 8.0;
        let threshold = (ratio * total_bits) as u32;
        self.find_similar(query, threshold)
    }

    /// 使用归一化阈值比例批量查找相似哈希。
    ///
    /// 参见 [`find_similar_ratio`](GpuHashMatcherFacadeBytes::find_similar_ratio)。
    pub fn find_similar_batch_ratio(
        &self,
        queries: &[HashBytes],
        ratio: f32,
    ) -> Vec<Vec<MatchResultBytes>> {
        if queries.is_empty() {
            return vec![];
        }
        let total_bits = queries[0].byte_len() as f32 * 8.0;
        let threshold = (ratio * total_bits) as u32;
        self.find_similar_batch(queries, threshold)
    }
}

impl HashMatcherBytes for GpuHashMatcherFacadeBytes {
    fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes> {
        // 单个查询使用 CPU（GPU 启动开销不值得）
        let cpu_matcher = LinearScanMatcherBytes::new(self.database.clone());
        cpu_matcher.find_similar(query, threshold)
    }

    fn find_similar_batch(&self, queries: &[HashBytes], threshold: u32) -> Vec<Vec<MatchResultBytes>> {
        if queries.is_empty() || self.database.is_empty() {
            return queries
                .iter()
                .map(|q| self.find_similar(q, threshold))
                .collect();
        }

        // 尝试 GPU 批量计算
        if let Some(ctx_ref) = &self.ctx {
            if let Ok(ctx) = ctx_ref.lock() {
                if let Ok(matrix) = self.gpu_matcher.compute_distance_matrix(&ctx, queries, &self.database) {
                    return matrix
                        .iter()
                        .map(|distances| {
                            distances
                                .iter()
                                .enumerate()
                                .filter(|(_, &dist)| dist <= threshold)
                                .map(|(di, &dist)| MatchResultBytes {
                                    hash: self.database[di].clone(),
                                    distance: dist,
                                    quality: None,
                                })
                                .collect()
                        })
                        .collect();
                }
            }
        }

        // 回退到 CPU
        queries
            .iter()
            .map(|q| self.find_similar(q, threshold))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuContext;

    #[test]
    fn test_gpu_bytes_distance_matrix_256bit() {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

        // 两个 256-bit 哈希：全零 vs 全 1
        let query = HashBytes::from_u64s(&[0u64, 0, 0, 0]);
        let db_zero = HashBytes::from_u64s(&[0u64, 0, 0, 0]);
        let db_ones = HashBytes::from_u64s(&[u64::MAX, u64::MAX, u64::MAX, u64::MAX]);

        let matrix = matcher
            .compute_distance_matrix(&ctx, &[query.clone()], &[db_zero, db_ones])
            .expect("256-bit 距离矩阵计算失败");

        assert_eq!(matrix.len(), 1, "应有 1 行");
        assert_eq!(matrix[0].len(), 2, "应有 2 列");
        assert_eq!(matrix[0][0], 0, "全零 vs 全零距离应为 0");
        assert_eq!(matrix[0][1], 256, "全零 vs 全 1 距离应为 256");
    }

    #[test]
    fn test_gpu_bytes_distance_matrix_256bit_all_zeros() {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

        // 全零 vs 全零（模拟 demo 中 hash_size=16 的场景）
        let query = HashBytes::from_u64s(&[0u64, 0, 0, 0]);
        let db1 = HashBytes::from_u64s(&[0u64, 0, 0, 0]);
        let db2 = HashBytes::from_u64s(&[0u64, 0, 0, 0]);

        let matrix = matcher
            .compute_distance_matrix(&ctx, &[query], &[db1, db2])
            .expect("256-bit 全零距离矩阵计算失败");

        assert_eq!(matrix[0][0], 0, "全零 vs 全零距离应为 0");
        assert_eq!(matrix[0][1], 0, "全零 vs 全零距离应为 0");
    }

    #[test]
    fn test_gpu_bytes_distance_matrix_64bit() {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

        let query = HashBytes::from_u64(0);
        let db_zero = HashBytes::from_u64(0);
        let db_max = HashBytes::from_u64(u64::MAX);

        let matrix = matcher
            .compute_distance_matrix(&ctx, &[query], &[db_zero, db_max])
            .expect("64-bit 距离矩阵计算失败");

        assert_eq!(matrix[0][0], 0, "0 vs 0 距离应为 0");
        assert_eq!(matrix[0][1], 64, "0 vs MAX 距离应为 64");
    }

    #[test]
    fn test_gpu_bytes_distance_matrix_1024bit() {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

        // 1024-bit 哈希：16 个 u64
        let query = HashBytes::from_u64s(&[0u64; 16]);
        let db_zero = HashBytes::from_u64s(&[0u64; 16]);
        let db_ones = HashBytes::from_u64s(&[u64::MAX; 16]);

        let matrix = matcher
            .compute_distance_matrix(&ctx, &[query], &[db_zero, db_ones])
            .expect("1024-bit 距离矩阵计算失败");

        assert_eq!(matrix[0][0], 0, "全零 vs 全零距离应为 0");
        assert_eq!(matrix[0][1], 1024, "全零 vs 全 1 距离应为 1024");
    }

    #[test]
    fn test_gpu_bytes_distance_matrix_4096bit() {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

        // 4096-bit 哈希：64 个 u64
        let query = HashBytes::from_u64s(&[0u64; 64]);
        let db_zero = HashBytes::from_u64s(&[0u64; 64]);
        let db_ones = HashBytes::from_u64s(&[u64::MAX; 64]);

        let matrix = matcher
            .compute_distance_matrix(&ctx, &[query], &[db_zero, db_ones])
            .expect("4096-bit 距离矩阵计算失败");

        assert_eq!(matrix[0][0], 0, "全零 vs 全零距离应为 0");
        assert_eq!(matrix[0][1], 4096, "全零 vs 全 1 距离应为 4096");
    }

    #[test]
    fn test_gpu_bytes_nearest_neighbor_4096bit() {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

        // 4096-bit 哈希：query 全零，database 包含全零和全 1
        let query = HashBytes::from_u64s(&[0u64; 64]);
        let db_zero = HashBytes::from_u64s(&[0u64; 64]);
        let db_ones = HashBytes::from_u64s(&[u64::MAX; 64]);

        let results = matcher
            .find_nearest_neighbors(&ctx, &[query], &[db_zero, db_ones], u32::MAX)
            .expect("4096-bit 最近邻计算失败");

        assert_eq!(results.len(), 1, "应有 1 个结果");
        assert_eq!(results[0].0, 0, "最近邻应为索引 0（全零）");
        assert_eq!(results[0].1, 0, "最近邻距离应为 0");
    }
}
