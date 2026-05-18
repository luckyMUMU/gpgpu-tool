use wgpu::Limits;

/// 分批元数据，描述一个 GPU 批次的偏移、大小和 dispatch 维度。
#[derive(Debug, Clone)]
pub struct BatchInfo {
    pub offset: u64,
    pub size: u64,
    pub dispatch_x: u32,
}

/// 通用分批工具，根据硬件 limits 动态计算分批策略。
///
/// 将超出缓冲区限制的输入拆分为多个批次，每个批次可独立提交 GPU 处理。
pub struct Chunker {
    max_storage_buffer_size: u64,
    max_workgroups_per_dim: u32,
}

impl Chunker {
    /// 根据 wgpu Limits 创建 Chunker。
    pub fn new(limits: &Limits) -> Self {
        Self {
            max_storage_buffer_size: limits.max_storage_buffer_binding_size as u64,
            max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// 计算分批策略，返回每个批次的元数据。
    pub fn compute_batches(
        &self,
        total_size: u64,
        element_size: u64,
        workgroup_size: u32,
    ) -> Vec<BatchInfo> {
        if total_size == 0 {
            return vec![];
        }

        let element_count = total_size.div_ceil(element_size);
        let max_elements_per_batch = self.max_storage_buffer_size / element_size;
        let max_elements_per_dispatch = self.max_workgroups_per_dim as u64 * workgroup_size as u64;

        let elements_per_batch = max_elements_per_batch.min(max_elements_per_dispatch);
        if elements_per_batch == 0 {
            return vec![];
        }

        let batch_count = element_count.div_ceil(elements_per_batch);
        let mut batches = Vec::with_capacity(batch_count as usize);

        for i in 0..batch_count {
            let start = i * elements_per_batch;
            let end = (start + elements_per_batch).min(element_count);
            let count = end - start;

            batches.push(BatchInfo {
                offset: start * element_size,
                size: count * element_size,
                dispatch_x: (count as u32).div_ceil(workgroup_size),
            });
        }

        batches
    }
}
