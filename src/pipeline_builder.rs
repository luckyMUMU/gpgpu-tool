use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::hash_common::HashSize;
use crate::tasks::phasher::{HashAlgorithm, PerceptualHasher};

/// 声明式管线步骤，描述 GPU 处理流水线中的单个操作。
#[derive(Debug, Clone)]
pub enum GpuPipelineStep {
    /// 高斯模糊预处理（sigma: 标准差，kernel_size: 核大小，须为正奇数）。
    Blur { sigma: f32, kernel_size: u32 },
    /// 图像缩放到目标尺寸（width × height）。
    Resize { width: u32, height: u32 },
    /// 感知哈希计算（algorithm: 哈希算法，hash_size: 网格尺寸）。
    Hash { algorithm: HashAlgorithm, hash_size: HashSize },
}

/// 声明式管线构建器，支持链式声明 GPU 处理步骤。
///
/// `GpuPipelineBuilder` 是 [`PerceptualHasher`] 的高层封装，
/// 提供更简洁的声明式 API 来构建 GPU 处理流水线。
/// 内部自动推导零拷贝传递路径：连续 GPU 步骤间传递 `GpuBuffer`，
/// GPU 不可用时自动降级到 CPU。
///
/// # 步骤顺序约束
///
/// - `Hash` 必须是最后一步
/// - `Blur` 必须在 `Resize` 之前
/// - 每种步骤最多出现一次
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, GpuPipelineBuilder, HashSize};
/// use gpgpu_tool::tasks::phasher::HashAlgorithm;
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let images = vec![vec![128u8; 256 * 256]];
/// let widths = vec![256u32];
/// let heights = vec![256u32];
///
/// let hashes = GpuPipelineBuilder::new()
///     .blur(1.0, 5)
///     .resize(8, 8)
///     .hash(HashAlgorithm::Mean, HashSize::default())
///     .execute(&mut ctx, &images, &widths, &heights)
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GpuPipelineBuilder {
    steps: Vec<GpuPipelineStep>,
}

impl GpuPipelineBuilder {
    /// 创建空管线。
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// 添加高斯模糊步骤。
    pub fn blur(mut self, sigma: f32, kernel_size: u32) -> Self {
        self.steps.push(GpuPipelineStep::Blur { sigma, kernel_size });
        self
    }

    /// 添加缩放步骤。
    pub fn resize(mut self, width: u32, height: u32) -> Self {
        self.steps.push(GpuPipelineStep::Resize { width, height });
        self
    }

    /// 添加哈希计算步骤。
    pub fn hash(mut self, algorithm: HashAlgorithm, hash_size: HashSize) -> Self {
        self.steps.push(GpuPipelineStep::Hash { algorithm, hash_size });
        self
    }

    /// 返回管线步骤列表。
    pub fn steps(&self) -> &[GpuPipelineStep] {
        &self.steps
    }

    /// 执行管线，按步骤顺序处理图像并返回每张图像的哈希值。
    ///
    /// 内部根据步骤列表构建 [`PerceptualHasher`] 配置，
    /// 自动推导零拷贝 GPU 传递路径，GPU 不可用时降级到 CPU。
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文（需 `&mut` 因创建管线时需编译着色器）
    /// - `images` — 灰度像素数据，每个元素为一张图像的宽×高字节
    /// - `widths` — 每张图像的原始宽度
    /// - `heights` — 每张图像的原始高度
    ///
    /// # 返回
    ///
    /// `Vec<Vec<u64>>` — 每张图像对应一个 `Vec<u64>`，包含该图像的哈希字。
    /// 64-bit 哈希返回 1 个 u64，256-bit 哈希返回 4 个 u64，以此类推。
    pub fn execute(
        &self,
        ctx: &mut GpuContext,
        images: &[Vec<u8>],
        widths: &[u32],
        heights: &[u32],
    ) -> Result<Vec<Vec<u64>>, GpuError> {
        self.validate()?;

        let blur_config = self.extract_blur_config();
        let resize_config = self.extract_resize_config();
        let (algorithm, hash_size) = self.extract_hash_config()?;

        if let Some((rw, rh)) = resize_config {
            let (tw, th) = algorithm.target_size_for(hash_size);
            if rw != tw || rh != th {
                return Err(GpuError::InvalidInput(format!(
                    "Resize 尺寸 ({}, {}) 与算法 {:?} 目标尺寸 ({}, {}) 不匹配",
                    rw, rh, algorithm, tw, th
                )));
            }
        }

        let use_gpu_resize = resize_config.is_some() || blur_config.is_some();
        let hasher = PerceptualHasher::with_full_config(
            ctx,
            algorithm,
            use_gpu_resize,
            hash_size,
            [8, 8, 1],
            blur_config.map(|(sigma, _)| sigma),
            blur_config.map(|(_, ks)| ks),
        )?;

        let dimensions: Vec<(u32, u32)> = widths
            .iter()
            .zip(heights.iter())
            .map(|(&w, &h)| (w, h))
            .collect();

        let flat_hashes = hasher.compute(ctx, images, &dimensions)?;

        let u64s_per_image = hash_size.u64s_per_image() as usize;
        let result = flat_hashes
            .chunks(u64s_per_image)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(result)
    }

    fn validate(&self) -> Result<(), GpuError> {
        let hash_positions: Vec<usize> = self
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s, GpuPipelineStep::Hash { .. }))
            .map(|(i, _)| i)
            .collect();

        if hash_positions.is_empty() {
            return Err(GpuError::InvalidInput("管线必须包含 Hash 步骤".to_string()));
        }
        if hash_positions.len() > 1 {
            return Err(GpuError::InvalidInput("管线最多包含一个 Hash 步骤".to_string()));
        }
        if hash_positions[0] != self.steps.len() - 1 {
            return Err(GpuError::InvalidInput("Hash 步骤必须是管线的最后一步".to_string()));
        }

        let blur_count = self.steps.iter().filter(|s| matches!(s, GpuPipelineStep::Blur { .. })).count();
        if blur_count > 1 {
            return Err(GpuError::InvalidInput("管线最多包含一个 Blur 步骤".to_string()));
        }

        let resize_count = self.steps.iter().filter(|s| matches!(s, GpuPipelineStep::Resize { .. })).count();
        if resize_count > 1 {
            return Err(GpuError::InvalidInput("管线最多包含一个 Resize 步骤".to_string()));
        }

        if let Some(blur_pos) = self.steps.iter().position(|s| matches!(s, GpuPipelineStep::Blur { .. })) {
            if let Some(resize_pos) = self.steps.iter().position(|s| matches!(s, GpuPipelineStep::Resize { .. })) {
                if blur_pos > resize_pos {
                    return Err(GpuError::InvalidInput("Blur 步骤必须在 Resize 步骤之前".to_string()));
                }
            }
        }

        Ok(())
    }

    fn extract_blur_config(&self) -> Option<(f32, u32)> {
        self.steps.iter().find_map(|s| match s {
            GpuPipelineStep::Blur { sigma, kernel_size } => Some((*sigma, *kernel_size)),
            _ => None,
        })
    }

    fn extract_resize_config(&self) -> Option<(u32, u32)> {
        self.steps.iter().find_map(|s| match s {
            GpuPipelineStep::Resize { width, height } => Some((*width, *height)),
            _ => None,
        })
    }

    fn extract_hash_config(&self) -> Result<(HashAlgorithm, HashSize), GpuError> {
        self.steps
            .iter()
            .find_map(|s| match s {
                GpuPipelineStep::Hash { algorithm, hash_size } => Some((*algorithm, *hash_size)),
                _ => None,
            })
            .ok_or_else(|| GpuError::InvalidInput("管线必须包含 Hash 步骤".to_string()))
    }
}

impl Default for GpuPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
