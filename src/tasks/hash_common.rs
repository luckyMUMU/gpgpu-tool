use crate::context::GpuContext;
use crate::error::GpuError;

/// 感知哈希公共参数，通过 uniform buffer 传递给 WGSL。
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhashParams {
    /// 图像宽度（像素数）
    pub width: u32,
    /// 图像高度（像素数）
    pub height: u32,
    /// 图像数量
    pub image_count: u32,
    /// 保留对齐
    pub _padding: u32,
}

/// 感知哈希计算器的统一接口。
///
/// 所有感知哈希算法（Mean、Gradient、Block 等）均实现此 trait，
/// 便于上层通过多态方式调用不同算法。
pub trait PerceptualHashComputer {
    /// 批量计算感知哈希。
    ///
    /// # 参数
    /// - `ctx`: GPU 上下文
    /// - `images`: 每幅图像的原始像素字节序列，所有图像尺寸须一致
    ///
    /// # 返回
    /// 每幅图像对应的 64bit 哈希值数组
    fn compute(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError>;
}
