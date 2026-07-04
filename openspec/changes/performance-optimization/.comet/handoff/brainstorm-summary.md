# Brainstorm Summary

- Change: performance-optimization
- Date: 2026-06-16

## 确认的技术方案

### 整体策略：方案 C — 能力层优先 + 业务层跟进

- **阶段 1（能力层重构）**：P0 显存安全 + P1 GPU 同步 + P4 抽象边界
- **阶段 2（业务层优化）**：P2 着色器 + P3 内存管理 + P5 逐图路径 + P6 验证

### 关键决策点

1. **P0 `BufferPool::acquire` 签名**：直接改签名为 `Result<Buffer, GpuError>`，同步修改约 15 处调用点
2. **P2 着色器优化范围**：全部 5 项实施（resize SAT + block_hash 预计算 + PDQ cos 表 + convolution LDS + hamming 并行归约）
3. **P4 Uniform 管理**：下沉到 `GpuContext`，新增 `UniformPool`，统一管理 Uniform 缓冲区复用

### 阶段 1 架构

- **P0 四层防御**：UncapturedErrorHandler 注册 + acquire 预检查 + compute_phash 分批 + OOM → CPU 降级
- **P1 连续批量提交**：wait_all 保留引用 + 解放 DualEncoderSubmitter + 跨 chunk 复用 submitter
- **P4 抽象边界**：UniformPool 下沉 GpuContext + PipelineCache RefCell 化 + LRU O(1) arena

### 阶段 2 架构

- **P2 着色器**：resize 积分图（仅大比例下采样）+ block_hash 预计算 + PDQ cos 表+裁剪 + convolution Full2D LDS + hamming 并行归约
- **P3 内存**：dihedral 64/256-bit 位操作 + 距离矩阵扁平化 + Arc 共享 + 先 filter 后 clone
- **P5 逐图路径**：尺寸分组 + 预处理合并 + Mean/Median GPU 阈值计算
- **P4 业务层**：index_database 预计算 + dihedral 批量查询

## 关键取舍与风险

1. **acquire breaking change**：minor 版本升级，编译期发现遗漏调用点
2. **OOM 后 GPU 状态损坏**：降级为终态，进程重启恢复
3. **UniformPool 增加抽象**：符合长期目标，消除业务层绕过封装
4. **DualEncoderSubmitter 默认启用**：保留 feature flag 回退
5. **resize 积分图显存占用**：仅大比例下采样启用，结合 P0 分批保护
6. **hamming 并行归约后端差异**：保留串行路径 fallback
7. **dihedral 位操作正确性**：exhaustive 测试对比 Vec<bool> 结果
8. **Mean/Median GPU 中位数**：均值用 reduce，中位数用排序网络或保持 CPU 预计算

## 测试策略

- **单元测试**：每层防御/每个着色器优化独立测试
- **集成测试**：DX12 后端大图像批量 OOM 降级验证
- **回归测试**：现有 23 个测试文件全部通过
- **基准测试**：Czkawka 缓存数据 20000 条 1024-bit 哈希性能验证
- **正确性测试**：dihedral 位操作 exhaustive 对比，距离矩阵扁平化对比

## Spec Patch

无。proposal/design/tasks 已覆盖所有验收场景，无需回写 delta spec。
