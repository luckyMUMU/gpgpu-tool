# GPU 加速哈希库优化计划

## TL;DR

> **快速摘要**: 修复 Block Hash 56-bit bug + 启用感知哈希测试 + BufferPool 复用优化
> 
> **交付物**:
> - 修复后的 Block Hash WGSL（64-bit 输出）
> - 启用的感知哈希测试（25 个测试从 #[ignore] 恢复）
> - compute_phash() 集成 BufferPool
> - 清理 README 文档
> 
> **预估工作量**: Medium (4-6h)
> **并行执行**: YES - 3 waves
> **关键路径**: Task 1 → Task 3 → Task 4 → Task 5

---

## Context

### 原始需求
基于代码分析发现的 7 个问题，经 Oracle 评估后修正为 5 个真问题 + 1 个遗漏优化。

### 分析摘要
**关键发现**:
- **Median Hash "仅 64 像素"** — 非 bug，8×8 缩放后 64 像素是正确行为
- **Device/Queue clone** — 伪问题，wgpu 内部是 Arc，clone 仅增加引用计数
- **Block Hash 56-bit** — 确认真 bug，8行×7比较=56 bit，应为 64 bit
- **感知哈希测试全部 #[ignore]** — 25 个测试在 6 个文件中，CI 不运行
- **compute_phash() 未用 BufferPool** — SHA-256 用了但 phash 没用

### Oracle 评估
**关键洞察**:
- 动态 binding layout **不建议现在做** — 所有 7 个算法都用 3-binding，无实际需求
- Block Hash 修复是 **breaking change** — hash 输出变化，需 bump minor version
- 去掉 #[ignore] 需要 CI 有 GPU — 建议在测试入口检测 GPU 可用性后 skip

---

## Work Objectives

### 核心目标
修复已确认的算法 bug 和测试覆盖问题，提升代码质量和性能。

### 具体交付物
- [ ] Block Hash WGSL 修复为 64-bit 输出
- [ ] 感知哈希测试启用（移除 #[ignore] + GPU 可用性守卫）
- [ ] compute_phash() 集成 BufferPool
- [ ] README 清理（删除 chunker.rs 引用）
- [ ] CPU 参考实现同步更新

### 完成定义
- [ ] `cargo test` 通过所有测试（包括感知哈希）
- [ ] `cargo clippy` 无警告
- [ ] Block Hash 输出 64-bit（非 56-bit）
- [ ] 感知哈希 GPU/CPU 结果一致

### Must Have
- Block Hash 64-bit 输出
- 感知哈希测试可运行
- GPU/CPU 结果一致性验证

### Must NOT Have (护栏)
- 不做动态 binding layout（当前无需求）
- 不修改 Median Hash 算法（已确认正确）
- 不重构 GpuBatchSubmitter（Device/Queue clone 是伪问题）
- 不添加新的外部依赖

---

## Verification Strategy (强制)

> **零人工干预** — 所有验证由 agent 执行。无例外。

### 测试决策
- **基础设施存在**: YES（已有 Cargo test + Criterion benchmarks）
- **自动化测试**: YES（Tests-after）
- **框架**: cargo test + cargo bench
- **测试后**: 每个任务完成后运行相关测试

### QA 策略
每个任务必须包含 agent 执行的 QA 场景。
证据保存到 `.omo/evidence/task-{N}-{scenario-slug}.{ext}`。

- **算法修复**: 使用 Bash（cargo test）验证正确性
- **性能优化**: 使用 Bash（cargo bench）验证性能
- **文档修复**: 使用 Read 验证内容

---

## Execution Strategy

### 并行执行波次

```
Wave 1 (立即开始 — 独立修复):
├── Task 1: README 清理 [quick]
├── Task 2: 感知哈希测试启用 [quick]

Wave 2 (Wave 1 后 — 算法修复):
├── Task 3: Block Hash WGSL 修复 [deep]
├── Task 4: Block Hash CPU 参考同步 [quick]

Wave 3 (Wave 2 后 — 性能优化):
├── Task 5: compute_phash() BufferPool 集成 [unspecified-high]

Wave FINAL (所有任务后 — 4 个并行审查):
├── Task F1: 计划合规审计 (oracle)
├── Task F2: 代码质量审查 (unspecified-high)
├── Task F3: 实际 QA 测试 (unspecified-high)
├── Task F4: 范围保真度检查 (deep)
-> 展示结果 -> 获取用户确认
```

### 依赖矩阵

| 任务 | 依赖 | 阻塞 | Wave |
|------|------|------|------|
| 1 | 无 | 无 | 1 |
| 2 | 无 | 3, 4, 5 | 1 |
| 3 | 2 | 4 | 2 |
| 4 | 2, 3 | 5 | 2 |
| 5 | 2 | FINAL | 3 |
| F1-F4 | 1-5 | 无 | FINAL |

### Agent 分配摘要

- **Wave 1**: 2 个任务 — T1 → `quick`, T2 → `quick`
- **Wave 2**: 2 个任务 — T3 → `deep`, T4 → `quick`
- **Wave 3**: 1 个任务 — T5 → `unspecified-high`
- **FINAL**: 4 个任务 — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. README 清理

  **What to do**:
  - 删除 README.md 中对 `chunker.rs` 的引用
  - 更新项目结构图，移除不存在的文件

  **Must NOT do**:
  - 不修改其他文档内容
  - 不添加新功能描述

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的文档清理，单文件修改
  - **Skills**: []
    - 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: 无
  - **Blocked By**: 无（可立即开始）

  **References**:

  **Pattern References**:
  - `README.md:37-45` — 项目结构图，包含 chunker.rs 引用

  **Acceptance Criteria**:

  **QA Scenarios (强制)**:

  ```
  Scenario: README 不再引用 chunker.rs
    Tool: Bash (grep)
    Preconditions: README.md 存在
    Steps:
      1. grep -n "chunker" README.md
      2. 验证无匹配结果
    Expected Result: 无输出（chunker 引用已删除）
    Failure Indicators: grep 返回匹配行
    Evidence: .omo/evidence/task-1-readme-chunker-check.txt
  ```

  **Commit**: YES
  - Message: `docs(readme): remove non-existent chunker.rs reference`
  - Files: `README.md`

---

- [ ] 2. 感知哈希测试启用

  **What to do**:
  - 移除所有感知哈希测试文件中的 `#[ignore]` 标记
  - 在每个测试函数开头添加 GPU 可用性检查
  - 如果 GPU 不可用，优雅跳过（而非失败）
  - 涉及文件：mean_hash_test.rs, median_hash_test.rs, gradient_hash_test.rs, block_hash_test.rs, vert_gradient_hash_test.rs, double_gradient_hash_test.rs

  **Must NOT do**:
  - 不修改测试逻辑
  - 不添加新测试用例
  - 不修改 SHA-256 测试（已正常运行）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 批量移除标记 + 添加守卫，模式固定
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3, 4, 5
  - **Blocked By**: 无（可立即开始）

  **References**:

  **Pattern References**:
  - `tests/sha256_test.rs:4-10` — 已有的 GPU 上下文初始化测试（无 #[ignore]）
  - `tests/mean_hash_test.rs:20-30` — 典型的 #[ignore] 测试示例

  **Acceptance Criteria**:

  **QA Scenarios (强制)**:

  ```
  Scenario: 感知哈希测试可运行
    Tool: Bash (cargo test)
    Preconditions: GPU 可用
    Steps:
      1. cargo test --test mean_hash_test -- --include-ignored
      2. 验证测试通过
    Expected Result: 所有测试 PASS
    Failure Indicators: 任何测试 FAIL
    Evidence: .omo/evidence/task-2-mean-hash-test.txt

  Scenario: GPU 不可用时优雅跳过
    Tool: Bash (cargo test)
    Preconditions: 无 GPU 环境（模拟）
    Steps:
      1. cargo test --test mean_hash_test
      2. 验证测试被跳过（而非失败）
    Expected Result: 测试标记为 "ignored" 或 "skipped"
    Failure Indicators: 测试 FAIL
    Evidence: .omo/evidence/task-2-no-gpu-skip.txt
  ```

  **Commit**: YES
  - Message: `test(hash): enable perceptual hash tests with GPU availability guard`
  - Files: `tests/mean_hash_test.rs`, `tests/median_hash_test.rs`, `tests/gradient_hash_test.rs`, `tests/block_hash_test.rs`, `tests/vert_gradient_hash_test.rs`, `tests/double_gradient_hash_test.rs`

---

- [ ] 3. Block Hash WGSL 修复

  **What to do**:
  - 修改 `src/tasks/block_hash.wgsl`，将输出从 56-bit 扩展到 64-bit
  - 方案：增加垂直邻居比较（行间比较）
    - 水平比较：8行 × 7比较 = 56 bit（保持）
    - 垂直比较：7行 × 8比较 = 56 bit（新增）
    - 组合：取前 64 bit（56 水平 + 8 垂直）
  - 更新 hash_low 和 hash_high 的生成逻辑

  **Must NOT do**:
  - 不修改其他 WGSL shader
  - 不改变输入格式（仍然是 u32 数组）
  - 不改变 bind group layout

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要理解 WGSL 语法和位操作，修改算法逻辑
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2（需要测试验证）

  **References**:

  **Pattern References**:
  - `src/tasks/block_hash.wgsl:21-67` — 当前 Block Hash 实现（56-bit）
  - `src/tasks/gradient_hash.wgsl:21-40` — 梯度哈希的位打包模式（参考）
  - `src/tasks/mean_hash.wgsl:29-41` — 均值哈希的 64-bit 输出模式

  **API/Type References**:
  - `src/tasks/hash_common.rs:10-21` — PhashParams 结构体（uniform 参数）

  **Acceptance Criteria**:

  **QA Scenarios (强制)**:

  ```
  Scenario: Block Hash 输出 64-bit
    Tool: Bash (cargo test)
    Preconditions: Task 2 完成（测试已启用）
    Steps:
      1. cargo test --test block_hash_test
      2. 验证所有测试通过
      3. 检查输出 hash 不为 0（非空输出）
    Expected Result: 所有测试 PASS，hash 输出非零
    Failure Indicators: 测试 FAIL 或 hash 全零
    Evidence: .omo/evidence/task-3-block-hash-test.txt

  Scenario: GPU/CPU 结果一致
    Tool: Bash (cargo test)
    Preconditions: Task 4 完成（CPU 参考已同步）
    Steps:
      1. cargo test --test real_image_hash_test -- block
      2. 验证 GPU hash == CPU hash
    Expected Result: 汉明距离为 0
    Failure Indicators: 汉明距离 > 0
    Evidence: .omo/evidence/task-3-gpu-cpu-consistency.txt
  ```

  **Commit**: YES
  - Message: `fix(hash): block hash now outputs 64-bit instead of 56-bit`
  - Files: `src/tasks/block_hash.wgsl`

---

- [ ] 4. Block Hash CPU 参考同步

  **What to do**:
  - 更新 `tests/common/hash_reference.rs` 中的 `block_hash()` 函数
  - 同步更新 `tests/real_image_hash_test.rs` 中的 `cpu_block_hash()` 函数（如果存在）
  - 确保 CPU 参考实现与新的 WGSL 算法一致

  **Must NOT do**:
  - 不修改其他算法的 CPU 参考
  - 不改变函数签名

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 同步更新 CPU 参考，模式固定
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3（需要新的 WGSL 算法）

  **References**:

  **Pattern References**:
  - `tests/common/hash_reference.rs:99-142` — 当前 Block Hash CPU 参考实现
  - `tests/common/hash_reference.rs:54-73` — Gradient Hash 参考（位打包模式）

  **Acceptance Criteria**:

  **QA Scenarios (强制)**:

  ```
  Scenario: CPU 参考与 GPU 结果一致
    Tool: Bash (cargo test)
    Preconditions: Task 3 完成
    Steps:
      1. cargo test --test block_hash_test
      2. cargo test --test real_image_hash_test -- block
      3. 验证所有测试通过
    Expected Result: 所有测试 PASS
    Failure Indicators: 任何测试 FAIL
    Evidence: .omo/evidence/task-4-cpu-reference-sync.txt
  ```

  **Commit**: YES
  - Message: `test(hash): sync block hash CPU reference with WGSL implementation`
  - Files: `tests/common/hash_reference.rs`

---

- [ ] 5. compute_phash() BufferPool 集成

  **What to do**:
  - 修改 `src/tasks/hash_common.rs` 中的 `compute_phash()` 函数
  - 添加 BufferPool 参数，复用 input/output buffer
  - 更新 `PerceptualHashComputer` trait 或 `PHasher` 以持有 BufferPool
  - 确保 buffer 大小变化时正确处理（图像数量变化）

  **Must NOT do**:
  - 不修改 SHA-256 的 BufferPool 使用
  - 不改变 compute_phash() 的公开接口（仅添加可选参数）
  - 不引入新的外部依赖

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解 BufferPool 生命周期和 buffer 复用策略
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 4)
  - **Blocks**: FINAL
  - **Blocked By**: Task 4（需要完整的测试覆盖）

  **References**:

  **Pattern References**:
  - `src/tasks/sha256.rs:140-196` — SHA-256 的 BufferPool 使用模式
  - `src/buffer_pool.rs:42-92` — BufferPool 的 acquire/release API
  - `src/tasks/hash_common.rs:47-110` — 当前 compute_phash() 实现

  **API/Type References**:
  - `src/buffer_pool.rs:18-24` — BufferPool 结构体
  - `src/buffer.rs:8-13` — BufferUsage 枚举

  **Acceptance Criteria**:

  **QA Scenarios (强制)**:

  ```
  Scenario: 感知哈希性能提升
    Tool: Bash (cargo bench)
    Preconditions: BufferPool 已集成
    Steps:
      1. cargo bench --bench mean_hash_bench
      2. 对比优化前后的性能数据
      3. 验证性能提升（或至少不退化）
    Expected Result: 性能提升或持平
    Failure Indicators: 性能显著退化（>10%）
    Evidence: .omo/evidence/task-5-phash-bench.txt

  Scenario: 功能正确性不变
    Tool: Bash (cargo test)
    Preconditions: BufferPool 已集成
    Steps:
      1. cargo test --test mean_hash_test
      2. cargo test --test real_image_hash_test
      3. 验证所有测试通过
    Expected Result: 所有测试 PASS
    Failure Indicators: 任何测试 FAIL
    Evidence: .omo/evidence/task-5-phash-correctness.txt
  ```

  **Commit**: YES
  - Message: `perf(hash): integrate BufferPool for perceptual hash computation`
  - Files: `src/tasks/hash_common.rs`, `src/tasks/phasher.rs`

---

## Final Verification Wave (强制 — 所有实现任务后)

> 4 个审查 agent 并行运行。全部必须 APPROVE。向用户展示结果并获取明确确认。

- [ ] F1. **计划合规审计** — `oracle`
  阅读计划端到端。对每个 "Must Have"：验证实现存在（读取文件、运行命令）。对每个 "Must NOT Have"：搜索代码库中的禁止模式 — 如果找到则拒绝。检查 .omo/evidence/ 中的证据文件。比较交付物与计划。
  输出: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **代码质量审查** — `unspecified-high`
  运行 `cargo clippy` + `cargo test`。审查所有更改文件：检查 `as any`/`@ts-ignore`、空 catch、console.log、注释掉的代码、未使用的导入。检查 AI 过度注释、过度抽象。
  输出: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **实际 QA 测试** — `unspecified-high`
  从干净状态开始。执行每个任务中的每个 QA 场景 — 按照精确步骤执行，捕获证据。测试跨任务集成。保存到 `.omo/evidence/final-qa/`。
  输出: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **范围保真度检查** — `deep`
  对每个任务：读取 "What to do"，读取实际 diff（git log/diff）。验证 1:1 — 规范中的所有内容都已构建（无遗漏），规范之外的内容未构建（无范围蔓延）。检查 "Must NOT do" 合规性。标记未 accounted 的更改。
  输出: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| 任务 | 提交信息 | 文件 |
|------|---------|------|
| 1 | `docs(readme): remove non-existent chunker.rs reference` | README.md |
| 2 | `test(hash): enable perceptual hash tests with GPU availability guard` | tests/*.rs |
| 3 | `fix(hash): block hash now outputs 64-bit instead of 56-bit` | src/tasks/block_hash.wgsl |
| 4 | `test(hash): sync block hash CPU reference with WGSL implementation` | tests/common/hash_reference.rs |
| 5 | `perf(hash): integrate BufferPool for perceptual hash computation` | src/tasks/hash_common.rs, src/tasks/phasher.rs |

---

## Success Criteria

### 验证命令
```bash
cargo test                    # 预期：所有测试 PASS
cargo clippy                  # 预期：无警告
cargo bench --bench block_hash_bench  # 预期：性能数据
```

### 最终检查清单
- [ ] 所有 "Must Have" 已实现
- [ ] 所有 "Must NOT Have" 未被违反
- [ ] Block Hash 输出 64-bit
- [ ] 感知哈希测试可运行
- [ ] GPU/CPU 结果一致
- [ ] 无新增外部依赖
