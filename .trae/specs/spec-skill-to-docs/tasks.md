# Tasks

> 本规范目标单一：在 `docs/skill/` 下创建 spec 驱动开发工作流程的 skill 文档。
> 任务拆分为文档骨架编写、章节内容填充、完整性校验三步，按顺序执行。

## Task 1: 创建 `docs/skill/SKILL.md` 文档骨架与元信息

- [x] 1.1: 创建 `docs/skill/` 目录（通过创建文件隐式建立）
- [x] 1.2: 写入 YAML frontmatter，包含 `name: "spec-driven-development"`、`version: "1.0.0"`、`updated: "2026-07-04"`、`description`（一句话说明 spec 驱动开发的用途）
- [x] 1.3: 写入文档主标题与一段简介，说明 spec 驱动开发的核心价值（先规范后实现，分阶段审批）

## Task 2: 编写「When to Invoke」与「Workflow Overview」章节

- [x] 2.1: 编写「When to Invoke」章节，列出触发 spec 模式的典型场景（新功能、架构变更、跨模块改动、需求不明确等）
- [x] 2.2: 编写「Workflow Overview」章节，以六阶段流程图或列表呈现：搜索匹配 → 编写 spec.md → 编写 tasks.md → 编写 checklist.md → 通知审批 → 实施 → 验证
- [x] 2.3: 在 Overview 中明确每阶段的产出物与退出条件

## Task 3: 编写「File Structure」与「Document Templates」章节

- [x] 3.1: 编写「File Structure」章节，说明 `.trae/specs/<change-id>/` 目录结构，三文档（spec.md / tasks.md / checklist.md）的职责分工
- [x] 3.2: 编写「Document Templates」章节，提供 spec.md 的内容结构示例（Why / What Changes / Impact / ADDED Requirements / MODIFIED Requirements / REMOVED Requirements）
- [x] 3.3: 在同一章节提供 tasks.md 的内容结构示例（有序任务列表 + 子任务 + Task Dependencies）
- [x] 3.4: 在同一章节提供 checklist.md 的内容结构示例（按 Phase 分组的可勾选检查点）

## Task 4: 编写「change-id Naming」「Implementation Rules」「Verification Rules」「Anti-patterns」章节

- [x] 4.1: 编写「change-id Naming」章节，规范：动词引导、kebab-case、唯一性、与用户意图关键词匹配
- [x] 4.2: 编写「Implementation Rules」章节，包含：TodoWrite 与 tasks.md 对齐、Sub-Agent 独占实施、无依赖任务可并行、完成后勾选 tasks.md 复选框
- [x] 4.3: 编写「Verification Rules」章节，包含：逐项核对 checklist.md、失败时新增 tasks.md 修复任务并回流实施、全部通过后返回最终响应
- [x] 4.4: 编写「Anti-patterns」章节，列出常见误用（如 proposal 阶段写代码、跳过 NotifyUser 直接实施、未勾选完成任务、删除 spec 文档等）

## Task 5: 完整性校验

- [x] 5.1: 通读 `docs/skill/SKILL.md`，验证 9 个章节（元信息头 + 8 个正文章节）齐全
- [x] 5.2: 验证正文为中文、文件路径与代码块为英文
- [x] 5.3: 验证三文档模板示例可被直接参照套用

# Task Dependencies

- [Task 2] depends on [Task 1]（章节建立在骨架之上）
- [Task 3] depends on [Task 1]
- [Task 4] depends on [Task 1]
- [Task 5] depends on [Task 2, Task 3, Task 4]（校验需所有章节就绪）
- Task 2 / Task 3 / Task 4 可并行编写（均依赖 Task 1，相互独立）

# 验证策略

- 每个任务完成后，Read `docs/skill/SKILL.md` 确认对应章节存在且内容完整
- Task 5 完成后，整体通读一遍，确认无遗漏章节、无中英文混用问题
