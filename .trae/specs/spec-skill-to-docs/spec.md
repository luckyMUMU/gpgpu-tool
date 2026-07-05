# Spec Skill 文档化规范

## Why

当前项目使用 `/spec` 模式进行 spec 驱动开发，但该工作流程仅存在于 Agent 的系统提示中，缺乏面向人类开发者与新 Agent 的可读文档。将 spec 工作流程沉淀为独立的 skill 文档（放置于 `docs/skill/`），可让：

- 新成员快速理解 spec 驱动开发的完整流程（propose → tasks → checklist → implement → verify）
- Agent 在非 `/spec` 模式下也能参照该 skill 执行规范化的变更管理
- spec 文档结构、命名约定、文件路径有统一参照，避免规范漂移

## What Changes

- 新增 `docs/skill/` 目录
- 新增 `docs/skill/SKILL.md` 文件，沉淀 spec 驱动开发工作流程，包含：
  - 触发条件（When to Invoke）
  - 工作流程六阶段（搜索匹配 → 编写 spec.md → 编写 tasks.md → 编写 checklist.md → 通知审批 → 实施 → 验证）
  - 文件结构约定（`.trae/specs/<change-id>/`）
  - spec.md / tasks.md / checklist.md 三文档的内容结构与示例
  - change-id 命名规范（动词引导、唯一性）
  - 实施阶段规则（TodoWrite 对齐、Sub-Agent 并行、任务勾选）
  - 验证阶段规则（逐项核对、失败回流）

## Impact

- **Affected specs**: 无（本变更为文档新增，不影响已有 spec）
- **Affected code**: 无源码改动
- **Affected docs**: 新增 `docs/skill/SKILL.md`
- **Affected workflow**: 为后续 spec 驱动开发提供可参照的 skill 文档

## ADDED Requirements

### Requirement: Spec Skill 文档存在性

系统 SHALL 在 `docs/skill/SKILL.md` 路径下提供 spec 驱动开发工作流程的 skill 文档。

#### Scenario: 开发者查阅 spec 工作流程

- **WHEN** 开发者打开 `docs/skill/SKILL.md`
- **THEN** 能看到完整的 spec 驱动开发六阶段流程说明
- **AND** 能看到 spec.md / tasks.md / checklist.md 三文档的内容结构示例
- **AND** 能看到 change-id 命名规范与文件路径约定

### Requirement: Spec Skill 文档内容完整性

`docs/skill/SKILL.md` SHALL 包含以下章节：

1. **元信息头**（YAML frontmatter）：`name`、`version`、`updated`、`description`
2. **When to Invoke**：列出触发 spec 模式的场景
3. **Workflow Overview**：六阶段流程总览
4. **File Structure**：`.trae/specs/<change-id>/` 目录结构与三文档职责
5. **Document Templates**：spec.md / tasks.md / checklist.md 的内容结构示例
6. **change-id Naming**：命名规范（动词引导、kebab-case、唯一性）
7. **Implementation Rules**：TodoWrite 对齐、Sub-Agent 并行、任务勾选规则
8. **Verification Rules**：逐项核对、失败回流机制
9. **Anti-patterns**：应避免的常见误用

#### Scenario: 文档章节齐全

- **WHEN** 检查 `docs/skill/SKILL.md` 的章节结构
- **THEN** 上述 9 个章节均存在
- **AND** 每个章节包含可操作的说明，而非空标题

### Requirement: 文档语言与项目约定一致

`docs/skill/SKILL.md` SHALL 使用中文撰写（对齐用户规则 1.1「全中文交流」），代码示例与文件路径使用英文。

#### Scenario: 中文正文与英文路径

- **WHEN** 阅读 `docs/skill/SKILL.md`
- **THEN** 章节标题与说明文字为中文
- **AND** 文件路径（如 `.trae/specs/<change-id>/spec.md`）与代码块保持英文原样
