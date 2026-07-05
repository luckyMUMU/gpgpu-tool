---
name: "spec-driven-development"
version: "1.0.0"
updated: "2026-07-04"
description: "Spec 驱动开发工作流程。当需要在实施前先规范变更范围、拆解任务、定义验证检查点时调用。覆盖 propose → tasks → checklist → implement → verify 全生命周期。"
---

# spec-driven-development — Spec 驱动开发工作流程

核心价值：先规范后实现，分阶段审批，确保变更可追溯、可验证。通过 spec.md / tasks.md / checklist.md 三文档协同，将"做什么 / 怎么做 / 如何验证"分离，使变更范围在实施前被充分澄清，实施过程被任务清单约束，实施结果被检查点逐项验证。

## When to Invoke

下列场景应触发 spec 模式：

- **新增功能或能力**：需要明确需求边界与验收标准的新功能开发
- **架构变更或跨模块改动**：涉及多个模块、影响能力层抽象的改动
- **需求不明确，需先澄清再实施**：用户描述含糊或边界模糊时，先用 spec 锚定意图
- **涉及 BREAKING 变更**：会破坏现有 API、行为或兼容性的改动
- **多步骤复杂任务（≥3 个可执行单元）**：需要任务拆解与依赖管理的复杂工作
- **用户显式调用 /spec 命令**：用户明确要求进入 spec 驱动流程

## Workflow Overview

spec 驱动开发按以下六阶段推进，每阶段有明确产出物与退出条件：

### 阶段 1：搜索匹配（Search）

- **操作**：LS `.trae/specs` 目录，查找是否已有匹配的 change-id
- **产出物**：匹配判定（无匹配 / 已存在且未完成 / 已存在且已完成）
- **退出条件**：无匹配则进入阶段 2；有匹配且未完成则跳到阶段 6 继续实施

### 阶段 2：编写 spec.md（Specify）

- **操作**：与用户澄清意图，撰写 Why / What Changes / Impact / ADDED/MODIFIED/REMOVED Requirements
- **产出物**：`.trae/specs/<change-id>/spec.md`
- **退出条件**：需求边界清晰，变更范围、影响面、Requirement 全部明确

### 阶段 3：编写 tasks.md（Plan）

- **操作**：基于 spec.md 拆解有序任务列表，每个任务拆分为子任务，并声明 Task Dependencies
- **产出物**：`.trae/specs/<change-id>/tasks.md`
- **退出条件**：任务可独立执行，依赖关系清晰

### 阶段 4：编写 checklist.md（Checkpoint）

- **操作**：按 Phase 分组编写可勾选检查点，确保覆盖 spec.md 中所有 Requirement
- **产出物**：`.trae/specs/<change-id>/checklist.md`
- **退出条件**：检查点覆盖所有 Requirement，无遗漏

### 阶段 5：通知审批（Notify）

- **操作**：调用 NotifyUser 工具，向用户呈现 spec.md / tasks.md / checklist.md 三文档摘要，等待审批
- **退出条件**：用户批准后进入阶段 6；用户要求修改则回流到对应阶段

### 阶段 6：实施与验证（Implement & Verify）

- **操作**：Sub-Agent 按 tasks.md 顺序执行任务，TodoWrite 与 tasks.md 保持同步，逐项核对 checklist.md
- **退出条件**：所有 checklist.md 复选框勾选完成，返回最终响应

## File Structure

spec 文档统一存放在 `.trae/specs/<change-id>/` 目录下：

```
.trae/specs/<change-id>/
├── spec.md        # 变更规范：Why / What Changes / Impact / Requirements
├── tasks.md       # 任务清单：有序任务 + 子任务 + 依赖关系
└── checklist.md   # 验证检查点：按 Phase 分组的可勾选项
```

三文档职责分工：

- **spec.md**：定义"做什么"与"为什么"，明确需求边界与变更影响面
- **tasks.md**：定义"怎么做"，将变更拆解为可独立执行的任务与依赖关系
- **checklist.md**：定义"如何验证"，提供按阶段分组的检查点清单

## Document Templates

三文档的内容结构示例如下。

### spec.md 模板

```markdown
# [Feature Name] Spec

## Why
[1-2 句问题/机会说明]

## What Changes
- [变更列表]
- [BREAKING 变更需标注 **BREAKING**]

## Impact
- Affected specs: [列表]
- Affected code: [关键文件/系统]

## ADDED Requirements
### Requirement: 新增能力
系统 SHALL ...
#### Scenario: 成功场景
- **WHEN** ...
- **THEN** ...

## MODIFIED Requirements
### Requirement: 已有能力
[完整修改后的需求]

## REMOVED Requirements
### Requirement: 移除的能力
**Reason**: [原因]
**Migration**: [迁移方式]
```

### tasks.md 模板

```markdown
# Tasks

## Task 1: [任务标题]
- [ ] 1.1: [子任务]
- [ ] 1.2: [子任务]

## Task 2: [任务标题]
- [ ] 2.1: [子任务]

# Task Dependencies
- [Task 2] depends on [Task 1]
```

### checklist.md 模板

```markdown
# Checklist

## Phase 1 — [阶段名]
- [ ] [检查点 1]
- [ ] [检查点 2]

## Phase 2 — [阶段名]
- [ ] [检查点 3]
```

## change-id Naming

change-id 命名遵循以下规范：

- **动词引导**：以动词开头，常用前缀包括 `add-`、`refactor-`、`migrate-`、`remove-`、`update-`、`fix-`
- **kebab-case**：全小写，单词以连字符 `-` 分隔，不使用下划线或驼峰
- **唯一性**：与 `.trae/specs` 下已有目录不冲突，创建前需 LS 检查
- **关键词匹配**：与用户意图关键词对应，便于后续检索与匹配

**示例**：

- `add-rgba-to-grayscale` — 新增 RGBA 转灰度能力
- `refactor-gpu-context-thread-safety` — 重构 GpuContext 线程安全
- `remove-sha256-default-feature` — 移除 SHA-256 默认 feature
- `migrate-buffer-pool-to-tiered` — 迁移 BufferPool 至分层结构

## Implementation Rules

实施阶段遵循以下规则：

- **TodoWrite 对齐**：使用 TodoWrite 工具与 tasks.md 任务保持同步，每完成一个任务即更新状态
- **Sub-Agent 独占实施**：不在主对话直接写代码，所有代码实现委托给 Sub-Agent 执行
- **无依赖任务可并行**：多个无依赖任务可在同一消息中并行调用 Sub-Agent，提升执行效率
- **完成后勾选**：每完成一个任务，修改 tasks.md 勾选对应复选框 `[ ]` 改为 `[x]`
- **不删除 spec 文档**：实施完成后保留 spec.md / tasks.md / checklist.md 作为变更记录，便于后续追溯

## Verification Rules

验证阶段遵循以下规则：

- **逐项核对**：读取 checklist.md，对每个检查点检查相关代码 / 文档 / 行为是否满足要求
- **通过则勾选**：满足要求的检查点改为 `[x]`，并写入 checklist.md
- **失败回流**：检查点失败时，在 tasks.md 新增修复任务，回流到实施阶段重新执行
- **全部通过后结束**：所有检查点勾选完成后，返回最终响应，不再调用 NotifyUser

## Anti-patterns

下列行为应避免：

- ❌ **proposal 阶段写实现代码**：应只创建 spec 文档，代码留到实施阶段
- ❌ **跳过 NotifyUser 直接实施**：应等待用户审批后再进入实施阶段
- ❌ **未勾选完成的任务**：应及时更新 tasks.md 复选框，保持状态同步
- ❌ **删除 spec 文档**：应保留作为变更记录，便于后续追溯与审计
- ❌ **创建无关文档**：应只创建 spec.md / tasks.md / checklist.md 三个文档
- ❌ **change-id 与现有目录冲突**：应保证唯一性，创建前 LS 检查
- ❌ **用 AskUserQuestion 询问审批**：应用 NotifyUser 请求审批，AskUserQuestion 仅用于澄清意图
- ❌ **实施阶段不使用 Sub-Agent**：应委托给 Sub-Agent 执行，主对话只负责协调与验证
