# Checklist

> 验证 spec skill 文档化规范的实施完成度。
> 检查点对齐 spec.md 的 3 个 Requirement 与 tasks.md 的 5 个任务。

---

## 文档存在性验证

- [x] `docs/skill/SKILL.md` 文件已创建
- [x] 文件路径精确为 `docs/skill/SKILL.md`（非 `docs/SKILL.md`，避免与已有项目 skill 混淆）

## 元信息头验证

- [x] YAML frontmatter 包含 `name` 字段（值为 `spec-driven-development` 或同类语义名称）
- [x] YAML frontmatter 包含 `version` 字段
- [x] YAML frontmatter 包含 `updated` 字段（日期为 `2026-07-04` 或之后）
- [x] YAML frontmatter 包含 `description` 字段，一句话说明 spec 驱动开发用途

## 章节完整性验证（对应 spec.md Requirement: Spec Skill 文档内容完整性）

- [x] 「When to Invoke」章节存在，列出触发 spec 模式的场景
- [x] 「Workflow Overview」章节存在，呈现六阶段流程
- [x] 「File Structure」章节存在，说明 `.trae/specs/<change-id>/` 目录结构
- [x] 「Document Templates」章节存在，包含 spec.md 内容结构示例
- [x] 「Document Templates」章节包含 tasks.md 内容结构示例
- [x] 「Document Templates」章节包含 checklist.md 内容结构示例
- [x] 「change-id Naming」章节存在，说明命名规范
- [x] 「Implementation Rules」章节存在，包含 TodoWrite 对齐、Sub-Agent 并行等规则
- [x] 「Verification Rules」章节存在，包含逐项核对与失败回流机制
- [x] 「Anti-patterns」章节存在，列出常见误用

## 内容质量验证

- [x] Workflow Overview 明确每阶段的产出物与退出条件
- [x] spec.md 模板示例包含 Why / What Changes / Impact / ADDED/MODIFIED/REMOVED Requirements 六大块
- [x] tasks.md 模板示例包含有序任务列表 + 子任务 + Task Dependencies
- [x] checklist.md 模板示例包含按 Phase 分组的可勾选检查点
- [x] change-id Naming 规范说明：动词引导、kebab-case、唯一性

## 语言与约定验证（对应 spec.md Requirement: 文档语言与项目约定一致）

- [x] 章节标题与说明文字为中文
- [x] 文件路径（如 `.trae/specs/<change-id>/spec.md`）保持英文原样
- [x] 代码块内容保持英文原样
- [x] 无中英文混用导致的语义歧义

## 可操作性验证

- [x] 开发者阅读后能独立创建符合规范的 spec.md / tasks.md / checklist.md
- [x] 文档可直接参照套用，无需额外查阅其他资料
- [x] Anti-patterns 章节能帮助识别并避免常见误用
