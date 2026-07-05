# 拓扑验证

> 设计文档：[validate.pdf](../../../docs/validate.pdf) | 源码：[validate.rs](../../../src/validate.rs)

## 简介

对半边网格执行拓扑自检，覆盖 twin/next/悬空 ID/退化/流形约束等 8 类检查，确保网格满足流形与一致性约束。

## 关键 API

| 名称 | 功能 |
|------|------|
| `check_topology` | 执行拓扑检查 |
| `validate_topology` | 拓扑验证 |
| `ValidationError` | 验证错误类型 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/validate.pdf)。
