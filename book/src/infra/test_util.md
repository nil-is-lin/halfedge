# 测试夹具

> 设计文档：[test_util.pdf](../../../docs/test_util.pdf) | 源码：[test_util.rs](../../../src/test_util.rs)

## 简介

提供测试用夹具与辅助工具，最核心的是 icosphere 球体生成器，用于在测试与示例中快速获得光滑封闭网格。

## 关键 API

| 名称 | 功能 |
|------|------|
| `build_icosphere` | 生成 icosphere 球体 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/test_util.pdf)。
