//! 拓扑操作共享的辅助类型。

use std::fmt;

use crate::ids::{HalfEdgeId, VertexId};

// ============================================================
// 错误类型
// ============================================================

/// 拓扑操作失败原因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// 传入的半边句柄无效或已被删除。
    InvalidHalfEdge(HalfEdgeId),
    /// 试图翻转边界边（禁止）。
    FlipOnBoundaryEdge(HalfEdgeId),
    /// 试图折叠边界边（禁止）。
    CollapseOnBoundaryEdge(HalfEdgeId),
    /// 半边没有 twin（拓扑未完整缝合）。
    NoTwin(HalfEdgeId),
    /// 半边没有 face（两侧均无面，无法操作）。
    NoFace(HalfEdgeId),
    /// 操作会产生退化三角形（三个顶点共线或重合）。
    DegenerateTriangle,
    /// 链接条件不满足，折叠会产生非流形。
    LinkConditionViolated { a: VertexId, b: VertexId },
    /// 网格拓扑不一致（twin 不互指、next/prev 不闭合等）。
    Inconsistent(String),
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHalfEdge(h) => write!(f, "无效半边句柄 {:?}", h),
            Self::FlipOnBoundaryEdge(h) => write!(f, "禁止翻转边界边 {:?}", h),
            Self::CollapseOnBoundaryEdge(h) => write!(f, "禁止折叠边界边 {:?}", h),
            Self::NoTwin(h) => write!(f, "半边 {:?} 没有 twin", h),
            Self::NoFace(h) => write!(f, "半边 {:?} 两侧均无面", h),
            Self::DegenerateTriangle => write!(f, "操作会产生退化三角形"),
            Self::LinkConditionViolated { a, b } => {
                write!(f, "链接条件不满足：折叠 {:?}-{:?} 会产生非流形", a, b)
            }
            Self::Inconsistent(msg) => write!(f, "网格拓扑不一致：{}", msg),
        }
    }
}

impl std::error::Error for TopologyError {}
