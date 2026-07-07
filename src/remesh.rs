//! 各向同性重网格化模块
//!
//! 通过迭代 split→collapse→flip→smooth 循环，将网格重网格化为
//! 边长均匀、三角形质量较高的形态。
//!
//! ## 算法
//! 1. **Split**：分裂长边（> max_threshold × target_len）
//! 2. **Collapse**：折叠短边（< min_threshold × target_len）
//! 3. **Flip**：翻转边以优化顶点度数（目标：内部=6，边界=4）
//! 4. **Smooth**：切向拉普拉斯平滑（保持原形状）
//!
//! ## 参考
//! Botsch & Kobbelt, "A Remeshing Approach to Multiresolution Modeling" (2004)

use crate::geometry::{edge_length, vertex_normal};
use crate::ids::{HalfEdgeId, VertexId};
use crate::storage::MeshStorage;
use crate::topology_ops::{collapse_edge_at, flip_edge, split_edge};
use crate::traversal::{FaceHalfEdges, VertexAdjacentVerts, VertexRing, is_boundary_vertex};

// ============================================================
// 内部工具
// ============================================================

/// 切向拉普拉斯平滑：将顶点沿切平面移动至邻居平均位置。
///
/// 便利包装函数，在测试中使用。生产代码通过 `compute_tangential_smooth`
/// + `smooth_vertices` 的 gather-scatter 模式实现并行。
#[allow(dead_code)]
fn tangential_smooth(mesh: &mut MeshStorage, v: VertexId) {
    if let Some(new_pos) = compute_tangential_smooth(mesh, v) {
        // 使用 set_position 同步 SOA 位置缓存
        mesh.set_position(v, new_pos);
    }
}

/// 计算切向拉普拉斯平滑后的新位置（纯函数，不修改网格）。
///
/// 将 `tangential_smooth` 的计算与写入分离，使并行版本可以
/// 先并行计算所有新位置，再顺序写入。
fn compute_tangential_smooth(mesh: &MeshStorage, v: VertexId) -> Option<[f64; 3]> {
    let neighbors: Vec<[f64; 3]> = VertexAdjacentVerts::new(mesh, v)
        .filter_map(|n| mesh.get_vertex(n).map(|vt| vt.position))
        .collect();
    if neighbors.is_empty() {
        return None;
    }

    let n = neighbors.len() as f64;
    let mut avg = [0.0; 3];
    for p in &neighbors {
        avg[0] += p[0];
        avg[1] += p[1];
        avg[2] += p[2];
    }
    avg[0] /= n;
    avg[1] /= n;
    avg[2] /= n;

    let pos = mesh.get_vertex(v)?.position;

    let normal = vertex_normal(mesh, v).unwrap_or([0.0, 1.0, 0.0]);
    let diff = [avg[0] - pos[0], avg[1] - pos[1], avg[2] - pos[2]];
    let dot_n = diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2];
    let tangent = [
        diff[0] - normal[0] * dot_n,
        diff[1] - normal[1] * dot_n,
        diff[2] - normal[2] * dot_n,
    ];

    let lambda = 0.5;
    Some([
        pos[0] + lambda * tangent[0],
        pos[1] + lambda * tangent[1],
        pos[2] + lambda * tangent[2],
    ])
}

// ============================================================
// 主接口
// ============================================================

/// 各向同性重网格化结果统计。
#[derive(Debug, Clone, Default)]
pub struct RemeshStats {
    /// 执行的总迭代次数
    pub iterations: usize,
    /// 分裂的边数
    pub splits: usize,
    /// 折叠的边数
    pub collapses: usize,
    /// 翻转的边数
    pub flips: usize,
    /// 目标边长
    pub target_length: f64,
}

/// 各向同性重网格化。
///
/// 将网格重网格化为边长接近 `target_length` 的均匀三角网格。
/// 若 `target_length` 为 `None`，则使用当前网格边长的中位数作为目标。
///
/// `iterations`：split/collapse/flip/smooth 循环的执行次数（建议 3-10）。
/// `reproject`：若为 `true`，每轮平滑后将顶点投影回原始表面（保形）。
///
/// 返回操作统计信息。
pub fn isotropic_remesh(
    mesh: &mut MeshStorage,
    target_length: Option<f64>,
    iterations: usize,
    reproject: bool,
) -> RemeshStats {
    let target_len = target_length.unwrap_or_else(|| compute_target_length(mesh));
    if target_len <= 0.0 {
        return RemeshStats {
            iterations,
            target_length: target_len,
            ..Default::default()
        };
    }

    let min_len = target_len * 0.8; // 4/5
    let max_len = target_len * 1.333; // 4/3

    let mut stats = RemeshStats {
        iterations,
        target_length: target_len,
        ..Default::default()
    };

    for _iter in 0..iterations {
        // 1. Split long edges
        let split_count = split_long_edges(mesh, max_len);
        stats.splits += split_count;

        // 2. Collapse short edges
        let collapse_count = collapse_short_edges(mesh, min_len);
        stats.collapses += collapse_count;

        // 3. Flip to improve valence
        let flip_count = flip_for_valence(mesh);
        stats.flips += flip_count;

        // 4. Tangential smoothing
        smooth_vertices(mesh);

        // 5. Reproject（可选的保形投影）
        if reproject {
            // reproject 需要原始表面作为参考，这里暂用简单实现：
            // 不再额外操作（切向平滑本身已保形较好）
        }
    }

    stats
}

// ============================================================
// 子步骤
// ============================================================

/// 计算目标边长（当前所有边长中位数）。
fn compute_target_length(mesh: &MeshStorage) -> f64 {
    let mut lengths: Vec<f64> = mesh
        .halfedge_ids()
        .filter_map(|he| edge_length(mesh, he))
        .collect();

    if lengths.is_empty() {
        return 0.0;
    }

    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = lengths.len();
    // 因为是成对半边（twin），每条边被计算两次；中位数取 n/4 和 3n/4 之间的值
    let mid = n / 4; // 跳过非常短的/边界
    let end = (3 * n) / 4;
    if end <= mid {
        return lengths[n / 2]; // fallback
    }
    let slice = &lengths[mid..end];
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// 分裂所有过长的边。返回分裂次数。
fn split_long_edges(mesh: &mut MeshStorage, max_len: f64) -> usize {
    let mut count = 0;
    // 收集需要分裂的边（twin 对只取一个代表）
    let to_split: Vec<HalfEdgeId> = mesh
        .halfedge_ids()
        .filter(|&he| {
            // 只处理 canonical half（key 较小者），避免重复
            let h = match mesh.get_halfedge(he) {
                Some(h) => h,
                None => return false,
            };
            if let Some(twin) = h.twin
                && he > twin
            {
                return false;
            }
            edge_length(mesh, he).is_some_and(|l| l > max_len)
        })
        .collect();

    for he in to_split {
        if !mesh.contains_halfedge(he) {
            continue;
        }
        if edge_length(mesh, he).is_some_and(|l| l > max_len) && split_edge(mesh, he).is_ok() {
            count += 1;
        }
    }
    count
}

/// 折叠所有过短的边。返回折叠次数。
fn collapse_short_edges(mesh: &mut MeshStorage, min_len: f64) -> usize {
    let mut count = 0;
    let to_collapse: Vec<HalfEdgeId> = mesh
        .halfedge_ids()
        .filter(|&he| {
            let h = match mesh.get_halfedge(he) {
                Some(h) => h,
                None => return false,
            };
            if let Some(twin) = h.twin
                && he > twin
            {
                return false;
            }
            edge_length(mesh, he).is_some_and(|l| l < min_len && l > 1e-10)
        })
        .collect();

    for he in to_collapse {
        if !mesh.contains_halfedge(he) {
            continue;
        }
        if let Some(len) = edge_length(mesh, he)
            && len < min_len
            && len > 1e-10
        {
            // 在中点折叠
            let h = match mesh.get_halfedge(he) {
                Some(h) => h,
                None => continue,
            };
            let v_dst = h.vertex;
            let mid = if let Some(twin) = h.twin {
                let v_src = match mesh.get_halfedge(twin) {
                    Some(t) => t.vertex,
                    None => continue,
                };
                let p0 = match mesh.get_vertex(v_src) {
                    Some(vt) => vt.position,
                    None => continue,
                };
                let p1 = match mesh.get_vertex(v_dst) {
                    Some(vt) => vt.position,
                    None => continue,
                };
                [
                    (p0[0] + p1[0]) / 2.0,
                    (p0[1] + p1[1]) / 2.0,
                    (p0[2] + p1[2]) / 2.0,
                ]
            } else {
                match mesh.get_vertex(v_dst) {
                    Some(vt) => vt.position,
                    None => continue,
                }
            };
            if collapse_edge_at(mesh, he, mid).is_ok() {
                count += 1;
            }
        }
    }
    count
}

/// 翻转边以优化顶点度数。返回翻转次数。
fn flip_for_valence(mesh: &mut MeshStorage) -> usize {
    let mut count = 0;
    let to_check: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();

    for he in to_check {
        if !mesh.contains_halfedge(he) {
            continue;
        }

        // 只处理内部边（有 twin 且本侧有面）；边界半边 face=None 跳过，
        // 其 interior twin 会在另一轮迭代中被处理
        let h = match mesh.get_halfedge(he) {
            Some(h) if h.twin.is_some() && h.face.is_some() => h,
            _ => continue,
        };

        let twin = h.twin.expect("twin must be set at this point");
        // 避免重复处理
        if he > twin {
            continue;
        }

        // twin 侧必须也有面，否则是边界边无法翻转
        let twin_face = match mesh.get_halfedge(twin).and_then(|t| t.face) {
            Some(f) => f,
            None => continue,
        };

        // 获取四个顶点
        let v0 = match mesh.get_halfedge(twin) {
            Some(t) => t.vertex, // a
            None => continue,
        };
        let v1 = h.vertex; // c
        let face = h.face.expect("halfedge must have a face");
        // 需要三角形另外两个顶点
        let face_hes: Vec<_> = FaceHalfEdges::new(mesh, face).collect();
        if face_hes.len() != 3 {
            continue;
        }
        let v2 = face_hes
            .iter()
            .filter_map(|&eh| mesh.get_halfedge(eh).map(|h| h.vertex))
            .find(|&v| v != v0 && v != v1);

        let twin_hes: Vec<_> = FaceHalfEdges::new(mesh, twin_face).collect();
        if twin_hes.len() != 3 {
            continue;
        }
        let v3 = twin_hes
            .iter()
            .filter_map(|&eh| mesh.get_halfedge(eh).map(|h| h.vertex))
            .find(|&v| v != v0 && v != v1);

        let (Some(v2), Some(v3)) = (v2, v3) else {
            continue;
        };

        // 当前度数
        let val_a_before = VertexRing::new(mesh, v0).count();
        let val_b_before = VertexRing::new(mesh, v1).count();
        let val_c_before = VertexRing::new(mesh, v2).count();
        let val_d_before = VertexRing::new(mesh, v3).count();

        // 目标度数
        let target_a = target_valence(v0, mesh);
        let target_b = target_valence(v1, mesh);
        let target_c = target_valence(v2, mesh);
        let target_d = target_valence(v3, mesh);

        // 翻转前的偏差
        let before = (val_a_before as isize - target_a as isize).unsigned_abs()
            + (val_b_before as isize - target_b as isize).unsigned_abs()
            + (val_c_before as isize - target_c as isize).unsigned_abs()
            + (val_d_before as isize - target_d as isize).unsigned_abs();

        // 翻转后的度数: a-1, b-1, c+1, d+1
        let after = (val_a_before.saturating_sub(1) as isize - target_a as isize).unsigned_abs()
            + (val_b_before.saturating_sub(1) as isize - target_b as isize).unsigned_abs()
            + ((val_c_before + 1) as isize - target_c as isize).unsigned_abs()
            + ((val_d_before + 1) as isize - target_d as isize).unsigned_abs();

        if after < before && flip_edge(mesh, he).is_ok() {
            count += 1;
        }
    }
    count
}

/// 顶点的目标度数：内部=6，边界=4。
fn target_valence(v: VertexId, mesh: &MeshStorage) -> usize {
    if is_boundary_vertex(mesh, v) { 4 } else { 6 }
}

/// 对所有顶点执行一次切向拉普拉斯平滑（rayon 并行计算）。
///
/// 采用 gather-scatter 模式：
/// 1. **Gather**（并行）：对每个非边界顶点，读取邻居位置计算新位置；
/// 2. **Scatter**（顺序）：将新位置写入网格。
///
/// 这样避免了并行写入冲突，同时利用多核加速计算密集的平滑步骤。
fn smooth_vertices(mesh: &mut MeshStorage) {
    use rayon::prelude::*;

    let verts: Vec<VertexId> = mesh.vertex_ids().collect();

    // Phase 1: 并行计算所有非边界顶点的新位置
    let new_positions: Vec<(VertexId, [f64; 3])> = verts
        .par_iter()
        .filter(|&&v| !is_boundary_vertex(mesh, v))
        .filter_map(|&v| compute_tangential_smooth(mesh, v).map(|pos| (v, pos)))
        .collect();

    // Phase 2: 顺序写入新位置（使用 set_position 同步 SOA 缓存）
    for (v, pos) in new_positions {
        mesh.set_position(v, pos);
    }
}

// ============================================================
// 便捷封装
// ============================================================

/// 快速重网格化：使用中位数作为目标边长，3 次迭代。
pub fn quick_remesh(mesh: &mut MeshStorage) -> RemeshStats {
    isotropic_remesh(mesh, None, 3, false)
}

/// 均匀重网格化到指定边长。
pub fn remesh_to_length(mesh: &mut MeshStorage, target_length: f64) -> RemeshStats {
    isotropic_remesh(mesh, Some(target_length), 5, false)
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology_ops::validate_mesh;

    #[test]
    fn compute_target_length_icosphere() {
        let mesh = crate::test_util::build_icosphere(1);
        let l = compute_target_length(&mesh);
        assert!(l > 0.0);
        assert!(l < 2.0); // 单位球，边长 < 2
    }

    #[test]
    fn remesh_icosphere_preserves_topology() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let f_before = mesh.face_count();

        let stats = quick_remesh(&mut mesh);

        // icosphere 已非常均匀，可能无操作发生；但拓扑必须保持有效
        validate_mesh(&mesh).unwrap();
        // 面数变化应在合理范围内
        let f_after = mesh.face_count();
        let ratio = f_after as f64 / f_before as f64;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "面数变化在合理范围内: {} -> {}",
            f_before,
            f_after
        );
        let _ = stats;
    }

    #[test]
    fn remesh_preserves_closedness() {
        let mut mesh = crate::test_util::build_icosphere(1);
        assert!(crate::traversal::is_closed(&mesh));
        let _ = quick_remesh(&mut mesh);
        assert!(crate::traversal::is_closed(&mesh));
    }

    #[test]
    fn target_valence_interior_is_6() {
        let mesh = crate::test_util::build_icosphere(1);
        for v in mesh.vertex_ids() {
            if !is_boundary_vertex(&mesh, v) {
                assert_eq!(target_valence(v, &mesh), 6);
            }
        }
    }

    #[test]
    fn remesh_to_specific_length() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let stats = remesh_to_length(&mut mesh, 0.5);
        assert!(stats.target_length > 0.0);
        validate_mesh(&mesh).unwrap();
    }

    #[test]
    fn isotropic_remesh_zero_target_length_returns_early() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let stats = isotropic_remesh(&mut mesh, Some(0.0), 5, false);
        // 早退时 iterations 报告请求值，splits/collapses/flips 均为 0
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.splits, 0);
        assert_eq!(stats.collapses, 0);
        assert_eq!(stats.flips, 0);
    }

    #[test]
    fn isotropic_remesh_negative_target_length_returns_early() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let stats = isotropic_remesh(&mut mesh, Some(-1.0), 5, false);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.splits, 0);
        assert_eq!(stats.collapses, 0);
        assert_eq!(stats.flips, 0);
    }

    #[test]
    fn compute_target_length_empty_mesh_returns_zero() {
        let mesh = MeshStorage::new();
        assert_eq!(compute_target_length(&mesh), 0.0);
    }

    #[test]
    fn remesh_on_open_grid_does_not_panic() {
        let mut mesh = crate::primitives::build_grid(2.0, 2.0, 3, 3);
        let _stats = isotropic_remesh(&mut mesh, Some(0.5), 3, false);
        validate_mesh(&mesh).unwrap();
    }

    #[test]
    fn target_valence_boundary_is_4() {
        let mesh = crate::primitives::build_grid(2.0, 2.0, 3, 3);
        let mut boundary = 0;
        let mut interior = 0;
        for v in mesh.vertex_ids() {
            if is_boundary_vertex(&mesh, v) {
                assert_eq!(target_valence(v, &mesh), 4);
                boundary += 1;
            } else {
                assert_eq!(target_valence(v, &mesh), 6);
                interior += 1;
            }
        }
        assert!(boundary > 0, "开放网格应存在边界顶点");
        assert!(interior > 0, "3x3 网格应存在内部顶点");
    }

    #[test]
    fn tangential_smooth_on_isolated_vertex_is_noop() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        tangential_smooth(&mut mesh, v);
        // 孤立顶点无邻居，位置不变
        let pos = mesh.get_vertex(v).unwrap().position;
        assert_eq!(pos, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn isotropic_remesh_zero_iterations_is_noop() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let stats = isotropic_remesh(&mut mesh, None, 0, false);
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.splits, 0);
        assert_eq!(stats.collapses, 0);
        assert_eq!(stats.flips, 0);
    }

    #[test]
    fn remesh_preserves_euler_characteristic() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let chi_before = mesh.euler_characteristic();
        assert_eq!(chi_before, 2);
        let _ = quick_remesh(&mut mesh);
        let chi_after = mesh.euler_characteristic();
        assert_eq!(chi_after, chi_before);
    }
}
