//! Catmull-Clark 细分模块
//!
//! 实现 Catmull-Clark 细分算法，支持任意多边形网格（三角形、四边形等）。
//! 一次细分后所有面变为四边形（输出时三角化为 2 个三角形）。
//!
//! ## 算法概述
//! Catmull-Clark 细分是\textbf{逼近型}细分方案，是双三次 B 样条曲面的精确推广。
//! 一次细分后所有非四边形面被消除（变为四边形），多次细分后网格趋于光滑。
//!
//! ### 步骤
//! 1. **面点（Face Point）**：每个面的顶点平均值
//!    $\text{fp} = \frac{1}{k}\sum_{i=0}^{k-1} v_i$
//! 2. **边点（Edge Point）**：边两端点 + 相邻两面点的平均值
//!    - 内部边：$\text{ep} = \frac{v_0 + v_1 + \text{fp}_L + \text{fp}_R}{4}$
//!    - 边界边：$\text{ep} = \frac{v_0 + v_1}{2}$
//! 3. **顶点更新（Vertex Point）**：
//!    - 内部：$v' = \frac{F + 2R + (n-3)P}{n}$，其中 $F$=邻接面点均值，
//!      $R$=邻接边中点均值，$P$=原顶点，$n$=度
//!    - 边界：$v' = \frac{v_{\text{prev}} + 6v + v_{\text{next}}}{8}$
//! 4. **面分裂**：每个原始 $k$ 边形分裂为 $k$ 个四边形，每个四边形三角化为 2 个三角形。
//!
//! ### 规模变化
//! 设原始网格有 $V$ 顶点、$E$ 边、$F$ 面，细分后：
//! $$V' = V + E + F, \quad F' = 2\sum_i k_i \quad (\text{三角化后})$$
//!
//! ## 与 Loop 细分的关键不同
//! - Catmull-Clark 输出四边形网格（三角化后为 2 三角形/面），Loop 输出三角形网格
//! - Catmull-Clark 适用于任意多边形输入，Loop 仅适用于三角形
//! - Catmull-Clark 一次细分后所有面变为四边形（即使输入含三角形）

use std::collections::HashMap;

use crate::ids::{FaceId, VertexId};
use crate::io::build_mesh_from_vertices_and_faces;
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexRing, is_boundary_edge, is_boundary_vertex};

// ============================================================
// 公开 API
// ============================================================

/// 对任意多边形网格执行一次 Catmull-Clark 细分，返回细分后的新网格。
///
/// 输入网格不修改，输出为全新构建的 `MeshStorage`（三角化后的四边形网格）。
///
/// # 算法
/// 1. **面点**：每个面的顶点平均值
/// 2. **边点**：内部边 `(v0+v1+fp_L+fp_R)/4`；边界边 `(v0+v1)/2`
/// 3. **顶点更新**：内部 `v'=(F+2R+(n-3)P)/n`；边界 `v'=(v_prev+6v+v_next)/8`
/// 4. **面分裂**：每个 k-gon → k 个四边形 → 2k 个三角形
///
/// # 规模变化
/// `V' = V + E + F`；`F' = 2 * sum(各面边数)`。
///
/// # 健壮性
/// - 所有查询通过 `mesh.get_*` + `Option` 链式调用，无效 ID 跳过；
/// - 边界顶点邻居数 ≠ 2 时保持原位置（退化情况）；
/// - 内部顶点 valence=0 时保持原位置（孤立顶点）。
pub fn catmull_clark_subdivide(mesh: &MeshStorage) -> MeshStorage {
    // ---------- 1. 收集原始顶点 ----------
    let orig_v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let n_orig = orig_v_ids.len();
    let mut v_index: HashMap<VertexId, u32> = HashMap::new();
    for (i, &v) in orig_v_ids.iter().enumerate() {
        v_index.insert(v, i as u32);
    }

    // ---------- 2. 收集所有面（任意多边形）----------
    let mut orig_faces: Vec<Vec<u32>> = Vec::new();
    let mut face_id_to_idx: HashMap<FaceId, usize> = HashMap::new();
    for f_id in mesh.face_ids() {
        let verts: Vec<u32> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| v_index.get(&v).copied())
            .collect();
        if verts.len() >= 3 {
            face_id_to_idx.insert(f_id, orig_faces.len());
            orig_faces.push(verts);
        }
    }
    let n_faces = orig_faces.len();

    // ---------- 3. 计算面点 ----------
    let face_points: Vec<[f64; 3]> = orig_faces
        .iter()
        .map(|face| {
            let k = face.len() as f64;
            let mut sum = [0.0; 3];
            for &vi in face {
                let pos = mesh
                    .get_vertex(orig_v_ids[vi as usize])
                    .map(|v| v.position)
                    .unwrap_or([0.0; 3]);
                sum[0] += pos[0];
                sum[1] += pos[1];
                sum[2] += pos[2];
            }
            [sum[0] / k, sum[1] / k, sum[2] / k]
        })
        .collect();

    // ---------- 4. 计算边点 ----------
    // edge_to_face_points: 每条边收集邻接面点
    let mut edge_face_points: HashMap<(u32, u32), Vec<[f64; 3]>> = HashMap::new();
    for (f_idx, face) in orig_faces.iter().enumerate() {
        let k = face.len();
        for i in 0..k {
            let a = face[i];
            let b = face[(i + 1) % k];
            let key = edge_key(a, b);
            edge_face_points
                .entry(key)
                .or_default()
                .push(face_points[f_idx]);
        }
    }

    // 遍历半边，每条无向边计算一次边点
    let mut edge_point_pos: HashMap<(u32, u32), [f64; 3]> = HashMap::new();
    for he_id in mesh.halfedge_ids() {
        let h = match mesh.get_halfedge(he_id) {
            Some(h) => h,
            None => continue,
        };
        let v_t = h.vertex;
        let v_o = match h.twin.and_then(|t| mesh.get_halfedge(t)) {
            Some(t) => t.vertex,
            None => continue,
        };
        let i_o = match v_index.get(&v_o) {
            Some(&i) => i,
            None => continue,
        };
        let i_t = match v_index.get(&v_t) {
            Some(&i) => i,
            None => continue,
        };
        if i_o == i_t {
            continue;
        }
        let key = edge_key(i_o, i_t);
        if edge_point_pos.contains_key(&key) {
            continue;
        }

        let p_o = mesh
            .get_vertex(orig_v_ids[i_o as usize])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);
        let p_t = mesh
            .get_vertex(orig_v_ids[i_t as usize])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);

        let fps = edge_face_points.get(&key).cloned().unwrap_or_default();
        let ep = match fps.len() {
            2 => {
                // 内部边
                [
                    (p_o[0] + p_t[0] + fps[0][0] + fps[1][0]) / 4.0,
                    (p_o[1] + p_t[1] + fps[0][1] + fps[1][1]) / 4.0,
                    (p_o[2] + p_t[2] + fps[0][2] + fps[1][2]) / 4.0,
                ]
            }
            _ => {
                // 边界边或退化（无邻接面 / 多于 2 面）
                [
                    (p_o[0] + p_t[0]) / 2.0,
                    (p_o[1] + p_t[1]) / 2.0,
                    (p_o[2] + p_t[2]) / 2.0,
                ]
            }
        };
        edge_point_pos.insert(key, ep);
    }

    // ---------- 5. 构建新顶点位置数组 ----------
    // 布局：[原始顶点(更新后), 边点, 面点]
    let mut edge_keys: Vec<(u32, u32)> = edge_point_pos.keys().cloned().collect();
    edge_keys.sort();
    let n_edges = edge_keys.len();

    let mut edge_point_idx: HashMap<(u32, u32), u32> = HashMap::new();
    for (i, &key) in edge_keys.iter().enumerate() {
        edge_point_idx.insert(key, (n_orig + i) as u32);
    }

    let face_point_start = n_orig + n_edges;

    let mut new_positions: Vec<[f64; 3]> = Vec::with_capacity(n_orig + n_edges + n_faces);
    // 占位原始顶点（稍后更新）
    for &v in &orig_v_ids {
        new_positions.push(mesh.get_vertex(v).map(|v| v.position).unwrap_or([0.0; 3]));
    }
    // 边点
    for &key in &edge_keys {
        new_positions.push(edge_point_pos[&key]);
    }
    // 面点
    for fp in &face_points {
        new_positions.push(*fp);
    }

    // ---------- 6. 更新原始顶点位置 ----------
    for (i, &v) in orig_v_ids.iter().enumerate() {
        let p_old = new_positions[i]; // 旧位置（已被边点/面点计算使用）

        let new_pos = if is_boundary_vertex(mesh, v) {
            // 边界顶点：v' = (v_prev + 6*v + v_next) / 8
            let boundary_neighbors: Vec<VertexId> = VertexRing::new(mesh, v)
                .filter(|&he| is_boundary_edge(mesh, he))
                .filter_map(|he| mesh.get_halfedge(he))
                .map(|h| h.vertex)
                .collect();
            if boundary_neighbors.len() == 2 {
                let p_prev = mesh
                    .get_vertex(boundary_neighbors[0])
                    .map(|v| v.position)
                    .unwrap_or([0.0; 3]);
                let p_next = mesh
                    .get_vertex(boundary_neighbors[1])
                    .map(|v| v.position)
                    .unwrap_or([0.0; 3]);
                [
                    (p_prev[0] + 6.0 * p_old[0] + p_next[0]) / 8.0,
                    (p_prev[1] + 6.0 * p_old[1] + p_next[1]) / 8.0,
                    (p_prev[2] + 6.0 * p_old[2] + p_next[2]) / 8.0,
                ]
            } else {
                p_old
            }
        } else {
            // 内部顶点：v' = (F + 2R + (n-3)*P) / n
            let outgoing: Vec<crate::ids::HalfEdgeId> = VertexRing::new(mesh, v).collect();
            let n = outgoing.len();
            if n == 0 {
                p_old
            } else {
                let n_f = n as f64;
                // F = 邻接面点均值
                let mut f_sum = [0.0; 3];
                let mut f_count = 0usize;
                // R = 邻接边中点均值
                let mut r_sum = [0.0; 3];

                for &he in &outgoing {
                    let h = match mesh.get_halfedge(he) {
                        Some(h) => h,
                        None => continue,
                    };
                    // 邻接面点
                    if let Some(f_id) = h.face
                        && let Some(&f_idx) = face_id_to_idx.get(&f_id)
                    {
                        let fp = face_points[f_idx];
                        f_sum[0] += fp[0];
                        f_sum[1] += fp[1];
                        f_sum[2] += fp[2];
                        f_count += 1;
                    }
                    // 邻接边中点 = (v + neighbor) / 2
                    let neighbor_pos = mesh
                        .get_vertex(h.vertex)
                        .map(|v| v.position)
                        .unwrap_or([0.0; 3]);
                    let mid = [
                        (p_old[0] + neighbor_pos[0]) / 2.0,
                        (p_old[1] + neighbor_pos[1]) / 2.0,
                        (p_old[2] + neighbor_pos[2]) / 2.0,
                    ];
                    r_sum[0] += mid[0];
                    r_sum[1] += mid[1];
                    r_sum[2] += mid[2];
                }

                if f_count == 0 || f_count != n {
                    // 退化（部分边无面），保持原位置
                    p_old
                } else {
                    let fc = f_count as f64;
                    let f_avg = [f_sum[0] / fc, f_sum[1] / fc, f_sum[2] / fc];
                    let r_avg = [r_sum[0] / n_f, r_sum[1] / n_f, r_sum[2] / n_f];
                    [
                        (f_avg[0] + 2.0 * r_avg[0] + (n_f - 3.0) * p_old[0]) / n_f,
                        (f_avg[1] + 2.0 * r_avg[1] + (n_f - 3.0) * p_old[1]) / n_f,
                        (f_avg[2] + 2.0 * r_avg[2] + (n_f - 3.0) * p_old[2]) / n_f,
                    ]
                }
            }
        };
        new_positions[i] = new_pos;
    }

    // ---------- 7. 构建新面（三角化四边形）----------
    let mut new_faces: Vec<[u32; 3]> = Vec::new();
    for (f_idx, face) in orig_faces.iter().enumerate() {
        let k = face.len();
        let fp_idx = (face_point_start + f_idx) as u32;
        for i in 0..k {
            let v_i = face[i];
            let v_next = face[(i + 1) % k];
            let v_prev = face[(i + k - 1) % k];
            let ep_i = edge_point_idx[&edge_key(v_i, v_next)];
            let ep_prev = edge_point_idx[&edge_key(v_prev, v_i)];
            // 四边形 (v_i, ep_i, fp, ep_prev) → 2 个三角形
            new_faces.push([v_i, ep_i, fp_idx]);
            new_faces.push([v_i, fp_idx, ep_prev]);
        }
    }

    // ---------- 8. 构建输出网格 ----------
    build_mesh_from_vertices_and_faces(&new_positions, &new_faces)
}

// ============================================================
// 内部辅助
// ============================================================

/// 边键：(min, max)，保证无向边唯一表示。
#[inline]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::build_mesh_from_polygons;
    use crate::validate::check_topology;

    /// 构建单位立方体（8 顶点、6 四边形面）。
    fn build_cube() -> MeshStorage {
        let vertices = [
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [0.0, 0.0, 1.0], // 4
            [1.0, 0.0, 1.0], // 5
            [1.0, 1.0, 1.0], // 6
            [0.0, 1.0, 1.0], // 7
        ];
        let faces = vec![
            vec![0, 3, 2, 1], // bottom (-z)
            vec![4, 5, 6, 7], // top (+z)
            vec![0, 1, 5, 4], // front (-y)
            vec![3, 7, 6, 2], // back (+y)
            vec![0, 4, 7, 3], // left (-x)
            vec![1, 2, 6, 5], // right (+x)
        ];
        build_mesh_from_polygons(&vertices, &faces)
    }

    // ---------- 规模验证 ----------

    #[test]
    fn catmull_clark_cube_vertex_count() {
        // 立方体：V=8, E=12, F=6
        // 细分后：V' = 8 + 12 + 6 = 26
        let mesh = build_cube();
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.face_count(), 6);

        let refined = catmull_clark_subdivide(&mesh);
        assert_eq!(
            refined.vertex_count(),
            26,
            "立方体细分后顶点数应为 8+12+6=26"
        );
    }

    #[test]
    fn catmull_clark_cube_face_count() {
        // 立方体 6 个四边形面，每个分裂为 4 个四边形 = 24 四边形
        // 三角化后 = 48 三角形
        let mesh = build_cube();
        let refined = catmull_clark_subdivide(&mesh);
        assert_eq!(refined.face_count(), 48, "立方体细分后面数应为 6*4*2=48");
    }

    #[test]
    fn catmull_clark_cube_halfedge_count() {
        // 闭合三角网格：HE = 3 * F = 3 * 48 = 144
        let mesh = build_cube();
        let refined = catmull_clark_subdivide(&mesh);
        assert_eq!(refined.halfedge_count(), 144);
    }

    // ---------- 拓扑校验 ----------

    #[test]
    fn catmull_clark_cube_passes_validation() {
        let mesh = build_cube();
        let refined = catmull_clark_subdivide(&mesh);
        assert!(
            check_topology(&refined).is_ok(),
            "细分后的网格应通过完整拓扑校验: {:?}",
            check_topology(&refined)
        );
    }

    // ---------- Euler 示性数 ----------

    #[test]
    fn catmull_clark_cube_preserves_euler_characteristic() {
        // 立方体（6 四边形面）Euler: V - E + F = 8 - 12 + 6 = 2
        // 细分后仍为闭合定向曲面，Euler = 2
        let mesh = build_cube();
        let refined = catmull_clark_subdivide(&mesh);
        let v = refined.vertex_count() as i64;
        let e = (refined.halfedge_count() / 2) as i64;
        let f = refined.face_count() as i64;
        assert_eq!(v - e + f, 2, "细分后 Euler 示性数应为 2");
    }

    // ---------- 单四边形 ----------

    #[test]
    fn catmull_clark_single_quad() {
        // 单四边形面（全是边界边）
        // V=4, E=4, F=1 → V'=4+4+1=9, F'=2*4=8
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![vec![0, 1, 2, 3]];
        let mesh = build_mesh_from_polygons(&vertices, &faces);
        let refined = catmull_clark_subdivide(&mesh);
        assert_eq!(
            refined.vertex_count(),
            9,
            "单四边形细分后顶点数应为 4+4+1=9"
        );
        assert_eq!(refined.face_count(), 8, "单四边形细分后面数应为 2*4=8");
        assert!(check_topology(&refined).is_ok());
    }

    // ---------- 三角形输入 ----------

    #[test]
    fn catmull_clark_triangle_input() {
        // 单三角形面（3 边形）
        // V=3, E=3, F=1 → V'=3+3+1=7, F'=2*3=6
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![vec![0, 1, 2]];
        let mesh = build_mesh_from_polygons(&vertices, &faces);
        let refined = catmull_clark_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 7, "三角形细分后顶点数应为 3+3+1=7");
        assert_eq!(refined.face_count(), 6, "三角形细分后面数应为 2*3=6");
        assert!(check_topology(&refined).is_ok());
    }

    // ---------- 闭合四面体（三角面）----------

    #[test]
    fn catmull_clark_closed_tetrahedron() {
        // 闭合四面体：V=4, E=6, F=4（三角形面）
        // 细分后：V'=4+6+4=14, F'=2*3*4=24
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![
            vec![0, 2, 1], // bottom
            vec![0, 1, 3], // front
            vec![0, 3, 2], // left
            vec![1, 2, 3], // top
        ];
        let mesh = build_mesh_from_polygons(&vertices, &faces);
        let refined = catmull_clark_subdivide(&mesh);
        assert_eq!(
            refined.vertex_count(),
            14,
            "四面体细分后顶点数应为 4+6+4=14"
        );
        assert_eq!(refined.face_count(), 24, "四面体细分后面数应为 2*3*4=24");

        let v = refined.vertex_count() as i64;
        let e = (refined.halfedge_count() / 2) as i64;
        let f = refined.face_count() as i64;
        assert_eq!(v - e + f, 2, "闭合曲面 Euler 示性数应为 2");
        assert!(check_topology(&refined).is_ok());
    }

    // ---------- 位置正确性 ----------

    #[test]
    fn catmull_clark_cube_face_point_at_center() {
        // 立方体每个面的面点应在面中心
        // 底面 [0,3,2,1] 中心 = (0.5, 0.5, 0)
        let mesh = build_cube();
        let refined = catmull_clark_subdivide(&mesh);

        // 面点索引范围：8(原顶点) + 12(边点) = 20 .. 26
        // 底面的面点是第 0 个面，索引 = 20
        let bottom_fp = refined
            .vertex_ids()
            .nth(20)
            .and_then(|v| refined.get_vertex(v))
            .map(|v| v.position)
            .expect("面点应存在");

        assert!(
            (bottom_fp[0] - 0.5).abs() < 1e-12
                && (bottom_fp[1] - 0.5).abs() < 1e-12
                && bottom_fp[2].abs() < 1e-12,
            "底面面点应在 (0.5, 0.5, 0)，实际在 {:?}",
            bottom_fp
        );
    }

    #[test]
    fn catmull_clark_cube_vertex_moves_inward() {
        // 立方体角点 (1,1,1) 细分后应向中心移动
        // 内部顶点公式：v' = (F + 2R + (n-3)*P) / n, n=3（立方体角点度数=3）
        // F = 3 个邻接面点均值, R = 3 个邻接边中点均值
        let mesh = build_cube();
        let refined = catmull_clark_subdivide(&mesh);

        // 原顶点 (1,1,1) = v_ids[6]，细分后仍为索引 6
        let v6 = refined
            .vertex_ids()
            .nth(6)
            .and_then(|v| refined.get_vertex(v))
            .map(|v| v.position)
            .expect("顶点应存在");

        // (1,1,1) 应向中心 (0.5,0.5,0.5) 移动
        assert!(
            v6[0] < 1.0 && v6[1] < 1.0 && v6[2] < 1.0,
            "角点 (1,1,1) 应向中心移动，实际在 {:?}",
            v6
        );
    }

    // ---------- 二次细分 ----------

    #[test]
    fn catmull_clark_double_subdivide() {
        // 两次细分后仍应通过拓扑校验
        let mesh = build_cube();
        let refined1 = catmull_clark_subdivide(&mesh);
        let refined2 = catmull_clark_subdivide(&refined1);
        assert!(check_topology(&refined2).is_ok(), "二次细分后应通过校验");

        // 第一次：V1=26, F1=48, E1=3*48/2=72
        // 第二次（输入为三角网格）：V2 = V1 + E1 + F1 = 26 + 72 + 48 = 146
        assert_eq!(
            refined2.vertex_count(),
            146,
            "二次细分顶点数应为 26+72+48=146"
        );
    }
}
