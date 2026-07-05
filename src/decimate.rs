//! QEM（Quadric Error Metrics）网格简化模块
//!
//! 参考 Garland & Heckbert 1997 "Surface Simplification Using Quadric Error Metrics"。
//!
//! 核心思想：为每个顶点维护一个 4×4 对称二次误差矩阵 $Q$，记录该顶点到所有
//! 邻接面平面的距离平方和。折叠边 $(v_0, v_1)$ 时，合并后的 Quadric
//! $Q = Q_{v_0} + Q_{v_1}$，最优新顶点位置 $p^* = \arg\min_p p^T Q p$，
//! 折叠代价 $= p^{*T} Q p^*$。算法贪心地选择代价最小的边折叠，直到达到目标面数。
//!
//! ## 算法步骤
//! 1. **初始化**：遍历所有面，计算平面方程 $\\(a,b,c,d\\)$，为每个顶点累加
//!    $K_p = p \cdot p^T$（$p = \[a,b,c,d\]$）。
//! 2. **建堆**：对每条非边界边计算 $\\(cost, p^*\\)$，压入最小堆。
//! 3. **折叠循环**：弹出最小代价边，校验有效性后调用
//!    [`collapse_edge_at`] 执行折叠，更新 $Q_K = Q_A + Q_B$，
//!    重新计算 $K$ 的邻接边代价并入堆。
//! 4. **终止**：面数 $\le$ `target_faces` 或堆空。
//!
//! ## 边界处理
//! - 边界边不参与折叠（保持边界拓扑）。
//! - QEM 矩阵奇异（共面区域）时回退到中点/端点中代价最小者。
//! - 折叠失败（链接条件、退化）时跳过该边继续。

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use crate::geometry::face_normal;
use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::MeshStorage;
use crate::topology_ops::{TopologyError, collapse_edge_at};
use crate::traversal::{FaceHalfEdges, VertexRing, is_boundary_edge};

// ============================================================
// Quadric 二次误差矩阵
// ============================================================

/// 4×4 对称二次误差矩阵，存储为 10 个 f64（利用对称性）。
#[derive(Clone, Debug)]
struct Quadric {
    // 按 Garland-Heckbert 约定存储 10 个独立分量：
    // [a², ab, ac, ad, b², bc, bd, c², cd, d²]
    // 对应矩阵:
    //   [ q00 q01 q02 q03 ]
    //   [ q01 q11 q12 q13 ]
    //   [ q02 q12 q22 q23 ]
    //   [ q03 q13 q23 q33 ]
    data: [f64; 10],
}

impl Quadric {
    /// 零矩阵。
    fn zero() -> Self {
        Self { data: [0.0; 10] }
    }

    /// 从平面方程 $ax+by+cz+d=0$ 构造基本二次误差矩阵 $K_p = p \cdot p^T$
    /// （$p = \[a,b,c,d\]$）。
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            data: [
                a * a,
                a * b,
                a * c,
                a * d, // q00 q01 q02 q03
                b * b,
                b * c,
                b * d, // q11 q12 q13
                c * c,
                c * d, // q22 q23
                d * d, // q33
            ],
        }
    }

    /// 矩阵加法：$Q_1 + Q_2$。
    fn add(&self, other: &Self) -> Self {
        let mut r = Self::zero();
        for i in 0..10 {
            r.data[i] = self.data[i] + other.data[i];
        }
        r
    }

    /// 评估顶点 $v=[x,y,z,1]$ 的二次误差：$v^T Q v$。
    fn evaluate(&self, pos: [f64; 3]) -> f64 {
        let [x, y, z] = pos;
        let q = &self.data;
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }

    /// 求解 $\arg\min v^T Q v$，即最优折叠位置。
    ///
    /// 令 $\partial(v^T Q v)/\partial x = \partial/\partial y = \partial/\partial z = 0$，
    /// 得 3×3 线性方程组：
    /// ```text
    /// [ q00 q01 q02 ] [x]   [-q03]
    /// [ q01 q11 q12 ] [y] = [-q13]
    /// [ q02 q12 q22 ] [z]   [-q23]
    /// ```
    /// 用 Cramer 法则求解。矩阵奇异（$|\det| < 10^{-12}$）时返回 `None`。
    fn find_optimal_position(&self) -> Option<[f64; 3]> {
        let q = &self.data;
        let (q00, q01, q02, q03) = (q[0], q[1], q[2], q[3]);
        let (q11, q12, q13) = (q[4], q[5], q[6]);
        let (q22, q23) = (q[7], q[8]);

        let det = q00 * (q11 * q22 - q12 * q12) - q01 * (q01 * q22 - q12 * q02)
            + q02 * (q01 * q12 - q11 * q02);

        if det.abs() < 1e-12 {
            return None;
        }

        let (c0, c1, c2) = (-q03, -q13, -q23);

        let det_x = c0 * (q11 * q22 - q12 * q12) - q01 * (c1 * q22 - q12 * c2)
            + q02 * (c1 * q12 - q11 * c2);

        let det_y = q00 * (c1 * q22 - q12 * c2) - c0 * (q01 * q22 - q12 * q02)
            + q02 * (q01 * c2 - c1 * q02);

        let det_z = q00 * (q11 * c2 - c1 * q12) - q01 * (q01 * c2 - c1 * q02)
            + c0 * (q01 * q12 - q11 * q02);

        Some([det_x / det, det_y / det, det_z / det])
    }
}

// ============================================================
// 辅助函数
// ============================================================

/// 计算面的平面方程 $\\(a, b, c, d\\)$，其中 $ax+by+cz+d=0$，$(a,b,c)$ 为单位法向。
fn face_plane(mesh: &MeshStorage, f: FaceId) -> Option<(f64, f64, f64, f64)> {
    let n = face_normal(mesh, f)?;
    let he = mesh.get_face(f)?.halfedge?;
    let v0 = mesh.get_halfedge(he)?.vertex;
    let p0 = mesh.get_vertex(v0)?.position;
    let d = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);
    Some((n[0], n[1], n[2], d))
}

/// 计算边 `he` 的折叠代价和最优位置。
///
/// 候选位置：$p^*$（若有）、$v_0$、$v_1$、中点。选代价最小者。
/// 返回 `(cost, position)`。`None` 表示拓扑不完整。
fn edge_cost_and_position(
    mesh: &MeshStorage,
    he: HalfEdgeId,
    quadrics: &HashMap<VertexId, Quadric>,
) -> Option<(f64, [f64; 3])> {
    let h = mesh.get_halfedge(he)?;
    let twin_id = h.twin?;
    let twin = mesh.get_halfedge(twin_id)?;
    let v0 = twin.vertex; // origin
    let v1 = h.vertex; // tip

    let q0 = quadrics.get(&v0)?;
    let q1 = quadrics.get(&v1)?;
    let q = q0.add(q1);

    let p0 = mesh.get_vertex(v0)?.position;
    let p1 = mesh.get_vertex(v1)?.position;
    let mid = [
        (p0[0] + p1[0]) * 0.5,
        (p0[1] + p1[1]) * 0.5,
        (p0[2] + p1[2]) * 0.5,
    ];

    // 候选位置：p_opt（若有）、v0、v1、中点
    let mut candidates: Vec<[f64; 3]> = vec![p0, p1, mid];
    if let Some(p_opt) = q.find_optimal_position() {
        candidates.push(p_opt);
    }

    let mut best_cost = f64::INFINITY;
    let mut best_pos = mid;
    for pos in &candidates {
        let c = q.evaluate(*pos);
        if c < best_cost {
            best_cost = c;
            best_pos = *pos;
        }
    }

    if best_cost.is_nan() {
        best_cost = f64::INFINITY;
    }

    Some((best_cost, best_pos))
}

// ============================================================
// 可比较的代价包装（f64 不实现 Ord）
// ============================================================

#[derive(Clone, Copy)]
struct CostKey(f64);

impl PartialEq for CostKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for CostKey {}

impl PartialOrd for CostKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CostKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// ============================================================
// 公开 API
// ============================================================

/// 使用 QEM 将网格简化到目标面数。
///
/// 每次折叠代价最小且满足链接条件的边，直到面数 $\le$ `target_faces`。
/// 若 `target_faces` $\ge$ 当前面数，不操作。
///
/// # 返回
/// 实际移除的面数。
///
/// # 算法
/// 1. 为每个顶点初始化 Quadric（累加邻接面平面 $K_p$）；
/// 2. 对每条非边界边计算 $(cost, p^*)$，压入最小堆；
/// 3. 贪心弹出最小代价边，调用 [`collapse_edge_at`] 折叠；
/// 4. 更新 $Q_K = Q_A + Q_B$，重新计算 $K$ 的邻接边代价。
///
/// # 边界处理
/// - 边界边不参与折叠；
/// - QEM 矩阵奇异时回退到端点/中点中代价最小者；
/// - 折叠失败（链接条件、退化）时跳过。
pub fn decimate_qem(mesh: &mut MeshStorage, target_faces: usize) -> Result<usize, TopologyError> {
    let initial_faces = mesh.face_count();
    if target_faces >= initial_faces {
        return Ok(0);
    }

    // ---------- 1. 初始化顶点 Quadric ----------
    let mut quadrics: HashMap<VertexId, Quadric> = HashMap::new();
    for v_id in mesh.vertex_ids() {
        quadrics.insert(v_id, Quadric::zero());
    }
    for f_id in mesh.face_ids() {
        if let Some((a, b, c, d)) = face_plane(mesh, f_id) {
            let kp = Quadric::from_plane(a, b, c, d);
            for he in FaceHalfEdges::new(mesh, f_id) {
                if let Some(h) = mesh.get_halfedge(he)
                    && let Some(q) = quadrics.get_mut(&h.vertex)
                {
                    *q = q.add(&kp);
                }
            }
        }
    }

    // ---------- 2. 构建边代价堆 ----------
    let mut heap: BinaryHeap<(Reverse<CostKey>, HalfEdgeId)> = BinaryHeap::new();
    let mut cost_map: HashMap<HalfEdgeId, (f64, [f64; 3])> = HashMap::new();

    for he_id in mesh.halfedge_ids() {
        if is_boundary_edge(mesh, he_id) {
            continue;
        }
        if let Some((cost, pos)) = edge_cost_and_position(mesh, he_id, &quadrics) {
            heap.push((Reverse(CostKey(cost)), he_id));
            cost_map.insert(he_id, (cost, pos));
        }
    }

    // ---------- 3. 贪心折叠循环 ----------
    let mut faces_removed = 0;

    while mesh.face_count() > target_faces {
        let (Reverse(CostKey(heap_cost)), he_id) = match heap.pop() {
            Some(entry) => entry,
            None => break, // 堆空，无法继续
        };

        // 半边是否仍有效
        if !mesh.contains_halfedge(he_id) {
            continue;
        }

        // 过期条目检测（代价已更新）
        let (stored_cost, stored_pos) = match cost_map.get(&he_id) {
            Some(&c) => c,
            None => continue,
        };
        if (heap_cost - stored_cost).abs() > 1e-9 {
            continue;
        }

        // 是否已变为边界边
        if is_boundary_edge(mesh, he_id) {
            continue;
        }

        // 获取端点与待删除半边
        let h = match mesh.get_halfedge(he_id) {
            Some(h) => h.clone(),
            None => continue,
        };
        let twin_id = match h.twin {
            Some(t) => t,
            None => continue,
        };
        let twin = match mesh.get_halfedge(twin_id) {
            Some(t) => t.clone(),
            None => continue,
        };
        let v_a = twin.vertex; // origin
        let v_b = h.vertex; // tip

        // 收集 6 条将被删除的半边
        let deleted_hes: Vec<HalfEdgeId> = [he_id, twin_id]
            .iter()
            .copied()
            .chain(h.next)
            .chain(h.prev)
            .chain(twin.next)
            .chain(twin.prev)
            .collect();

        // 合并 Quadric
        let q_a = quadrics.get(&v_a).cloned().unwrap_or_else(Quadric::zero);
        let q_b = quadrics.get(&v_b).cloned().unwrap_or_else(Quadric::zero);
        let q_k = q_a.add(&q_b);

        // 执行折叠
        match collapse_edge_at(mesh, he_id, stored_pos) {
            Ok(k) => {
                faces_removed += 2;

                // 更新 quadrics
                quadrics.remove(&v_a);
                quadrics.remove(&v_b);
                quadrics.insert(k, q_k);

                // 清理被删除半边的 cost_map
                for &dh in &deleted_hes {
                    cost_map.remove(&dh);
                }

                // 重新计算 K 的邻接边代价
                for out_he in VertexRing::new(mesh, k).collect::<Vec<_>>() {
                    if is_boundary_edge(mesh, out_he) {
                        continue;
                    }
                    if let Some((cost, pos)) = edge_cost_and_position(mesh, out_he, &quadrics) {
                        let twin_he = mesh.get_halfedge(out_he).and_then(|h| h.twin);
                        heap.push((Reverse(CostKey(cost)), out_he));
                        cost_map.insert(out_he, (cost, pos));
                        // 同步更新 twin（同一边的两个方向）
                        if let Some(t) = twin_he {
                            heap.push((Reverse(CostKey(cost)), t));
                            cost_map.insert(t, (cost, pos));
                        }
                    }
                }
            }
            Err(_) => {
                // 折叠失败（链接条件、退化等），跳过
                continue;
            }
        }
    }

    Ok(faces_removed)
}

/// 简化到目标顶点数（等价于 $2 \cdot \text{target\_verts} - 4$ 个面，闭合流形）。
///
/// 对于闭合三角网格，Euler 公式 $V - E + F = 2$ 且 $E = 3F/2$，故
/// $V = F/2 + 2$，即 $F = 2(V - 2) = 2V - 4$。
pub fn decimate_to_vertices(
    mesh: &mut MeshStorage,
    target_verts: usize,
) -> Result<usize, TopologyError> {
    let target_faces = 2usize.saturating_mul(target_verts).saturating_sub(4);
    decimate_qem(mesh, target_faces)
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::build_mesh_from_vertices_and_faces;
    use crate::test_util::build_icosphere;
    use crate::topology_ops::validate_mesh;
    use crate::validate::check_topology;

    // ---------- Quadric 单元测试 ----------

    #[test]
    fn quadric_zero_evaluates_to_zero() {
        let q = Quadric::zero();
        assert_eq!(q.evaluate([1.0, 2.0, 3.0]), 0.0);
        assert_eq!(q.evaluate([0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn quadric_from_plane_zero_at_plane_point() {
        // 平面 x + y + z = 0，单位法向 (1/√3, 1/√3, 1/√3)
        let s = 1.0 / 3f64.sqrt();
        let q = Quadric::from_plane(s, s, s, 0.0);
        // 原点在平面上，误差应为 0
        assert!(q.evaluate([0.0, 0.0, 0.0]).abs() < 1e-12);
        // (1,-1,0) 也在平面上
        assert!(q.evaluate([1.0, -1.0, 0.0]).abs() < 1e-12);
        // (1,0,0) 到平面距离 = 1/√3，误差 = 距离² = 1/3
        let err = q.evaluate([1.0, 0.0, 0.0]);
        assert!((err - 1.0 / 3.0).abs() < 1e-10, "err = {}", err);
    }

    #[test]
    fn quadric_add_is_commutative() {
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, 0.0);
        let s1 = q1.add(&q2);
        let s2 = q2.add(&q1);
        assert_eq!(s1.data, s2.data);
    }

    #[test]
    fn quadric_find_optimal_position_singular_returns_none() {
        // 秩 1 矩阵（平面法向沿 x 轴），3×3 子矩阵奇异
        let q = Quadric::from_plane(1.0, 0.0, 0.0, 0.0);
        assert!(q.find_optimal_position().is_none());
    }

    #[test]
    fn quadric_find_optimal_position_two_planes() {
        // 两个正交平面 x=0 和 y=0，最优位置应在 z 轴任意处
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, 0.0);
        let q = q1.add(&q2);
        // 3×3 子矩阵 = diag(1,1,0)，仍奇异（z 方向自由）
        assert!(q.find_optimal_position().is_none());
    }

    #[test]
    fn quadric_find_optimal_position_three_planes() {
        // 三个正交平面 x=0, y=0, z=0，最优位置 = 原点
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, 0.0);
        let q3 = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);
        let q = q1.add(&q2).add(&q3);
        let p = q.find_optimal_position().expect("三正交平面应有唯一最优解");
        assert!(p[0].abs() < 1e-10);
        assert!(p[1].abs() < 1e-10);
        assert!(p[2].abs() < 1e-10);
    }

    // ---------- decimate_qem 集成测试 ----------

    #[test]
    fn decimate_icosphere2_to_80_faces() {
        let mut mesh = build_icosphere(2); // V=162, F=320
        assert_eq!(mesh.face_count(), 320);

        let removed = decimate_qem(&mut mesh, 80).expect("简化应成功");
        assert!(removed > 0);

        let f = mesh.face_count();
        assert!(f <= 80, "面数 {} 应 ≤ 80", f);
        // 闭合流形 V = F/2 + 2
        let v = mesh.vertex_count();
        assert!(v <= 42 + 2, "顶点数 {} 应 ≤ 44（F≈80 → V≈42）", v);

        // 拓扑校验
        assert!(validate_mesh(&mesh).is_ok(), "简化后网格应通过拓扑校验");
    }

    #[test]
    fn decimate_icosphere2_to_tetrahedron() {
        let mut mesh = build_icosphere(2); // V=162, F=320
        let removed = decimate_qem(&mut mesh, 4).expect("简化应成功");
        assert!(removed > 0);

        let f = mesh.face_count();
        // 4 面体有 4 面，但可能因链接条件限制无法精确达到
        assert!(f <= 8, "面数 {} 应 ≤ 8（接近四面体）", f);

        // 仍为有效闭合网格
        assert!(validate_mesh(&mesh).is_ok(), "极端简化后网格应通过校验");
        assert!(mesh.vertex_count() >= 4, "至少 4 个顶点");
    }

    #[test]
    fn decimate_flat_plane_all_zero_cost() {
        // 2×2 平面网格：4 顶点，2 三角面
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [1, 3, 2]];
        let mut mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
        assert_eq!(mesh.face_count(), 2);

        // 简化到 0 面：所有代价为 0（共面），应正确折叠对角线
        let removed = decimate_qem(&mut mesh, 0).expect("简化应成功");
        assert_eq!(removed, 2, "应移除 2 面");
        assert_eq!(mesh.face_count(), 0, "面数应为 0");
        // 4 顶点 → 折叠 1 次 → 3 顶点
        assert_eq!(mesh.vertex_count(), 3, "顶点数应为 3");
    }

    #[test]
    fn decimate_boundary_edges_not_collapsed() {
        // 单个三角形：3 条边界边，无法折叠
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0, 1, 2]];
        let mut mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
        assert_eq!(mesh.face_count(), 1);

        let removed = decimate_qem(&mut mesh, 0).expect("简化应成功");
        assert_eq!(removed, 0, "边界边不应折叠，移除 0 面");
        assert_eq!(mesh.face_count(), 1, "面数不变");
        assert_eq!(mesh.vertex_count(), 3, "顶点数不变");
    }

    #[test]
    fn decimate_target_geq_current_is_noop() {
        let mut mesh = build_icosphere(0); // V=12, F=20
        let removed = decimate_qem(&mut mesh, 20).expect("简化应成功");
        assert_eq!(removed, 0);
        assert_eq!(mesh.face_count(), 20);
    }

    #[test]
    fn decimate_icosphere1_half_simplification() {
        let mut mesh = build_icosphere(1); // V=42, F=80
        let target = 40;
        let removed = decimate_qem(&mut mesh, target).expect("简化应成功");
        assert!(removed > 0);

        let f = mesh.face_count();
        assert!(f <= target, "面数 {} 应 ≤ {}", f, target);
        assert!(validate_mesh(&mesh).is_ok(), "简化后网格应通过校验");
    }

    #[test]
    fn decimate_preserves_closed_topology() {
        let mut mesh = build_icosphere(1); // V=42, F=80
        decimate_qem(&mut mesh, 20).expect("简化应成功");

        // 闭合网格 Euler 示性数 = 2
        let chi = mesh.euler_characteristic();
        assert_eq!(chi, 2, "闭合网格 Euler 示性数应保持为 2，实际 {}", chi);
    }

    #[test]
    fn decimate_to_vertices_icosphere2() {
        let mut mesh = build_icosphere(2); // V=162, F=320
        // 目标 42 顶点 → 2*42-4 = 80 面
        let removed = decimate_to_vertices(&mut mesh, 42).expect("简化应成功");
        assert!(removed > 0);

        let f = mesh.face_count();
        assert!(f <= 80, "面数 {} 应 ≤ 80", f);
        assert!(validate_mesh(&mesh).is_ok());
    }

    #[test]
    fn decimate_multiple_iterations_stay_valid() {
        let mut mesh = build_icosphere(1); // V=42, F=80
        // 连续简化 3 次
        for target in [60, 30, 10] {
            decimate_qem(&mut mesh, target).expect("简化应成功");
            assert!(validate_mesh(&mesh).is_ok(), "target={} 时校验失败", target);
            assert!(
                check_topology(&mesh).is_ok(),
                "target={} 时完整校验失败",
                target
            );
        }
    }
}
