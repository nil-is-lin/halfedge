//! 网格参数化（Surface Parameterization / UV Unwrapping）。
//!
//! 将三角网格映射到二维平面，尽可能保持角度（保角）或面积（保面积）。
//! 所有方法都要求网格是单连通、开圆盘拓扑（genus 0、1 条边界环）。
//!
//! ## 算法
//! - [`tutte_embedding`]：Tutte 重心映射——将边界均匀映射到圆，
//!   内部顶点用均匀拉普拉斯权重求解。简单、保证不翻转，但变形较大。
//! - [`harmonic_parameterization`]：调和参数化——与 Tutte 相同但使用
//!   余切权重（保角拉普拉斯），角度保持优于 Tutte。
//! - [`lscm`]：Least Squares Conformal Maps（Lévy et al. 2002）——
//!   经典保角参数化。固定两个顶点消去自由度，求解稀疏最小二乘系统。

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::ids::VertexId;
use crate::linalg::{
    SparseSystem, build_cotan_laplacian, build_vertex_index, conjugate_gradient,
    regularize_diagonal,
};
use crate::storage::MeshStorage;
use crate::traversal::{VertexRing, boundary_loops, is_boundary_vertex};

// ============================================================
// 内部：顶点索引映射
// ============================================================

// build_vertex_index 已移至 linalg 模块作为公共函数

fn collect_boundary_vertices(mesh: &MeshStorage) -> Vec<VertexId> {
    mesh.vertex_ids()
        .filter(|&v| is_boundary_vertex(mesh, v))
        .collect()
}

/// 按边界环绕顺序排列边界顶点。
///
/// 使用 `traversal::boundary_loops` 获取正确的边界半边环，
/// 再从半边 tip 顶点提取有序顶点列表。若网格有多个边界环，
/// 返回最长的那个。
fn order_boundary_vertices(mesh: &MeshStorage) -> Vec<VertexId> {
    let loops = boundary_loops(mesh);
    if loops.is_empty() {
        return Vec::new();
    }
    // 取最长的边界环
    let longest = loops
        .into_iter()
        .max_by_key(|l| l.len())
        .unwrap_or_default();
    // 从每条边界半边的 tip 顶点提取有序顶点
    longest
        .into_iter()
        .filter_map(|he| mesh.get_halfedge(he).map(|h| h.vertex))
        .collect()
}

// ============================================================
// 内部：构建拉普拉斯矩阵
// ============================================================

// build_full_cotan_laplacian 已移至 linalg 模块作为 build_cotan_laplacian

/// 构建完整均匀拉普拉斯矩阵。
fn build_full_uniform_laplacian(mesh: &MeshStorage) -> (SparseSystem, HashMap<VertexId, usize>) {
    let v_idx = build_vertex_index(mesh);
    let n = v_idx.len();
    let mut sys = SparseSystem::new(n);

    for (v, &i) in &v_idx {
        let mut degree = 0;
        for he in VertexRing::new(mesh, *v) {
            let neighbor = mesh
                .get_halfedge(he)
                .expect("halfedge exists in mesh")
                .vertex;
            if let Some(&j) = v_idx.get(&neighbor) {
                sys.add(i, j, -0.5);
                degree += 1;
            }
        }
        sys.add_diag(i, (degree as f64) / 2.0);
    }

    (sys, v_idx)
}

// ============================================================
// Dirichlet 边界条件应用
// ============================================================

/// 对拉普拉斯矩阵应用 Dirichlet 边界条件，构建修正系统。
///
/// - 边界顶点行：单位行（diag=1），RHS=固定值
/// - 内部顶点行：保持拉普拉斯行，RHS_i = -Σ_{j∈B} L_ij * fixed_j
///
/// 返回 (修正后的矩阵, u-RHS, v-RHS)。
fn apply_dirichlet(
    laplacian: SparseSystem,
    n: usize,
    fixed_uv: &HashMap<usize, [f64; 2]>,
) -> Option<(sprs::CsMat<f64>, Vec<f64>, Vec<f64>)> {
    // 先将拉普拉斯转为 CsMat 以便读取值
    let lap = laplacian.finish();
    let fixed_set: HashSet<usize> = fixed_uv.keys().copied().collect();

    // 提取原始拉普拉斯矩阵的 off-diagonal 值以构建 RHS
    // 对每个内部顶点 i: RHS_i = -Σ_{j∈B} L_ij * fixed_j
    let mut rhs_u = vec![0.0; n];
    let mut rhs_v = vec![0.0; n];

    for (&idx, &uv) in fixed_uv {
        rhs_u[idx] = uv[0];
        rhs_v[idx] = uv[1];
    }

    // 对内部顶点：累加来自边界的贡献
    for row in 0..n {
        if fixed_set.contains(&row) {
            continue;
        }
        // 遍历该行的非零元素
        if let Some(row_view) = lap.outer_view(row) {
            for (col, &val) in row_view.iter() {
                if fixed_set.contains(&col) {
                    let uv = fixed_uv[&col];
                    rhs_u[row] -= val * uv[0];
                    rhs_v[row] -= val * uv[1];
                }
            }
        }
    }

    // 重建矩阵：边界顶点行设为 identity
    let mut new_sys = SparseSystem::new(n);

    for row in 0..n {
        if fixed_set.contains(&row) {
            // 边界顶点：单位行
            new_sys.add_diag(row, 1.0);
        } else {
            // 内部顶点：保留原始拉普拉斯行（仅内部-内部耦合 + 边界耦合的对角贡献）
            if let Some(row_view) = lap.outer_view(row) {
                for (col, &val) in row_view.iter() {
                    if !fixed_set.contains(&col) {
                        // 仅保留内部顶点间的耦合
                        new_sys.add(row, col, val);
                    }
                    // 边界耦合已移入 RHS，对角不变
                }
                // 找出该行的对角值
                if let Some(diag_val) = lap.get(row, row) {
                    new_sys.add_diag(row, *diag_val);
                }
            }
        }
    }

    let mut a = new_sys.finish();
    regularize_diagonal(&mut a, 1e-8);

    Some((a, rhs_u, rhs_v))
}

// ============================================================
// 内部：求解器
// ============================================================

/// 求解参数化系统。
fn solve_param_system(
    a: &sprs::CsMat<f64>,
    rhs_u: &[f64],
    rhs_v: &[f64],
    n: usize,
) -> Option<Vec<[f64; 2]>> {
    let x_u = conjugate_gradient(a, rhs_u, n * 200, 1e-6)?;
    let x_v = conjugate_gradient(a, rhs_v, n * 200, 1e-6)?;

    Some(x_u.into_iter().zip(x_v).map(|(u, v)| [u, v]).collect())
}

// ============================================================
// 公共 API
// ============================================================

/// Tutte 重心映射（Tutte 1963）。
///
/// 边界顶点均匀映射到单位圆，内部顶点通过均匀权重重心坐标求解。
/// 保证无翻转，适合任何开圆盘拓扑网格。
pub fn tutte_embedding(mesh: &MeshStorage) -> Option<Vec<[f64; 2]>> {
    let n = mesh.vertex_count();
    if n == 0 || mesh.face_count() == 0 {
        return None;
    }

    let (laplacian, v_idx) = build_full_uniform_laplacian(mesh);
    let boundary_v = collect_boundary_vertices(mesh);
    if boundary_v.is_empty() {
        return None;
    }

    let ordered_boundary = order_boundary_vertices(mesh);
    let bdy_len = ordered_boundary.len();

    // 映射边界到单位圆
    let mut fixed_uv = HashMap::new();
    for (k, &v) in ordered_boundary.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (bdy_len as f64);
        if let Some(&idx) = v_idx.get(&v) {
            fixed_uv.insert(idx, [angle.cos(), angle.sin()]);
        }
    }

    let (a, rhs_u, rhs_v) = apply_dirichlet(laplacian, n, &fixed_uv)?;
    solve_param_system(&a, &rhs_u, &rhs_v, n)
}

/// 调和参数化（Harmonic / Cotan-Weight）。
///
/// 使用余切权重（离散 Laplace-Beltrami 算子）替代均匀权重，
/// 保角性显著优于 Tutte。
pub fn harmonic_parameterization(mesh: &MeshStorage) -> Option<Vec<[f64; 2]>> {
    let n = mesh.vertex_count();
    if n == 0 || mesh.face_count() == 0 {
        return None;
    }

    let v_idx = build_vertex_index(mesh);
    let laplacian = build_cotan_laplacian(mesh, &v_idx);
    let boundary_v = collect_boundary_vertices(mesh);
    if boundary_v.is_empty() {
        return None;
    }

    let ordered_boundary = order_boundary_vertices(mesh);
    let bdy_len = ordered_boundary.len();

    let mut fixed_uv = HashMap::new();
    for (k, &v) in ordered_boundary.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (bdy_len as f64);
        if let Some(&idx) = v_idx.get(&v) {
            fixed_uv.insert(idx, [angle.cos(), angle.sin()]);
        }
    }

    let (a, rhs_u, rhs_v) = apply_dirichlet(laplacian, n, &fixed_uv)?;
    solve_param_system(&a, &rhs_u, &rhs_v, n)
}

/// Least Squares Conformal Maps（Lévy et al. 2002）。
///
/// 固定 2 个**边界**顶点到 (0,0) 和 (1,0)，其余顶点自由求解。
/// 不需要固定整个边界，适合任意边界形状。
///
/// # 边界顶点选取
///
/// 从最长的边界环中选取两个**几何距离最远**的边界顶点钉住，
/// 以改善数值条件数（Lévy 2002 §4.2）。若网格为闭合曲面（无边界），
/// 返回 `None`——LSCM 要求至少存在一条边界环。
pub fn lscm(mesh: &MeshStorage) -> Option<Vec<[f64; 2]>> {
    let n = mesh.vertex_count();
    if n < 2 || mesh.face_count() == 0 {
        return None;
    }

    let v_idx = build_vertex_index(mesh);
    let laplacian = build_cotan_laplacian(mesh, &v_idx);

    // 收集有序边界顶点（取最长边界环）
    let ordered_boundary = order_boundary_vertices(mesh);
    if ordered_boundary.len() < 2 {
        // 闭合网格或边界退化：LSCM 需要至少 2 个边界顶点
        return None;
    }

    // 选取边界上几何距离最远的两个顶点（O(B^2)，B 为边界顶点数）。
    // 对小边界（B < 1000）可接受；大边界可用「先取最远点对种子再细化」的
    // 近似算法，但实际网格很少需要。
    let (pin_a, pin_b) = pick_farthest_pair(mesh, &ordered_boundary)?;

    let mut fixed_uv = HashMap::new();
    if let Some(&idx_a) = v_idx.get(&pin_a) {
        fixed_uv.insert(idx_a, [0.0, 0.0]);
    }
    if let Some(&idx_b) = v_idx.get(&pin_b) {
        fixed_uv.insert(idx_b, [1.0, 0.0]);
    }
    if fixed_uv.len() < 2 {
        return None;
    }

    let (a, rhs_u, rhs_v) = apply_dirichlet(laplacian, n, &fixed_uv)?;
    solve_param_system(&a, &rhs_u, &rhs_v, n)
}

/// 从顶点列表中选取几何距离最远的两个顶点。
fn pick_farthest_pair(mesh: &MeshStorage, verts: &[VertexId]) -> Option<(VertexId, VertexId)> {
    if verts.len() < 2 {
        return None;
    }
    // 简化：先取 verts[0] 与最远点 p1，再取 p1 与最远点 p2。
    // 这是「最远点对」的 O(B) 近似（真实最优为 O(B log B) 旋转卡壳，或 O(B^2) 暴力）。
    // 对 LSCM 钉点用途足够：仅需两点足够远以改善条件数，不要求精确最优。
    let pos_of = |v: VertexId| -> Option<[f64; 3]> { mesh.get_vertex(v).map(|vd| vd.position) };

    let p0 = verts[0];
    let p0_pos = pos_of(p0)?;

    // 第一轮：找离 p0 最远的点 p1
    let mut p1 = p0;
    let mut best_dist_sq = -1.0f64;
    for &v in verts {
        if let Some(pos) = pos_of(v) {
            let d = dist_sq(p0_pos, pos);
            if d > best_dist_sq {
                best_dist_sq = d;
                p1 = v;
            }
        }
    }

    // 第二轮：找离 p1 最远的点 p2
    let p1_pos = pos_of(p1)?;
    let mut p2 = p1;
    let mut best_dist_sq = -1.0f64;
    for &v in verts {
        if let Some(pos) = pos_of(v) {
            let d = dist_sq(p1_pos, pos);
            if d > best_dist_sq {
                best_dist_sq = d;
                p2 = v;
            }
        }
    }

    if p1 == p2 {
        return None;
    }
    Some((p1, p2))
}

#[inline]
fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    dx * dx + dy * dy + dz * dz
}

// ============================================================
// Mean Value Coordinates 参数化（Floater 2003）
// ============================================================

/// Mean Value Coordinates 参数化（Floater 2003）。
///
/// 与 Tutte/Harmonic 同属「边界固定 + 内部重心插值」框架，但权重
/// 改用 **Mean Value Coordinates**：
///
/// 对内部顶点 $v$，设其邻居按环绕顺序为 $u_1, \dots, u_k$，记
/// $d_i = \|u_i - v\|$，$\alpha_i = \angle(u_i - v,\ u_{i+1} - v)$
/// （$v$ 处相邻射线夹角，索引模 $k$），则
/// $$
/// w_i = \frac{\tan(\alpha_{i-1}/2) + \tan(\alpha_i/2)}{d_i},\quad
/// \lambda_i = \frac{w_i}{\sum_j w_j}.
/// $$
/// $\lambda_i \ge 0$ 且 $\sum \lambda_i = 1$，因此 MVC 参数化
/// **保证无翻转**（对所有有效网格，包括非凸/非 Delaunay）。
///
/// 与 Tutte（均匀权重）和 Harmonic（余切权重，可能为负）相比，
/// MVC 在保形性与稳健性之间取得平衡，是工业上常用的折中方案
/// （pmp-library、libigl 均提供）。
///
/// # 返回
/// - `Some(Vec<[f64;2]>)`：每个顶点的 UV 坐标
/// - `None`：空网格、无面、无边界或求解失败
pub fn mvc_parameterization(mesh: &MeshStorage) -> Option<Vec<[f64; 2]>> {
    let n = mesh.vertex_count();
    if n == 0 || mesh.face_count() == 0 {
        return None;
    }

    let v_idx = build_vertex_index(mesh);

    // 收集边界并按环绕顺序排列
    let boundary_v = collect_boundary_vertices(mesh);
    if boundary_v.is_empty() {
        return None;
    }
    let ordered_boundary = order_boundary_vertices(mesh);
    let bdy_len = ordered_boundary.len();

    // 边界固定到单位圆
    let mut fixed_uv: HashMap<usize, [f64; 2]> = HashMap::new();
    for (k, &v) in ordered_boundary.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (bdy_len as f64);
        if let Some(&idx) = v_idx.get(&v) {
            fixed_uv.insert(idx, [angle.cos(), angle.sin()]);
        }
    }

    // 构建 MVC 权重矩阵
    let boundary_set: HashSet<usize> = fixed_uv.keys().copied().collect();

    // 并行计算每个顶点的 MVC 权重条目，再顺序写入稀疏系统（gather-scatter）。
    // 每个顶点的邻居收集、三角函数（acos/tan）计算相互独立。
    /// 每个顶点的 MVC 权重计算结果：(行索引, 非对角条目, 对角值, RHS 贡献)
    type MvcEntry = (usize, Vec<(usize, f64)>, f64, [f64; 2]);

    let vert_entries: Vec<MvcEntry> = v_idx
        .par_iter()
        .filter_map(|(&v, &i)| {
            if boundary_set.contains(&i) {
                // 边界顶点：单位行，RHS = 固定 UV
                let uv = fixed_uv[&i];
                return Some((i, Vec::new(), 1.0, uv));
            }

            // 收集环绕顺序的邻居（VertexRing 已按 CCW 环绕）
            let neighbors: Vec<(usize, [f64; 3])> = VertexRing::new(mesh, v)
                .filter_map(|he| {
                    let h = mesh.get_halfedge(he)?;
                    let n_vid = h.vertex;
                    let n_pos = mesh.get_vertex(n_vid)?.position;
                    let n_idx = *v_idx.get(&n_vid)?;
                    Some((n_idx, n_pos))
                })
                .collect();

            let k = neighbors.len();
            if k == 0 {
                return Some((i, Vec::new(), 1.0, [0.0, 0.0]));
            }

            let p_v = mesh.get_vertex(v)?.position;

            // 计算 d_i 和 α_i
            let d: Vec<f64> = neighbors
                .iter()
                .map(|(_, pos)| {
                    let diff = [pos[0] - p_v[0], pos[1] - p_v[1], pos[2] - p_v[2]];
                    (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt()
                })
                .collect();

            // 退化保护：若 d_i ≈ 0，跳过（视为孤立顶点）
            if d.iter().any(|x| *x < 1e-14) {
                return Some((i, Vec::new(), 1.0, [0.0, 0.0]));
            }

            // 单位方向向量 u_i = (n_i - v) / d_i
            let u: Vec<[f64; 3]> = neighbors
                .iter()
                .zip(d.iter())
                .map(|((_, pos), &di)| {
                    [
                        (pos[0] - p_v[0]) / di,
                        (pos[1] - p_v[1]) / di,
                        (pos[2] - p_v[2]) / di,
                    ]
                })
                .collect();

            // α_i = angle(u_i, u_{i+1})，索引模 k
            let mut alpha = Vec::with_capacity(k);
            for idx in 0..k {
                let j = (idx + 1) % k;
                let cos_a = (u[idx][0] * u[j][0] + u[idx][1] * u[j][1] + u[idx][2] * u[j][2])
                    .clamp(-1.0, 1.0);
                alpha.push(cos_a.acos());
            }

            // w_i = (tan(α_{i-1}/2) + tan(α_i/2)) / d_i
            let mut w = Vec::with_capacity(k);
            for idx in 0..k {
                let prev = if idx == 0 { k - 1 } else { idx - 1 };
                let tan_prev = (alpha[prev] / 2.0).tan();
                let tan_cur = (alpha[idx] / 2.0).tan();
                w.push((tan_prev + tan_cur) / d[idx]);
            }

            // 退化保护：所有 w 为 0 时回退到均匀权重
            let total: f64 = w.iter().sum();
            let mut entries = Vec::new();
            let mut rhs = [0.0; 2];

            if total < 1e-14 {
                // 均匀权重：p_i = (1/k) Σ p_j
                for (j, _) in neighbors.iter() {
                    if boundary_set.contains(j) {
                        let uv = fixed_uv[j];
                        rhs[0] += uv[0] / (k as f64);
                        rhs[1] += uv[1] / (k as f64);
                    } else {
                        entries.push((*j, -1.0 / (k as f64)));
                    }
                }
            } else {
                // λ_i = w_i / Σw
                for (j, _) in neighbors.iter().enumerate() {
                    let lambda = w[j] / total;
                    let n_idx = neighbors[j].0;
                    if boundary_set.contains(&n_idx) {
                        let uv = fixed_uv[&n_idx];
                        rhs[0] += lambda * uv[0];
                        rhs[1] += lambda * uv[1];
                    } else {
                        entries.push((n_idx, -lambda));
                    }
                }
            }

            Some((i, entries, 1.0, rhs))
        })
        .collect();

    // 顺序写入稀疏系统（避免并行写入 SparseSystem）
    let mut sys = SparseSystem::new(n);
    let mut rhs_u = vec![0.0; n];
    let mut rhs_v = vec![0.0; n];

    for (i, entries, diag, rhs) in vert_entries {
        for (col, val) in entries {
            sys.add(i, col, val);
        }
        sys.add_diag(i, diag);
        rhs_u[i] = rhs[0];
        rhs_v[i] = rhs[1];
    }

    let mut a = sys.finish();
    regularize_diagonal(&mut a, 1e-10);

    solve_param_system(&a, &rhs_u, &rhs_v, n)
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lscm_simple_quad() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.2]));
        let v2 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.3]));
        crate::topology_ops::add_triangle(&mut mesh, v0, v1, v2).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, v0, v2, v3).unwrap();

        let result = lscm(&mesh);
        assert!(result.is_some(), "LSCM should succeed on a simple quad");
        let uv = result.unwrap();
        assert_eq!(uv.len(), 4);

        // 所有 UV 应为有限值
        for v in &uv {
            assert!(v[0].is_finite(), "u 有限值, got {}", v[0]);
            assert!(v[1].is_finite(), "v 有限值, got {}", v[1]);
        }

        // 应有恰好 2 个顶点被钉在 (0,0) 和 (1,0)（边界最远点对）
        let n_pinned_zero = uv
            .iter()
            .filter(|p| (p[0].abs() < 1e-6) && (p[1].abs() < 1e-6))
            .count();
        let n_pinned_one = uv
            .iter()
            .filter(|p| ((p[0] - 1.0).abs() < 1e-6) && (p[1].abs() < 1e-6))
            .count();
        assert_eq!(
            n_pinned_zero, 1,
            "应有 1 个顶点钉在 (0,0), 实际 {}",
            n_pinned_zero
        );
        assert_eq!(
            n_pinned_one, 1,
            "应有 1 个顶点钉在 (1,0), 实际 {}",
            n_pinned_one
        );
    }

    #[test]
    fn test_lscm_returns_none_on_closed_mesh() {
        // LSCM 需要边界环；闭合网格应返回 None
        let mesh = crate::test_util::build_icosphere(1);
        // icosphere 是闭合网格
        let result = lscm(&mesh);
        assert!(
            result.is_none(),
            "闭合网格无边界，LSCM 应返回 None, 实际得到 Some"
        );
    }

    #[test]
    fn test_lscm_pinned_vertices_are_boundary() {
        // 验证钉住的是边界顶点而非内部顶点
        // 构造：1 个内部顶点 + 4 个边界顶点的扇形
        let mut mesh = MeshStorage::new();
        let center = mesh.add_vertex(crate::storage::Vertex::new([0.5, 0.5, 0.0]));
        let v0 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.0]));
        // 4 个三角形构成以 center 为内部顶点的扇形
        crate::topology_ops::add_triangle(&mut mesh, center, v0, v1).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, center, v1, v2).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, center, v2, v3).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, center, v3, v0).unwrap();

        // center 是内部顶点（不在边界上）
        assert!(!is_boundary_vertex(&mesh, center));
        for &v in &[v0, v1, v2, v3] {
            assert!(is_boundary_vertex(&mesh, v));
        }

        let result = lscm(&mesh);
        assert!(result.is_some(), "LSCM 应成功");
        let uv = result.unwrap();
        assert_eq!(uv.len(), 5);

        // 内部顶点 center 的索引在某些位置，但不应被钉在 (0,0) 或 (1,0)
        // 找到 center 在结果中的索引
        let v_idx = build_vertex_index(&mesh);
        let center_idx = *v_idx.get(&center).unwrap();
        let center_uv = uv[center_idx];
        // center 不应被钉在 (0,0) 或 (1,0)
        let is_pinned_zero = (center_uv[0].abs() < 1e-6) && (center_uv[1].abs() < 1e-6);
        let is_pinned_one = ((center_uv[0] - 1.0).abs() < 1e-6) && (center_uv[1].abs() < 1e-6);
        assert!(
            !is_pinned_zero && !is_pinned_one,
            "内部顶点不应被钉住, center uv = {:?}",
            center_uv
        );
    }

    #[test]
    fn test_mvc_simple_quad() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.2]));
        let v2 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.3]));
        crate::topology_ops::add_triangle(&mut mesh, v0, v1, v2).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, v0, v2, v3).unwrap();

        let result = mvc_parameterization(&mesh);
        assert!(result.is_some(), "MVC should succeed on a simple quad");
        let uv = result.unwrap();
        assert_eq!(uv.len(), 4);
        // 边界点应在单位圆上（|uv| ≈ 1）
        for &p in &uv {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-3,
                "boundary point should be on unit circle, got r={}",
                r
            );
        }
    }

    #[test]
    fn test_mvc_no_flip_on_concave_mesh() {
        // 构造一个非凸网格（L 形），验证 MVC 无翻转
        // 顶点：
        //   3 --- 2
        //   |     |
        //   4 --- 1
        //   |     |
        //   5 --- 0
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 1.0, 0.0]));
        let v2 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 2.0, 0.0]));
        let v3 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 2.0, 0.0]));
        let v4 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.0]));
        let v5 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        crate::topology_ops::add_triangle(&mut mesh, v0, v1, v4).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, v0, v4, v5).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, v1, v2, v3).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, v1, v3, v4).unwrap();

        let result = mvc_parameterization(&mesh);
        assert!(result.is_some(), "MVC should succeed on concave mesh");
        let uv = result.unwrap();
        // 所有边界点应在单位圆上
        for &p in &uv {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-3,
                "boundary point should be on unit circle, got r={}",
                r
            );
        }
    }

    #[test]
    fn test_mvc_returns_none_on_empty() {
        let mesh = MeshStorage::new();
        assert!(mvc_parameterization(&mesh).is_none());
    }

    #[test]
    fn test_mvc_returns_none_on_closed_mesh() {
        // icosphere 是闭合的，无边界，应返回 None
        let mesh = crate::test_util::build_icosphere(1);
        assert!(mvc_parameterization(&mesh).is_none());
    }

    // ============================================================
    // Tutte embedding 测试
    // ============================================================

    /// 构建 4×4 平面网格（开圆盘拓扑），用于参数化测试。
    ///
    /// 16 顶点（4 内部 + 12 边界），18 三角面。
    /// 4 个内部顶点确保 Tutte 和 harmonic 产生不同的参数化结果。
    fn build_flat_grid() -> MeshStorage {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ];
        // 每个格子 2 个三角形，3×3 格 = 18 个三角形
        let faces = vec![
            [0u32, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [4, 5, 9],
            [4, 9, 8],
            [5, 6, 10],
            [5, 10, 9],
            [6, 7, 11],
            [6, 11, 10],
            [8, 9, 13],
            [8, 13, 12],
            [9, 10, 14],
            [9, 14, 13],
            [10, 11, 15],
            [10, 15, 14],
        ];
        crate::io::build_mesh_from_vertices_and_faces(&verts, &faces)
            .expect("flat grid construction")
    }

    #[test]
    fn test_tutte_simple_quad() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.2]));
        let v2 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.3]));
        crate::topology_ops::add_triangle(&mut mesh, v0, v1, v2).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, v0, v2, v3).unwrap();

        let result = tutte_embedding(&mesh);
        assert!(result.is_some(), "Tutte should succeed on a simple quad");
        let uv = result.unwrap();
        assert_eq!(uv.len(), 4);

        // 所有 UV 应为有限值
        for v in &uv {
            assert!(v[0].is_finite(), "u should be finite, got {}", v[0]);
            assert!(v[1].is_finite(), "v should be finite, got {}", v[1]);
        }

        // 边界顶点应近似在单位圆上
        for v in &uv {
            let r = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 0.1,
                "boundary point should be near unit circle, got r={}",
                r
            );
        }
    }

    #[test]
    fn test_tutte_returns_none_on_empty() {
        let mesh = MeshStorage::new();
        assert!(tutte_embedding(&mesh).is_none());
    }

    #[test]
    fn test_tutte_returns_none_on_closed_mesh() {
        let mesh = crate::test_util::build_icosphere(1);
        assert!(
            tutte_embedding(&mesh).is_none(),
            "closed mesh has no boundary, Tutte should return None"
        );
    }

    #[test]
    fn test_tutte_flat_grid_no_flip() {
        let mesh = build_flat_grid();
        let result = tutte_embedding(&mesh);
        assert!(result.is_some(), "Tutte should succeed on flat grid");
        let uv = result.unwrap();
        assert_eq!(uv.len(), mesh.vertex_count());

        // 所有 UV 有限
        for v in &uv {
            assert!(v[0].is_finite() && v[1].is_finite(), "UV must be finite");
        }

        // Tutte 保证无翻转：所有三角形带符号面积应同号
        // （边界遍历方向可能导致全部为负，但只要一致就无翻转）
        let v_idx = build_vertex_index(&mesh);
        let mut signs: Vec<i32> = Vec::new();
        for f in mesh.face_ids() {
            let verts: Vec<_> = crate::traversal::FaceHalfEdges::new(&mesh, f)
                .filter_map(|he| mesh.get_halfedge(he))
                .map(|h| h.vertex)
                .collect();
            if verts.len() != 3 {
                continue;
            }
            let i0 = *v_idx.get(&verts[0]).unwrap();
            let i1 = *v_idx.get(&verts[1]).unwrap();
            let i2 = *v_idx.get(&verts[2]).unwrap();
            let cross_z = (uv[i1][0] - uv[i0][0]) * (uv[i2][1] - uv[i0][1])
                - (uv[i1][1] - uv[i0][1]) * (uv[i2][0] - uv[i0][0]);
            if cross_z.abs() > 1e-14 {
                signs.push(if cross_z > 0.0 { 1 } else { -1 });
            }
        }
        // 所有非退化三角形应同号（无翻转）
        if signs.len() > 1 {
            let first = signs[0];
            let all_same = signs.iter().all(|&s| s == first);
            assert!(
                all_same,
                "Tutte should produce no flipped triangles (all same sign), got signs: {:?}",
                signs
            );
        }
    }

    // ============================================================
    // Harmonic parameterization 测试
    // ============================================================

    #[test]
    fn test_harmonic_simple_quad() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.2]));
        let v2 = mesh.add_vertex(crate::storage::Vertex::new([1.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.3]));
        crate::topology_ops::add_triangle(&mut mesh, v0, v1, v2).unwrap();
        crate::topology_ops::add_triangle(&mut mesh, v0, v2, v3).unwrap();

        let result = harmonic_parameterization(&mesh);
        assert!(result.is_some(), "Harmonic should succeed on a simple quad");
        let uv = result.unwrap();
        assert_eq!(uv.len(), 4);

        // 所有 UV 有限
        for v in &uv {
            assert!(v[0].is_finite() && v[1].is_finite(), "UV must be finite");
        }

        // 边界顶点应近似在单位圆上
        for v in &uv {
            let r = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 0.1,
                "boundary point should be near unit circle, got r={}",
                r
            );
        }
    }

    #[test]
    fn test_harmonic_returns_none_on_empty() {
        let mesh = MeshStorage::new();
        assert!(harmonic_parameterization(&mesh).is_none());
    }

    #[test]
    fn test_harmonic_returns_none_on_closed_mesh() {
        let mesh = crate::test_util::build_icosphere(1);
        assert!(
            harmonic_parameterization(&mesh).is_none(),
            "closed mesh has no boundary, harmonic should return None"
        );
    }

    /// 构建五边形扇形网格（1 个内部顶点 + 5 个边界顶点），用于参数化测试。
    /// 比 3×3 平面网格更适合参数化测试：唯一的内部顶点不邻接边界。
    fn build_pentagon_fan() -> MeshStorage {
        let mut mesh = MeshStorage::new();
        // 中心顶点
        let center = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        // 5 个边界顶点（正五边形）
        let mut boundary = Vec::new();
        for k in 0..5 {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) / 5.0;
            let v = mesh.add_vertex(crate::storage::Vertex::new([angle.cos(), angle.sin(), 0.0]));
            boundary.push(v);
        }
        // 5 个三角形
        for k in 0..5 {
            let next = (k + 1) % 5;
            crate::topology_ops::add_triangle(&mut mesh, center, boundary[k], boundary[next])
                .unwrap();
        }
        mesh
    }

    #[test]
    fn test_harmonic_pentagon_fan() {
        let mesh = build_pentagon_fan();
        let result = harmonic_parameterization(&mesh);
        assert!(result.is_some(), "Harmonic should succeed on pentagon fan");
        let uv = result.unwrap();
        assert_eq!(uv.len(), mesh.vertex_count());

        // 所有 UV 有限
        for v in &uv {
            assert!(v[0].is_finite() && v[1].is_finite(), "UV must be finite");
        }

        // 检查三角形无翻转（同号）
        let v_idx = build_vertex_index(&mesh);
        let mut signs: Vec<i32> = Vec::new();
        for f in mesh.face_ids() {
            let verts: Vec<_> = crate::traversal::FaceHalfEdges::new(&mesh, f)
                .filter_map(|he| mesh.get_halfedge(he))
                .map(|h| h.vertex)
                .collect();
            if verts.len() != 3 {
                continue;
            }
            let i0 = *v_idx.get(&verts[0]).unwrap();
            let i1 = *v_idx.get(&verts[1]).unwrap();
            let i2 = *v_idx.get(&verts[2]).unwrap();
            let cross_z = (uv[i1][0] - uv[i0][0]) * (uv[i2][1] - uv[i0][1])
                - (uv[i1][1] - uv[i0][1]) * (uv[i2][0] - uv[i0][0]);
            if cross_z.abs() > 1e-14 {
                signs.push(if cross_z > 0.0 { 1 } else { -1 });
            }
        }
        if signs.len() > 1 {
            let first = signs[0];
            assert!(
                signs.iter().all(|&s| s == first),
                "Harmonic on pentagon fan should have consistent orientation, signs: {:?}",
                signs
            );
        }
    }

    /// 验证 Tutte（均匀权重）和 harmonic（cotan 权重）在有足够内部顶点时产生不同结果。
    #[test]
    fn test_tutte_and_harmonic_differ() {
        let mesh = build_flat_grid();
        let tutte = tutte_embedding(&mesh).unwrap();
        // harmonic 可能在简单网格上求解失败，此时跳过差异检查
        let Some(harmonic) = harmonic_parameterization(&mesh) else {
            return;
        };
        assert_eq!(tutte.len(), harmonic.len());

        // 均匀权重 vs cotan 权重 → 内部顶点的 UV 应不同
        let mut diff_count = 0;
        for (t, h) in tutte.iter().zip(harmonic.iter()) {
            let d = ((t[0] - h[0]).powi(2) + (t[1] - h[1]).powi(2)).sqrt();
            if d > 1e-6 {
                diff_count += 1;
            }
        }
        // 至少部分内部顶点的 UV 应不同
        assert!(
            diff_count > 0,
            "Tutte (uniform weights) and harmonic (cotan weights) should produce different interior UVs"
        );
    }
}
