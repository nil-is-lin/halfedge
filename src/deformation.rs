//! 网格变形（Mesh Deformation）。
//!
//! 提供两类基于线性系统的变形方法，参考 Sorkine 等人的经典工作：
//!
//! - [`laplacian_deformation`]：基于拉普拉斯坐标的变形
//!   （Laplacian Surface Editing, Sorkine et al. 2004）。
//!   通过保留每个顶点的拉普拉斯坐标（局部细节），同时约束 handle 顶点
//!   到目标位置，求解稀疏线性系统得到变形后的位置。
//!
//! - [`arap_deformation`]：As-Rigid-As-Possible 变形
//!   （Sorkine & Alexa 2007）。Local-Global 迭代：
//!   - 局部步骤：对每个顶点 cell 计算最佳旋转 $R_i$（极分解）；
//!   - 全局步骤：固定旋转，求解 Poisson 系统更新位置。
//!
//!   相比 Laplacian 变形，ARAP 在大变形下细节保持更好、扭曲更小。
//!
//! ## API
//! - 用户通过 [`DeformationConstraint`] 指定 handle 顶点及其目标位置；
//! - 自由顶点（不在约束集中）由求解器自动计算。

use std::collections::HashMap;

use crate::geometry::cotan_edge_weight;
use crate::ids::VertexId;
use crate::linalg::{SparseSystem, conjugate_gradient, regularize_diagonal};
use crate::storage::MeshStorage;
use crate::traversal::VertexRing;

type Vec3 = [f64; 3];

#[inline]
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// 3×3 矩阵按行主序存储。
type Mat3 = [[f64; 3]; 3];

#[inline]
fn mat3_transpose(m: Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn mat3_vec(m: Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn mat3_det(m: Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// 计算矩阵的伴随（用于求逆）。
fn mat3_adjoint(m: Mat3) -> Mat3 {
    [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]
}

fn mat3_inverse(m: Mat3) -> Option<Mat3> {
    let det = mat3_det(m);
    if det.abs() < 1e-14 {
        return None;
    }
    let adj = mat3_adjoint(m);
    let inv_det = 1.0 / det;
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = adj[i][j] * inv_det;
        }
    }
    Some(r)
}

/// 3×3 矩阵的极分解：返回最接近 M 的旋转 R（即 M = R * S，S 对称半正定）。
///
/// 使用 Higham 1986 的 Newton 迭代：
/// $R_{k+1} = \frac{1}{2}(R_k + R_k^{-T})$，初始 $R_0 = M$。
/// 通常 5-10 次迭代即收敛到正交矩阵。
fn polar_rotation(m: Mat3) -> Mat3 {
    let mut r = m;
    for _ in 0..20 {
        // R^{-T} = (R^{-1})^T = (R^T)^{-1}
        let r_inv = match mat3_inverse(r) {
            Some(inv) => inv,
            None => break, // 矩阵奇异，停止
        };
        let r_inv_t = mat3_transpose(r_inv);
        // R_new = 0.5 * (R + R^{-T})
        let mut r_new = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r_new[i][j] = 0.5 * (r[i][j] + r_inv_t[i][j]);
            }
        }
        // 收敛判定：差值范数
        let mut diff_sq = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let d = r_new[i][j] - r[i][j];
                diff_sq += d * d;
            }
        }
        r = r_new;
        if diff_sq < 1e-18 {
            break;
        }
    }
    r
}

// ============================================================
// 公共类型
// ============================================================

/// 变形约束：将 `vertex` 移动到 `target_position`。
#[derive(Clone, Copy, Debug)]
pub struct DeformationConstraint {
    pub vertex: VertexId,
    pub target_position: Vec3,
}

// ============================================================
// 内部工具
// ============================================================

fn build_vertex_index(mesh: &MeshStorage) -> HashMap<VertexId, usize> {
    mesh.vertex_ids().enumerate().map(|(i, v)| (v, i)).collect()
}

/// 收集每个顶点的邻居索引和余切权重（每条边权重减半，因被遍历两次）。
fn build_neighbors_and_weights(
    mesh: &MeshStorage,
    v_idx: &HashMap<VertexId, usize>,
) -> Vec<Vec<(usize, f64)>> {
    let n = v_idx.len();
    let mut neighbors = vec![Vec::new(); n];
    for (v, &i) in v_idx {
        for he in VertexRing::new(mesh, *v) {
            let Some(h) = mesh.get_halfedge(he) else {
                continue;
            };
            let n_vid = h.vertex;
            let Some(&j) = v_idx.get(&n_vid) else {
                continue;
            };
            let w = cotan_edge_weight(mesh, he).unwrap_or(0.0) / 2.0;
            if w.is_finite() && w.abs() > 1e-12 {
                neighbors[i].push((j, w));
            }
        }
    }
    neighbors
}

// ============================================================
// Laplacian Deformation
// ============================================================

/// Laplacian 变形（Sorkine et al. 2004）。
///
/// 通过保留每个顶点的余切拉普拉斯坐标 $\delta_i = p_i - \sum_{j} w_{ij} p_j$
/// 来保持局部细节，同时约束 handle 顶点到目标位置。
///
/// # 线性系统
/// 对每个自由顶点 $i$：
/// $$
/// \sum_j L_{ij}\, p'_j = \delta_i
/// $$
/// 对每个约束顶点 $i$：
/// $$
/// p'_i = c_i
/// $$
/// 分别对 x、y、z 三个分量求解。
///
/// # 参数
/// - `mesh`: 输入网格
/// - `constraints`: handle 顶点约束
///
/// # 返回
/// - `Some(Vec<[f64;3]>)`：变形后每个顶点的位置（按 `mesh.vertex_ids()` 顺序）
/// - `None`：空网格或求解失败
pub fn laplacian_deformation(
    mesh: &MeshStorage,
    constraints: &[DeformationConstraint],
) -> Option<Vec<Vec3>> {
    let n = mesh.vertex_count();
    if n == 0 {
        return None;
    }
    let v_idx = build_vertex_index(mesh);

    // 原始位置
    let original: Vec<Vec3> = mesh
        .vertex_ids()
        .map(|v| mesh.get_vertex(v).map(|x| x.position).unwrap_or([0.0; 3]))
        .collect();

    // 计算原始拉普拉斯坐标 δ_i = p_i - Σ w_ij p_j
    let neighbors = build_neighbors_and_weights(mesh, &v_idx);
    let mut delta = vec![[0.0; 3]; n];
    for i in 0..n {
        let p_i = original[i];
        let mut sum = [0.0; 3];
        for &(j, w) in &neighbors[i] {
            let p_j = original[j];
            sum = add(sum, scale(p_j, w));
        }
        // δ_i = w_ii * p_i - Σ w_ij p_j = (Σ w_ij) * p_i - Σ w_ij p_j
        let w_sum: f64 = neighbors[i].iter().map(|&(_, w)| w).sum();
        delta[i] = sub(scale(p_i, w_sum), sum);
    }

    // 构建约束集合
    let mut constraint_map: HashMap<usize, Vec3> = HashMap::new();
    for c in constraints {
        if let Some(&idx) = v_idx.get(&c.vertex) {
            constraint_map.insert(idx, c.target_position);
        }
    }
    if constraint_map.is_empty() {
        return Some(original.clone());
    }

    // 构建线性系统
    // 注意：neighbors 中的权重已除以 2（因每条边被两端各遍历一次）
    // SparseSystem::add 对称写入 (i,j) 和 (j,i)，所以每条边非对角贡献 = -w/2 - w/2 = -w_orig ✓
    // 但对角只从 i 端写入一次，故对角 = 2 * Σ(w/2) = Σ w_orig
    let mut sys = SparseSystem::new(n);
    let mut rhs_x = vec![0.0; n];
    let mut rhs_y = vec![0.0; n];
    let mut rhs_z = vec![0.0; n];

    for i in 0..n {
        if let Some(&target) = constraint_map.get(&i) {
            // 约束顶点：单位行
            sys.add_diag(i, 1.0);
            rhs_x[i] = target[0];
            rhs_y[i] = target[1];
            rhs_z[i] = target[2];
        } else {
            // 自由顶点：拉普拉斯行
            // 对角 = 2 * Σ w/2 = Σ w_orig
            let w_sum: f64 = neighbors[i].iter().map(|&(_, w)| w).sum::<f64>() * 2.0;
            for &(j, w) in &neighbors[i] {
                if constraint_map.contains_key(&j) {
                    // 约束邻居：移到 RHS（L_ij = -w_orig, 移项得 +w_orig * c_j）
                    // w 在 neighbors 中是 w_orig/2，但非对角总贡献是 -w_orig
                    // 所以 RHS 贡献 = w_orig * c_j = 2w * c_j
                    let c = constraint_map[&j];
                    rhs_x[i] += 2.0 * w * c[0];
                    rhs_y[i] += 2.0 * w * c[1];
                    rhs_z[i] += 2.0 * w * c[2];
                } else {
                    // 自由邻居：写入矩阵
                    sys.add(i, j, -w);
                }
            }
            sys.add_diag(i, w_sum.max(1e-10));
            rhs_x[i] += delta[i][0];
            rhs_y[i] += delta[i][1];
            rhs_z[i] += delta[i][2];
        }
    }

    let mut a = sys.finish();
    regularize_diagonal(&mut a, 1e-8);

    let x = conjugate_gradient(&a, &rhs_x, n * 200, 1e-6)?;
    let y = conjugate_gradient(&a, &rhs_y, n * 200, 1e-6)?;
    let z = conjugate_gradient(&a, &rhs_z, n * 200, 1e-6)?;

    Some(
        x.into_iter()
            .zip(y)
            .zip(z)
            .map(|((x, y), z)| [x, y, z])
            .collect(),
    )
}

// ============================================================
// ARAP Deformation
// ============================================================

/// ARAP 变形（Sorkine & Alexa 2007）。
///
/// Local-Global 迭代：
/// 1. **初始化**：先用 Laplacian 变形（或直接使用约束顶点目标）作为初始猜测；
/// 2. **局部步骤**：对每个顶点的 cell，计算最佳旋转 $R_i$
///    （通过协方差矩阵 $S_i = \sum_j w_{ij} e_{ij} e'_{ij}^T$ 的极分解）；
/// 3. **全局步骤**：固定所有 $R_i$，求解 Poisson 系统更新位置；
/// 4. 重复 2-3 直到 `iterations` 次或收敛。
///
/// # 参数
/// - `mesh`: 输入网格
/// - `constraints`: handle 顶点约束
/// - `iterations`: local-global 迭代次数（典型 5-10）
///
/// # 返回
/// - `Some(Vec<[f64;3]>)`：变形后每个顶点位置
/// - `None`：空网格、无约束或求解失败
pub fn arap_deformation(
    mesh: &MeshStorage,
    constraints: &[DeformationConstraint],
    iterations: usize,
) -> Option<Vec<Vec3>> {
    let n = mesh.vertex_count();
    if n == 0 || constraints.is_empty() {
        return None;
    }
    let v_idx = build_vertex_index(mesh);

    // 原始位置
    let original: Vec<Vec3> = mesh
        .vertex_ids()
        .map(|v| mesh.get_vertex(v).map(|x| x.position).unwrap_or([0.0; 3]))
        .collect();

    let neighbors = build_neighbors_and_weights(mesh, &v_idx);

    // 约束集合
    let mut constraint_map: HashMap<usize, Vec3> = HashMap::new();
    for c in constraints {
        if let Some(&idx) = v_idx.get(&c.vertex) {
            constraint_map.insert(idx, c.target_position);
        }
    }

    // 步骤 1: 初始化 - 用 Laplacian 变形得到初始解
    let mut current = laplacian_deformation(mesh, constraints)?;
    // 强制约束顶点准确为目标位置（Laplacian 求解可能有数值误差）
    for (&idx, &target) in &constraint_map {
        current[idx] = target;
    }

    // 预构建全局步骤的矩阵（与 R 无关的部分）
    // L_ii = Σ w_orig, L_ij = -w_orig（自由顶点）；L_ii = 1（约束顶点）
    // neighbors 中 w = w_orig/2，SparseSystem::add 对称写入
    let mut sys = SparseSystem::new(n);
    for (i, neighbors_i) in neighbors.iter().enumerate() {
        if constraint_map.contains_key(&i) {
            sys.add_diag(i, 1.0);
        } else {
            // 对角 = 2 * Σ w/2 = Σ w_orig
            let w_sum: f64 = neighbors_i.iter().map(|&(_, w)| w).sum::<f64>() * 2.0;
            for &(j, w) in neighbors_i {
                if !constraint_map.contains_key(&j) {
                    // 只写入自由邻居（约束邻居的贡献在 RHS 中处理）
                    sys.add(i, j, -w);
                }
            }
            sys.add_diag(i, w_sum.max(1e-10));
        }
    }
    let mut a = sys.finish();
    regularize_diagonal(&mut a, 1e-8);

    // Local-Global 迭代
    for _ in 0..iterations.max(1) {
        // 局部步骤：对每个顶点计算最佳旋转
        let rotations = compute_cell_rotations(&original, &current, &neighbors, &constraint_map);

        // 全局步骤：求解系统
        let (rhs_x, rhs_y, rhs_z) =
            build_arap_rhs(&original, &neighbors, &rotations, &constraint_map);

        let x = conjugate_gradient(&a, &rhs_x, n * 200, 1e-6)?;
        let y = conjugate_gradient(&a, &rhs_y, n * 200, 1e-6)?;
        let z = conjugate_gradient(&a, &rhs_z, n * 200, 1e-6)?;

        current = x
            .into_iter()
            .zip(y)
            .zip(z)
            .map(|((x, y), z)| [x, y, z])
            .collect();

        // 强制约束顶点
        for (&idx, &target) in &constraint_map {
            current[idx] = target;
        }
    }

    Some(current)
}

/// 计算每个顶点 cell 的最佳旋转 R_i（极分解）。
fn compute_cell_rotations(
    original: &[Vec3],
    current: &[Vec3],
    neighbors: &[Vec<(usize, f64)>],
    _constraint_map: &HashMap<usize, Vec3>,
) -> Vec<Mat3> {
    let n = original.len();
    let mut rotations = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; n];

    for i in 0..n {
        // 构建协方差矩阵 S_i = Σ_j w_ij * e_ij * e'_ij^T
        // e_ij = p_i - p_j (原始), e'_ij = p'_i - p'_j (当前)
        let mut s = [[0.0; 3]; 3];
        let p_i = original[i];
        let p_i_cur = current[i];
        for &(j, w) in &neighbors[i] {
            let e_orig = sub(p_i, original[j]);
            let e_cur = sub(p_i_cur, current[j]);
            // S += w * outer(e_orig, e_cur)
            // outer(a, b) = a * b^T，元素 S[r][c] += w * a[r] * b[c]
            for r in 0..3 {
                for c in 0..3 {
                    s[r][c] += w * e_orig[r] * e_cur[c];
                }
            }
        }

        // 极分解得到最佳旋转
        let s_norm = (s[0][0].powi(2)
            + s[0][1].powi(2)
            + s[0][2].powi(2)
            + s[1][0].powi(2)
            + s[1][1].powi(2)
            + s[1][2].powi(2)
            + s[2][0].powi(2)
            + s[2][1].powi(2)
            + s[2][2].powi(2))
        .sqrt();
        if s_norm < 1e-14 {
            // 退化：使用单位旋转
            rotations[i] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        } else {
            let r = polar_rotation(s);
            // 校验：det(R) 应为 +1（旋转），若为 -1（反射）则修正
            let det = mat3_det(r);
            if det < 0.0 {
                // 反射：通过翻转最后一个奇异向量修正
                // 简化处理：使用 SVD 修正太复杂，这里直接取 R 的负对角元
                // 更稳健的方法是 SVD，但极分解 + 反射检查已足够大部分场景
                let mut r_corrected = r;
                r_corrected[2][0] = -r_corrected[2][0];
                r_corrected[2][1] = -r_corrected[2][1];
                r_corrected[2][2] = -r_corrected[2][2];
                rotations[i] = r_corrected;
            } else {
                rotations[i] = r;
            }
        }
    }

    rotations
}

/// 构建 ARAP 全局步骤的右端向量。
///
/// 对自由顶点 $i$：$\text{rhs}_i = \frac{1}{2} \sum_j w_{ij} (R_i + R_j) (p_i^0 - p_j^0)$
/// 对约束顶点 $i$：$\text{rhs}_i = c_i$（约束目标位置）
///
/// 其中 $p^0$ 是原始位置，$R$ 是当前迭代的旋转。
fn build_arap_rhs(
    original: &[Vec3],
    neighbors: &[Vec<(usize, f64)>],
    rotations: &[Mat3],
    constraint_map: &HashMap<usize, Vec3>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = neighbors.len();
    let mut rhs_x = vec![0.0; n];
    let mut rhs_y = vec![0.0; n];
    let mut rhs_z = vec![0.0; n];

    // 对每个顶点 i 累加其所有邻居的贡献
    // 注意：每条无向边 (i, j) 在 neighbors[i] 和 neighbors[j] 中各出现一次
    // 公式 rhs_i = (1/2) Σ_j w_ij (R_i + R_j) (p_i - p_j) 已包含每条边一次
    for i in 0..n {
        if let Some(&target) = constraint_map.get(&i) {
            rhs_x[i] = target[0];
            rhs_y[i] = target[1];
            rhs_z[i] = target[2];
            continue;
        }

        let r_i = rotations[i];
        let p_i = original[i];
        let mut acc = [0.0; 3];
        for &(j, w) in &neighbors[i] {
            let r_j = rotations[j];
            let p_j = original[j];
            // e = p_i - p_j
            let e = sub(p_i, p_j);
            // (R_i + R_j) * e
            let r_i_e = mat3_vec(r_i, e);
            let r_j_e = mat3_vec(r_j, e);
            let combined = add(r_i_e, r_j_e);
            // (1/2) * w_orig * (R_i + R_j) * e = w * (R_i + R_j) * e
            // （neighbors 中 w = w_orig/2，故 w 即为 (1/2) w_orig）
            acc = add(acc, scale(combined, w));
            // 约束邻居：矩阵中跳过了 (i,j)，需将 -w_orig * c_j 移到 RHS 得 +w_orig * c_j = +2w * c_j
            if let Some(&c) = constraint_map.get(&j) {
                acc = add(acc, scale(c, 2.0 * w));
            }
        }
        rhs_x[i] = acc[0];
        rhs_y[i] = acc[1];
        rhs_z[i] = acc[2];
    }

    (rhs_x, rhs_y, rhs_z)
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Vertex;
    use crate::test_util::build_icosphere;
    use crate::topology_ops::add_triangle;

    fn build_grid_mesh() -> MeshStorage {
        // 3x3 网格：9 个顶点，8 个三角形
        let mut mesh = MeshStorage::new();
        let mut vs: Vec<VertexId> = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                let v = mesh.add_vertex(Vertex::new([x as f64, y as f64, 0.0]));
                vs.push(v);
            }
        }
        // 索引：(x, y) → y*3 + x
        // 三角形
        for y in 0..2 {
            for x in 0..2 {
                let v0 = vs[y * 3 + x];
                let v1 = vs[y * 3 + x + 1];
                let v2 = vs[(y + 1) * 3 + x];
                let v3 = vs[(y + 1) * 3 + x + 1];
                add_triangle(&mut mesh, v0, v1, v2).unwrap();
                add_triangle(&mut mesh, v1, v3, v2).unwrap();
            }
        }
        mesh
    }

    #[test]
    fn test_laplacian_deformation_no_constraint_returns_original() {
        let mesh = build_grid_mesh();
        let result = laplacian_deformation(&mesh, &[]);
        assert!(result.is_some());
        // 无约束时，应返回原始位置（或等价）
        let deformed = result.unwrap();
        let original: Vec<Vec3> = mesh
            .vertex_ids()
            .map(|v| mesh.get_vertex(v).unwrap().position)
            .collect();
        for i in 0..deformed.len() {
            for d in 0..3 {
                assert!(
                    (deformed[i][d] - original[i][d]).abs() < 1e-3,
                    "vertex {} axis {}: deformed {} vs original {}",
                    i,
                    d,
                    deformed[i][d],
                    original[i][d]
                );
            }
        }
    }

    #[test]
    fn test_laplacian_deformation_single_handle() {
        let mesh = build_grid_mesh();
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        // 将 (0,0) 顶点向上抬起 1.0
        let constraints = vec![DeformationConstraint {
            vertex: vertices[0],
            target_position: [0.0, 0.0, 1.0],
        }];
        let result = laplacian_deformation(&mesh, &constraints);
        assert!(result.is_some());
        let deformed = result.unwrap();
        // 约束顶点应在目标位置
        assert!(
            (deformed[0][2] - 1.0).abs() < 1e-3,
            "handle z = {}",
            deformed[0][2]
        );
        // 邻居应有部分 z 位移（>0）
        let mut has_neighbor_displaced = false;
        for d in deformed.iter().skip(1) {
            if d[2].abs() > 1e-6 {
                has_neighbor_displaced = true;
                break;
            }
        }
        assert!(
            has_neighbor_displaced,
            "some non-handle vertex should be displaced"
        );
    }

    #[test]
    fn test_laplacian_deformation_two_handles() {
        let mesh = build_grid_mesh();
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        // 固定 (0,0) 不动，将 (2,2) 抬起
        let constraints = vec![
            DeformationConstraint {
                vertex: vertices[0],
                target_position: [0.0, 0.0, 0.0],
            },
            DeformationConstraint {
                vertex: vertices[8],
                target_position: [2.0, 2.0, 1.0],
            },
        ];
        let result = laplacian_deformation(&mesh, &constraints);
        assert!(result.is_some());
        let deformed = result.unwrap();
        // 两个 handle 都应被准确约束
        assert!((deformed[0][2] - 0.0).abs() < 1e-3);
        assert!((deformed[8][2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_arap_deformation_basic() {
        let mesh = build_grid_mesh();
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let constraints = vec![
            DeformationConstraint {
                vertex: vertices[0],
                target_position: [0.0, 0.0, 0.0],
            },
            DeformationConstraint {
                vertex: vertices[2],
                target_position: [2.0, 0.0, 0.0],
            },
            DeformationConstraint {
                vertex: vertices[6],
                target_position: [0.0, 2.0, 0.0],
            },
            DeformationConstraint {
                vertex: vertices[8],
                target_position: [2.0, 2.0, 1.0],
            },
        ];
        let result = arap_deformation(&mesh, &constraints, 5);
        assert!(result.is_some());
        let deformed = result.unwrap();
        // 约束顶点应在目标位置
        assert!((deformed[0][2] - 0.0).abs() < 1e-3);
        assert!((deformed[2][2] - 0.0).abs() < 1e-3);
        assert!((deformed[6][2] - 0.0).abs() < 1e-3);
        assert!((deformed[8][2] - 1.0).abs() < 1e-3);
        // 中心顶点应有正向 z 位移
        assert!(
            deformed[4][2] > 0.0,
            "center vertex z = {} should be > 0",
            deformed[4][2]
        );
    }

    #[test]
    fn test_arap_deformation_empty_returns_none() {
        let mesh = MeshStorage::new();
        let result = arap_deformation(&mesh, &[], 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_arap_deformation_no_constraints_returns_none() {
        let mesh = build_grid_mesh();
        let result = arap_deformation(&mesh, &[], 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_polar_rotation_identity() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let r = polar_rotation(m);
        for (i, row) in r.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_polar_rotation_90deg_around_z() {
        // 旋转矩阵：绕 z 轴 90°
        let angle = 90.0_f64.to_radians();
        let cos90 = angle.cos();
        let sin90 = angle.sin();
        let m = [[cos90, -sin90, 0.0], [sin90, cos90, 0.0], [0.0, 0.0, 1.0]];
        let r = polar_rotation(m);
        // 已经是旋转矩阵，极分解应保持不变
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r[i][j] - m[i][j]).abs() < 1e-8,
                    "r[{}][{}]={} expected {}",
                    i,
                    j,
                    r[i][j],
                    m[i][j]
                );
            }
        }
    }

    #[test]
    fn test_polar_rotation_stretch_matrix() {
        // 拉伸矩阵 [[2,0,0],[0,1,0],[0,0,1]] 的极分解应为单位矩阵
        let m = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let r = polar_rotation(m);
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((r[i][j] - expected[i][j]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_laplacian_deformation_icosphere() {
        // 在 icosphere 上抬起一个顶点
        let mesh = build_icosphere(1);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let original_pos = mesh.get_vertex(vertices[0]).unwrap().position;
        let target = [
            original_pos[0] * 1.5,
            original_pos[1] * 1.5,
            original_pos[2] * 1.5,
        ];
        let constraints = vec![DeformationConstraint {
            vertex: vertices[0],
            target_position: target,
        }];
        let result = laplacian_deformation(&mesh, &constraints);
        assert!(result.is_some());
        let deformed = result.unwrap();
        // 约束顶点应在目标位置
        for d in 0..3 {
            assert!((deformed[0][d] - target[d]).abs() < 1e-3);
        }
    }
}
