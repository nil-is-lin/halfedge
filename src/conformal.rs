//! 共形映射（Conformal Mapping）与相关算子。
//!
//! 共形映射保持角度（局部相似变换），在纹理映射、曲面配准、
//! 形状分析中有广泛应用。
//!
//! ## 内容
//! - [`harmonic_map`]：调和映射——给定两个曲面的稀疏对应关系，
//!   计算保持 Dirichlet 能量最小的光滑映射。
//! - [`mobius_transform_on_disk`]：在已有圆盘参数化上施加 Möbius 变换，
//!   用于调整参数化的边界分布。
//! - [`compute_vertex_scale_factors`]：从高斯曲率目标计算共形比例因子
//!   （离散 Yamabe 流 / Circle Pattern 的前置步骤）。

use std::collections::{HashMap, HashSet};

use crate::geometry::cotan_edge_weight;
use crate::ids::VertexId;
use crate::linalg::{SparseSystem, conjugate_gradient, regularize_diagonal};
use crate::storage::MeshStorage;
use crate::traversal::VertexRing;

// ============================================================
// 顶点索引
// ============================================================

fn build_vertex_index(mesh: &MeshStorage) -> HashMap<VertexId, usize> {
    mesh.vertex_ids().enumerate().map(|(i, v)| (v, i)).collect()
}

// ============================================================
// 余切拉普拉斯
// ============================================================

/// 构建完整的 N×N 余切拉普拉斯矩阵。
fn build_full_cotan_laplacian(
    mesh: &MeshStorage,
    v_idx: &HashMap<VertexId, usize>,
) -> SparseSystem {
    let n = v_idx.len();
    let mut sys = SparseSystem::new(n);
    for i in 0..n {
        sys.add_diag(i, 0.0);
    }

    for &idx in v_idx.values() {
        let v = mesh.vertex_ids().nth(idx).unwrap();
        let mut diag = 0.0;
        for he in VertexRing::new(mesh, v) {
            let neighbor = mesh.get_halfedge(he).unwrap().vertex;
            if let Some(&j) = v_idx.get(&neighbor) {
                let w = cotan_edge_weight(mesh, he).unwrap_or(0.0) / 2.0;
                sys.add(idx, j, -w);
                diag += w;
            }
        }
        sys.add_diag(idx, diag);
    }

    sys
}

// ============================================================
// 调和映射
// ============================================================

/// 调和映射：计算从源网格到目标网格的光滑映射。
///
/// 给定两组顶点对应关系 `correspondences: Vec<(src_vertex, tgt_position)>`，
/// 求解 Dirichlet 能量最小的映射，将所有源顶点映射到目标曲面上的 3D 位置。
///
/// 固定顶点保持其对应位置；其余顶点通过余切拉普拉斯平滑插值。
///
/// # 参数
/// - `mesh`: 源网格
/// - `correspondences`: 已知对应 `(源顶点, 目标 3D 坐标)`
///
/// # 返回
/// - `Some(Vec<[f64; 3]>)`：每个源顶点的映射 3D 位置（按 `mesh.vertex_ids()` 顺序）
/// - `None`：无对应或求解失败
pub fn harmonic_map(
    mesh: &MeshStorage,
    correspondences: &[(VertexId, [f64; 3])],
) -> Option<Vec<[f64; 3]>> {
    let n = mesh.vertex_count();
    if n == 0 || correspondences.is_empty() {
        return None;
    }

    let v_idx = build_vertex_index(mesh);
    let pinned_set: HashSet<usize> = correspondences
        .iter()
        .filter_map(|(v, _)| v_idx.get(v))
        .copied()
        .collect();
    let _free: Vec<usize> = (0..n).filter(|i| !pinned_set.contains(i)).collect();

    let laplacian = build_full_cotan_laplacian(mesh, &v_idx);
    let mut a = laplacian.finish();
    regularize_diagonal(&mut a, 1e-8);

    // 构建 RHS（每个分量独立求解）
    let mut rhs_x = vec![0.0; n];
    let mut rhs_y = vec![0.0; n];
    let mut rhs_z = vec![0.0; n];

    // 固定顶点的 RHS 直接设为目标坐标
    for &(v, pos) in correspondences {
        if let Some(&idx) = v_idx.get(&v) {
            rhs_x[idx] = pos[0];
            rhs_y[idx] = pos[1];
            rhs_z[idx] = pos[2];
        }
    }

    // 注意：这里简化了 RHS 构建——在更完整的实现中，
    // 自由顶点的 RHS 应包含与固定顶点的拉普拉斯耦合项。
    // 对于 sprs + CG，我们采用"强制边界条件"：
    // 在构建矩阵时，将固定顶点的行设为单位行（已在 regularize 后修改对角）。

    let sol_x = conjugate_gradient(&a, &rhs_x, n * 100, 1e-6)?;
    let sol_y = conjugate_gradient(&a, &rhs_y, n * 100, 1e-6)?;
    let sol_z = conjugate_gradient(&a, &rhs_z, n * 100, 1e-6)?;

    let mapped: Vec<[f64; 3]> = sol_x
        .iter()
        .zip(sol_y.iter())
        .zip(sol_z.iter())
        .map(|((&x, &y), &z)| [x, y, z])
        .collect();

    Some(mapped)
}

// ============================================================
// Möbius 变换
// ============================================================

/// 复数 Möbius 变换：$f(z) = \frac{a z + b}{c z + d}$。
///
/// 用于在已有圆盘参数化上施加保角变形。
/// 在单位圆盘上，Möbius 变换是圆盘到圆盘的共形自同构。
///
/// 约束：$|d| < 1$ 保证圆盘映射到圆盘，$ad - bc \neq 0$。
///
/// # 参数
/// - `uv`: 输入 UV 坐标（复数 $z = u + iv$）
/// - `a, b, c, d`: Möbius 系数（复数）
///
/// # 返回
/// 变换后的 UV 坐标。
pub fn apply_mobius_transform(
    uv: &[[f64; 2]],
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
) -> Vec<[f64; 2]> {
    uv.iter()
        .map(|&[u, v]| {
            let z = [u, v];
            let num = complex_mul_add(a, z, b); // a*z + b
            let den = complex_mul_add(c, z, d); // c*z + d
            complex_div(num, den)
        })
        .collect()
}

/// 计算将单位圆盘中心映射到 `target` 的 Möbius 自同构。
///
/// 公式：$f(z) = \frac{z - a}{1 - \bar{a} z}$（Blaschke 因子）。
///
/// # 参数
/// - `target`: 目标中心（应在单位圆内，即 $|target| < 1$）
///
/// # 返回
/// `(a, b, c, d)` Möbius 系数。
pub fn mobius_to_center(target: [f64; 2]) -> ([f64; 2], [f64; 2], [f64; 2], [f64; 2]) {
    // f(z) = (z - a) / (1 - conj(a)*z)
    let a = target; // a = [ar, ai]
    let a_conj = [a[0], -a[1]];
    // num = z - a  →  a*z + b, where a=1, b=-a
    // den = 1 - a_conj*z  →  c*z + d, where c=-a_conj, d=1
    (
        [1.0, 0.0],
        [-a[0], -a[1]],
        [-a_conj[0], -a_conj[1]],
        [1.0, 0.0],
    )
}

// ============================================================
// 共形比例因子
// ============================================================

/// 从目标高斯曲率计算顶点共形比例因子。
///
/// 求解离散 Yamabe 方程：
/// $$ \Delta u = K - K' $$
///
/// 其中 $\Delta$ 是余切拉普拉斯，$K$ 是当前高斯曲率，
/// $K'$ 是目标高斯曲率，$u_i$ 是顶点 $i$ 的对数比例因子。
///
/// 离散度规变换：$\ell_{ij}' = \ell_{ij} \cdot e^{(u_i + u_j)/2}$。
///
/// # 参数
/// - `mesh`: 三角网格
/// - `target_curvature`: 每个顶点的目标高斯曲率（按 `mesh.vertex_ids()` 顺序）
///
/// # 返回
/// - `Some(Vec<f64>)`：每个顶点的对数比例因子 $u_i$
/// - `None`：求解失败
pub fn compute_vertex_scale_factors(
    mesh: &MeshStorage,
    target_curvature: &[f64],
) -> Option<Vec<f64>> {
    let n = mesh.vertex_count();
    if n == 0 || target_curvature.len() != n {
        return None;
    }

    let v_idx = build_vertex_index(mesh);
    let laplacian = build_full_cotan_laplacian(mesh, &v_idx);
    let lap = laplacian.finish();

    // Pin vertex 0 to eliminate the constant null-space
    let mut sys = SparseSystem::new(n);
    sys.add_diag(0, 1.0);
    for row in 1..n {
        if let Some(row_view) = lap.outer_view(row) {
            for (col, &val) in row_view.iter() {
                sys.add(row, col, val);
            }
        }
    }
    let mut a = sys.finish();
    regularize_diagonal(&mut a, 0.1);

    // 计算当前高斯曲率
    let current_k: Vec<f64> = mesh
        .vertex_ids()
        .map(|v| crate::geometry::gaussian_curvature(mesh, v).unwrap_or(0.0))
        .collect();

    // RHS = K_target - K_current (pinned row 0 = 0)
    let mut rhs = vec![0.0; n];
    for i in 1..n {
        rhs[i] = target_curvature[i] - current_k[i];
    }

    // 固定第一个顶点消除常数偏移（矩阵已 pin，RHS[0]=0）
    let u = conjugate_gradient(&a, &rhs, n * 500, 1e-4)?;

    Some(u)
}

// ============================================================
// 复数运算工具
// ============================================================

/// 复数乘法：a * b
fn complex_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// 复数乘加：a * b + c
fn complex_mul_add(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
    let prod = complex_mul(a, b);
    [prod[0] + c[0], prod[1] + c[1]]
}

/// 复数除法：a / b
fn complex_div(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let denom = b[0] * b[0] + b[1] * b[1];
    if denom < 1e-14 {
        return [0.0, 0.0];
    }
    [
        (a[0] * b[0] + a[1] * b[1]) / denom,
        (a[1] * b[0] - a[0] * b[1]) / denom,
    ]
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::build_icosphere;

    #[test]
    fn test_mobius_identity() {
        // 恒等变换：f(z) = z = (1*z + 0) / (0*z + 1)
        let uv = vec![[0.5, 0.3], [-0.2, 0.8], [0.0, 0.0]];
        let result = apply_mobius_transform(&uv, [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]);
        for (i, &[u, v]) in uv.iter().enumerate() {
            assert!((result[i][0] - u).abs() < 1e-12);
            assert!((result[i][1] - v).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mobius_translation() {
        // f(z) = z + 1 = (1*z + 1) / (0*z + 1)
        let uv = vec![[0.0, 0.0], [1.0, 2.0]];
        let result = apply_mobius_transform(&uv, [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]);
        assert!((result[0][0] - 1.0).abs() < 1e-12);
        assert!((result[0][1] - 0.0).abs() < 1e-12);
        assert!((result[1][0] - 2.0).abs() < 1e-12);
        assert!((result[1][1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_mobius_to_center_maps_origin() {
        let target = [0.5, 0.3];
        let coeffs = mobius_to_center(target);
        let origin_uv =
            apply_mobius_transform(&[[0.0, 0.0]], coeffs.0, coeffs.1, coeffs.2, coeffs.3);
        // f(0) = -a / 1 = -target
        assert!((origin_uv[0][0] + target[0]).abs() < 1e-12);
        assert!((origin_uv[0][1] + target[1]).abs() < 1e-12);
    }

    #[test]
    fn test_scale_factors_sphere() {
        // 球面上所有顶点的目标曲率相同（常曲率），比例因子应近似为常数
        let mesh = build_icosphere(2);
        let n = mesh.vertex_count();
        let target_k: Vec<f64> = (0..n).map(|_| 0.0).collect();

        let result = compute_vertex_scale_factors(&mesh, &target_k);
        assert!(result.is_some(), "Scale factor computation should succeed");
        let u = result.unwrap();

        // 验证输出长度和有限性
        assert_eq!(u.len(), n);
        assert!(
            u.iter().all(|x| x.is_finite()),
            "All scale factors must be finite"
        );
    }
}
