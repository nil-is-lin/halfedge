//! 稀疏线性代数工具。
//!
//! 在 [`sprs`] 的 [`CsMat`](sprs::CsMat) 稀疏矩阵之上提供：
//! - [`SparseSystem`]：从网格拉普拉斯矩阵的三元组构建稀疏对称系统；
//! - [`conjugate_gradient`]：共轭梯度法求解 $Ax = b$（A 对称正定）；
//! - [`conjugate_gradient_preconditioned`]：预条件共轭梯度法（Jacobi 对角预条件）；
//! - [`jacobi_preconditioner`]：构建 Jacobi 预条件向量；
//! - 辅助工具：正则化对角偏移、残差范数计算。
//!
//! ## 使用场景
//! - 参数化：求解调和 / LSCM / Tutte 线性系统
//! - 测地线：Heat Method 中的两次 Poisson 求解
//! - 共形映射：离散调和映射

use sprs::{CsMat, TriMat};

// ============================================================
// 稀疏系统构建器
// ============================================================

/// 正在构建中的稀疏对称线性系统。
///
/// 内部用 [`TriMat`] 收集三元组，调用 [`finish`](SparseSystem::finish) 后转为
/// CSR 格式的 [`CsMat`]，可直接用于共轭梯度求解。
pub struct SparseSystem {
    tri: TriMat<f64>,
    dim: usize,
}

impl SparseSystem {
    /// 新建一个 `dim × dim` 的空稀疏系统。
    pub fn new(dim: usize) -> Self {
        Self {
            tri: TriMat::new((dim, dim)),
            dim,
        }
    }

    /// 添加矩阵元素 `matrix[i][j] += val`（同时写入 `matrix[j][i]` 保持对称）。
    ///
    /// 若 `i == j` 则只累加一次对角元。
    pub fn add(&mut self, i: usize, j: usize, val: f64) {
        if i < self.dim && j < self.dim {
            self.tri.add_triplet(i, j, val);
            if i != j {
                self.tri.add_triplet(j, i, val);
            }
        }
    }

    /// 仅累加对角元 `matrix[i][i] += val`。
    pub fn add_diag(&mut self, i: usize, val: f64) {
        if i < self.dim {
            self.tri.add_triplet(i, i, val);
        }
    }

    /// 完成构建，转为 CSR 稀疏矩阵。
    pub fn finish(self) -> CsMat<f64> {
        self.tri.to_csr()
    }

    /// 矩阵维度。
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ============================================================
// 网格拉普拉斯构建
// ============================================================

/// 构建顶点索引映射 `HashMap<VertexId, usize>`。
///
/// 按 `mesh.vertex_ids()` 的遍历顺序为每个顶点分配连续索引。
/// 这是所有拉普拉斯/线性系统构建的公共前置步骤。
pub fn build_vertex_index(
    mesh: &crate::storage::MeshStorage,
) -> std::collections::HashMap<crate::ids::VertexId, usize> {
    mesh.vertex_ids().enumerate().map(|(i, v)| (v, i)).collect()
}

/// 构建对称余切拉普拉斯稀疏系统。
///
/// 遍历每个顶点的邻域，用 [`cotan_edge_weight`](crate::geometry::cotan_edge_weight)
/// 计算边权重（每边除以 2 以补偿双向遍历），组装为 `SparseSystem`。
///
/// 对角线 = 该顶点所有出边权重之和；非对角线 = 负的边权重。
///
/// # 参数
/// - `mesh`: 网格存储
/// - `v_idx`: 顶点到矩阵索引的映射（由 [`build_vertex_index`] 构建）
///
/// # 返回
/// `SparseSystem`，调用 `.finish()` 后得到 CSR 矩阵。
pub fn build_cotan_laplacian(
    mesh: &crate::storage::MeshStorage,
    v_idx: &std::collections::HashMap<crate::ids::VertexId, usize>,
) -> SparseSystem {
    use crate::geometry::cotan_edge_weight;
    use crate::traversal::VertexRing;

    let n = v_idx.len();
    let mut sys = SparseSystem::new(n);

    for (&v, &i) in v_idx {
        let mut diag = 0.0;
        for he in VertexRing::new(mesh, v) {
            let neighbor = match mesh.get_halfedge(he) {
                Some(h) => h.vertex,
                None => continue,
            };
            if let Some(&j) = v_idx.get(&neighbor) {
                let w = cotan_edge_weight(mesh, he).unwrap_or(0.0) / 2.0;
                sys.add(i, j, -w);
                diag += w;
            }
        }
        sys.add_diag(i, diag);
    }

    sys
}

// ============================================================
// 共轭梯度法
// ============================================================

/// 共轭梯度法求解 $Ax = b$。
///
/// **前置条件：** $A$ 必须为对称正定（SPD）矩阵。若 $A$ 半正定（如拉普拉斯矩阵），
/// 调用方应先做正则化（对 $A$ 的对角线加一个小偏移）。
///
/// 内部委托给 [`conjugate_gradient_preconditioned`]，使用单位预条件（全 1），
/// 因此等价于无预条件的标准 CG。
///
/// # 参数
/// - `a`: 稀疏 SPD 矩阵（CSR 格式）
/// - `b`: 右端向量（长度 = `a.rows()`）
/// - `max_iter`: 最大迭代次数
/// - `tol`: 残差相对下降容差（$\|r_k\| / \|b\| < tol$ 时停止）
///
/// # 返回
/// - `Some(x)`：近似解
/// - `None`：达到最大迭代次数仍未收敛
pub fn conjugate_gradient(
    a: &CsMat<f64>,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> Option<Vec<f64>> {
    let n = a.rows();
    if n != b.len() {
        return None;
    }
    // 无预条件 = 单位预条件向量
    let preconditioner = vec![1.0; n];
    conjugate_gradient_preconditioned(a, b, &preconditioner, max_iter, tol)
}

/// 预条件共轭梯度法求解 $Ax = b$。
///
/// 使用 Jacobi（对角）预条件矩阵 $M^{-1}$ 加速收敛。算法在每步用
/// $z = M^{-1} r$ 代替 $r$ 来构造搜索方向，对病态系统（如不规则网格上的
/// 余切拉普拉斯矩阵）可显著减少迭代次数。
///
/// **前置条件：** $A$ 必须为对称正定（SPD）矩阵，且 `preconditioner` 长度
/// 等于 `a.rows()`，每个元素为 $M^{-1}$ 的对角元（即 $1 / A_{ii}$）。
///
/// # 算法
/// 1. $r_0 = b - Ax_0$（初始 $x_0 = 0$，故 $r_0 = b$）
/// 2. $z_0 = M^{-1} r_0$（逐元素乘）
/// 3. $p_0 = z_0$
/// 4. 每次迭代：
///    - $Ap = A \cdot p$
///    - $\alpha = (r \cdot z) / (p \cdot Ap)$
///    - $x \mathrel{+}= \alpha \cdot p$
///    - $r_{\text{new}} = r - \alpha \cdot Ap$
///    - $z_{\text{new}} = M^{-1} r_{\text{new}}$
///    - $\beta = (r_{\text{new}} \cdot z_{\text{new}}) / (r \cdot z)$
///    - $p = z_{\text{new}} + \beta \cdot p$
///    - $r = r_{\text{new}},\ z = z_{\text{new}}$
///
/// # 参数
/// - `a`: 稀疏 SPD 矩阵（CSR 格式）
/// - `b`: 右端向量（长度 = `a.rows()`）
/// - `preconditioner`: 预条件对角向量 $M^{-1}$（长度 = `a.rows()`）
/// - `max_iter`: 最大迭代次数
/// - `tol`: 残差相对下降容差（$\|r_k\| / \|b\| < tol$ 时停止）
///
/// # 返回
/// - `Some(x)`：近似解
/// - `None`：达到最大迭代次数仍未收敛，或维度不匹配
pub fn conjugate_gradient_preconditioned(
    a: &CsMat<f64>,
    b: &[f64],
    preconditioner: &[f64],
    max_iter: usize,
    tol: f64,
) -> Option<Vec<f64>> {
    let n = a.rows();
    // 维度不匹配时返回 None 而非 panic
    if n != b.len() || n != preconditioner.len() {
        return None;
    }
    if n == 0 {
        return Some(Vec::new());
    }

    let b_norm = norm2(b);
    if b_norm < 1e-30 {
        return Some(vec![0.0; n]);
    }

    let mut x = vec![0.0; n];
    let mut r = b.to_vec(); // r_0 = b - A*x_0 = b（x_0 = 0）

    // z_0 = M^{-1} * r_0（逐元素乘）
    let mut z: Vec<f64> = r
        .iter()
        .zip(preconditioner.iter())
        .map(|(ri, mi)| ri * mi)
        .collect();
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    for _iter in 0..max_iter {
        // Ap = A * p
        let ap = sparse_matvec(a, &p);

        let p_ap = dot(&p, &ap);
        // 除零守卫：半正定系统未正则化时 p·Ap 可能为零
        if p_ap.abs() < 1e-30 {
            return None;
        }
        let alpha = rz_old / p_ap;

        // x_{k+1} = x_k + alpha * p_k
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r_{k+1} = r_k - alpha * Ap
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let residual_rel = norm2(&r) / b_norm;
        if residual_rel < tol {
            return Some(x);
        }

        // z_{k+1} = M^{-1} * r_{k+1}（逐元素乘）
        for i in 0..n {
            z[i] = r[i] * preconditioner[i];
        }

        let rz_new = dot(&r, &z);

        // 除零守卫：rz_old 为零时无法计算 beta
        if rz_old.abs() < 1e-30 {
            return None;
        }
        let beta = rz_new / rz_old;

        // p_{k+1} = z_{k+1} + beta * p_k
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    None
}

/// 构建 Jacobi（对角）预条件向量。
///
/// 提取稀疏矩阵 $A$ 的对角线，返回 $M^{-1}$ 的对角元
/// （$1 / A_{ii}$），用于 [`conjugate_gradient_preconditioned`]。
///
/// 对角元为零或缺失的行用 `1.0` 替代以避免除零。
pub fn jacobi_preconditioner(a: &CsMat<f64>) -> Vec<f64> {
    let n = a.rows();
    let mut diag_inv = vec![1.0; n];
    for (i, slot) in diag_inv.iter_mut().enumerate() {
        if let Some(&val) = a.get(i, i)
            && val.abs() > 1e-30
        {
            *slot = 1.0 / val;
        }
    }
    diag_inv
}

/// 对稀疏 SPD 矩阵做对角正则化：`A_reg = A + lambda * I`。
///
/// 用于半正定系统（如拉普拉斯矩阵），使 CG 收敛。
pub fn regularize_diagonal(a: &mut CsMat<f64>, lambda: f64) {
    let n = a.rows();
    for i in 0..n {
        // CsMat 的行索引可能有空洞——我们需要找到 (i,i) 位置
        // sprs CsMat 的行范围通过 outer_iterator 访问
        // 更简单的方法：直接通过索引修改
        if let Some(val) = a.get_mut(i, i) {
            *val += lambda;
        }
        // 如果 (i,i) 原本为 0，则 sprs 不会存储该条目。
        // 对于我们的应用（拉普拉斯矩阵），对角线一定非零，所以不需要处理。
    }
}

// ============================================================
// 内部工具
// ============================================================

/// 稀疏矩阵-向量乘法 `y = A * x`，返回稠密向量。
fn sparse_matvec(a: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.rows()];
    for (row_idx, row) in a.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row.iter() {
            // 越界守卫：列索引超出 x 长度时跳过，避免 panic
            if col_idx < x.len() {
                sum += val * x[col_idx];
            }
        }
        y[row_idx] = sum;
    }
    y
}

/// 向量内积。
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 向量 2-范数。
pub(crate) fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

// ============================================================
// 三维向量工具（[f64; 3]）
// ============================================================

/// 三维向量基础运算。
///
/// 统一提供全库共用的 `[f64; 3]` 向量原语，避免 `geometry.rs` /
/// `geodesics.rs` / `deformation.rs` / `boolean.rs` / `validate.rs` /
/// `test_util.rs` 各自重复定义。
pub(crate) mod vec3 {
    pub(crate) type Vec3 = [f64; 3];

    #[inline]
    pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    #[inline]
    pub(crate) fn add(a: Vec3, b: Vec3) -> Vec3 {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    #[inline]
    pub(crate) fn scale(a: Vec3, s: f64) -> Vec3 {
        [a[0] * s, a[1] * s, a[2] * s]
    }

    #[inline]
    pub(crate) fn dot(a: Vec3, b: Vec3) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[inline]
    pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    #[inline]
    pub(crate) fn length(a: Vec3) -> f64 {
        dot(a, a).sqrt()
    }

    #[inline]
    pub(crate) fn normalize(a: Vec3) -> Vec3 {
        let l = length(a);
        if l < 1e-12 { a } else { scale(a, 1.0 / l) }
    }

    #[inline]
    pub(crate) fn angle_between(u: Vec3, v: Vec3) -> f64 {
        let lu = length(u);
        let lv = length(v);
        if lu < 1e-12 || lv < 1e-12 {
            return 0.0;
        }
        let c = dot(u, v) / (lu * lv);
        c.clamp(-1.0, 1.0).acos()
    }

    /// 三角形面积（顶点位置）：$0.5 \cdot |(b-a) \times (c-a)|$。
    #[inline]
    pub(crate) fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f64 {
        0.5 * length(cross(sub(b, a), sub(c, a)))
    }

    /// 三角形单位法向（顶点位置）：$\mathrm{normalize}((b-a) \times (c-a))$。
    #[inline]
    pub(crate) fn triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        normalize(cross(sub(b, a), sub(c, a)))
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_identity() {
        let mut tri = TriMat::new((3, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        tri.add_triplet(2, 2, 1.0);
        let a = tri.to_csr();
        let b = vec![1.0, 2.0, 3.0];
        let x = conjugate_gradient(&a, &b, 50, 1e-12).unwrap();
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cg_small_spd() {
        // A = [[2,  1,  0],
        //      [1,  3, -1],
        //      [0, -1,  2]]
        let mut tri = TriMat::new((3, 3));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(1, 1, 3.0);
        tri.add_triplet(2, 2, 2.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 2, -1.0);
        tri.add_triplet(2, 1, -1.0);
        let a = tri.to_csr();
        // Solution to Ax = [1, 0, 1]^T should be x ≈ [0.3, 0.4, 0.7]
        let b = vec![1.0, 0.0, 1.0];
        let x = conjugate_gradient(&a, &b, 50, 1e-12).unwrap();
        // Verify Ax ≈ b
        let ax = sparse_matvec(&a, &x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "component {i}: Ax={} b={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_regularize_diagonal() {
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 2.0);
        let mut a = tri.to_csr();
        regularize_diagonal(&mut a, 0.5);
        assert!((a.get(0, 0).unwrap() - 1.5).abs() < 1e-14);
        assert!((a.get(1, 1).unwrap() - 2.5).abs() < 1e-14);
    }

    #[test]
    fn test_sparse_system_builder() {
        let mut sys = SparseSystem::new(3);
        sys.add(0, 0, 4.0);
        sys.add(0, 1, -1.0);
        sys.add(1, 2, -2.0);
        sys.add_diag(2, 3.0);
        let a = sys.finish();
        assert_eq!(a.rows(), 3);
        assert!((a.get(0, 0).unwrap() - 4.0).abs() < 1e-14);
        assert!((a.get(0, 1).unwrap() + 1.0).abs() < 1e-14);
        assert!((a.get(1, 0).unwrap() + 1.0).abs() < 1e-14);
        assert!((a.get(1, 2).unwrap() + 2.0).abs() < 1e-14);
        assert!((a.get(2, 1).unwrap() + 2.0).abs() < 1e-14);
        assert!((a.get(2, 2).unwrap() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn cg_dimension_mismatch_returns_none() {
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        let a = tri.to_csr();
        let b = vec![1.0, 2.0, 3.0]; // 长度 3 ≠ 矩阵维度 2
        assert!(conjugate_gradient(&a, &b, 10, 1e-10).is_none());
    }

    #[test]
    fn cg_empty_matrix_returns_empty() {
        let a = TriMat::new((0, 0)).to_csr();
        let b: Vec<f64> = vec![];
        assert_eq!(conjugate_gradient(&a, &b, 10, 1e-10), Some(Vec::new()));
    }

    #[test]
    fn cg_zero_right_hand_side_returns_zero_solution() {
        let mut tri = TriMat::new((3, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        tri.add_triplet(2, 2, 1.0);
        let a = tri.to_csr();
        let b = vec![0.0, 0.0, 0.0];
        assert_eq!(
            conjugate_gradient(&a, &b, 10, 1e-10),
            Some(vec![0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn cg_max_iter_zero_returns_none() {
        // 可解的 3x3 SPD 系统
        let mut tri = TriMat::new((3, 3));
        tri.add_triplet(0, 0, 4.0);
        tri.add_triplet(1, 1, 3.0);
        tri.add_triplet(2, 2, 2.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        let a = tri.to_csr();
        let b = vec![1.0, 2.0, 3.0];
        assert!(conjugate_gradient(&a, &b, 0, 1e-10).is_none());
    }

    #[test]
    fn cg_semidefinite_returns_none() {
        // A = [[1,-1],[-1,1]]，特征值 0 和 2（半正定），p·Ap=0 触发除零守卫
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        tri.add_triplet(0, 1, -1.0);
        tri.add_triplet(1, 0, -1.0);
        let a = tri.to_csr();
        let b = vec![1.0, 1.0];
        assert!(conjugate_gradient(&a, &b, 100, 1e-10).is_none());
    }

    #[test]
    fn cg_known_solution_2x2() {
        // A = [[4,1],[1,3]]，b = [1,2]，精确解 x = [1/11, 7/11]
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 4.0);
        tri.add_triplet(1, 1, 3.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        let a = tri.to_csr();
        let b = vec![1.0, 2.0];
        let x = conjugate_gradient(&a, &b, 100, 1e-12).expect("SPD 系统应收敛");
        let exact = [1.0 / 11.0, 7.0 / 11.0];
        for i in 0..2 {
            assert!(
                (x[i] - exact[i]).abs() < 1e-8,
                "分量 {i}: 得到 {}，期望 {}",
                x[i],
                exact[i]
            );
        }
    }

    #[test]
    fn sparse_system_add_out_of_bounds_silently_drops() {
        let mut sys = SparseSystem::new(2);
        sys.add(5, 0, 1.0); // i=5 越界，应丢弃
        sys.add_diag(5, 1.0); // i=5 越界，应丢弃
        let a = sys.finish();
        assert_eq!(a.rows(), 2);
        // 2x2 矩阵所有位置应为空（无任何条目被写入）
        assert!(a.get(0, 0).is_none());
        assert!(a.get(0, 1).is_none());
        assert!(a.get(1, 0).is_none());
        assert!(a.get(1, 1).is_none());
    }

    #[test]
    fn sparse_system_add_diag_accumulates() {
        let mut sys = SparseSystem::new(2);
        sys.add_diag(0, 1.0);
        sys.add_diag(0, 2.0);
        let a = sys.finish();
        assert!((a.get(0, 0).unwrap() - 3.0).abs() < 1e-14);
    }

    // ------------------------------------------------------------
    // 预条件共轭梯度法（PCG）测试
    // ------------------------------------------------------------

    #[test]
    fn test_jacobi_preconditioner_extracts_diagonal() {
        // A 对角线 = [2, 4, 0.5, 8]，期望 M^{-1} = [0.5, 0.25, 2.0, 0.125]
        let mut tri = TriMat::new((4, 4));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(1, 1, 4.0);
        tri.add_triplet(2, 2, 0.5);
        tri.add_triplet(3, 3, 8.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(2, 3, -1.0);
        tri.add_triplet(3, 2, -1.0);
        let a = tri.to_csr();
        let precond = jacobi_preconditioner(&a);
        assert!((precond[0] - 0.5).abs() < 1e-14);
        assert!((precond[1] - 0.25).abs() < 1e-14);
        assert!((precond[2] - 2.0).abs() < 1e-14);
        assert!((precond[3] - 0.125).abs() < 1e-14);
    }

    #[test]
    fn test_jacobi_preconditioner_zero_diagonal_guard() {
        // 对角元为零或缺失时应用 1.0 守卫
        let mut tri = TriMat::new((3, 3));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(1, 1, 0.0); // 显式零对角元
        // (2,2) 不存储，缺失对角元
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        let a = tri.to_csr();
        let precond = jacobi_preconditioner(&a);
        assert!((precond[0] - 0.5).abs() < 1e-14);
        assert!((precond[1] - 1.0).abs() < 1e-14, "零对角元应守卫为 1.0");
        assert!((precond[2] - 1.0).abs() < 1e-14, "缺失对角元应守卫为 1.0");
    }

    #[test]
    fn test_pcg_fewer_iterations_than_plain_cg() {
        // 构造病态对角矩阵：对角线 = [1, 2, 4, ..., 2^19]
        // 条件数 = 2^19 ≈ 5e5，普通 CG 需多达 n 次迭代
        // Jacobi 预条件将其变为单位矩阵，PCG 1 步即收敛
        let n = 20;
        let mut tri = TriMat::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, 2f64.powi(i as i32));
        }
        let a = tri.to_csr();
        let b = vec![1.0; n];

        let precond = jacobi_preconditioner(&a);

        // 普通 CG 在 5 次迭代内不收敛
        assert!(
            conjugate_gradient(&a, &b, 5, 1e-10).is_none(),
            "普通 CG 不应在 5 次迭代内收敛此病态系统"
        );

        // PCG 在 5 次迭代内收敛
        let x_pcg = conjugate_gradient_preconditioned(&a, &b, &precond, 5, 1e-10)
            .expect("PCG 应在 5 次迭代内收敛");

        // 验证解的正确性
        let ax = sparse_matvec(&a, &x_pcg);
        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "分量 {i}: Ax={} b={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_pcg_same_solution_as_plain_cg() {
        // A = [[2, 1, 0], [1, 3, -1], [0, -1, 2]]
        let mut tri = TriMat::new((3, 3));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(1, 1, 3.0);
        tri.add_triplet(2, 2, 2.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 2, -1.0);
        tri.add_triplet(2, 1, -1.0);
        let a = tri.to_csr();
        let b = vec![1.0, 0.0, 1.0];

        let x_plain = conjugate_gradient(&a, &b, 100, 1e-12).unwrap();
        let precond = jacobi_preconditioner(&a);
        let x_pcg = conjugate_gradient_preconditioned(&a, &b, &precond, 100, 1e-12).unwrap();

        for i in 0..3 {
            assert!(
                (x_plain[i] - x_pcg[i]).abs() < 1e-10,
                "分量 {i}: 普通 CG={} PCG={}",
                x_plain[i],
                x_pcg[i]
            );
        }
    }

    #[test]
    fn pcg_dimension_mismatch_returns_none() {
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        let a = tri.to_csr();
        let b = vec![1.0, 2.0];
        let precond = vec![1.0, 2.0, 3.0]; // 长度 3 ≠ 矩阵维度 2
        assert!(conjugate_gradient_preconditioned(&a, &b, &precond, 10, 1e-10).is_none());
    }
}
