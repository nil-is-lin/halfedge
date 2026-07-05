//! 稀疏线性代数工具。
//!
//! 在 [`sprs`] 的 [`CsMat`](sprs::CsMat) 稀疏矩阵之上提供：
//! - [`SparseSystem`]：从网格拉普拉斯矩阵的三元组构建稀疏对称系统；
//! - [`conjugate_gradient`]：共轭梯度法求解 $Ax = b$（A 对称正定）；
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
// 共轭梯度法
// ============================================================

/// 共轭梯度法求解 $Ax = b$。
///
/// **前置条件：** $A$ 必须为对称正定（SPD）矩阵。若 $A$ 半正定（如拉普拉斯矩阵），
/// 调用方应先做正则化（对 $A$ 的对角线加一个小偏移）。
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
    assert_eq!(n, b.len(), "A 与 b 维度不匹配");
    if n == 0 {
        return Some(Vec::new());
    }

    let b_norm = norm2(b);
    if b_norm < 1e-30 {
        return Some(vec![0.0; n]);
    }

    let mut x = vec![0.0; n];
    let mut r = b.to_vec(); // r_0 = b - A*x_0 = b

    // r = b - A*x (初始 x=0)
    // 实际上我们用 r = b 开始，但严格来说应该是 r = b - A*x
    // 初始残差已正确，因为 x_0 = 0

    let mut p = r.clone();
    let mut rsold = dot(&r, &r);

    for _iter in 0..max_iter {
        // Ap = A * p
        let ap = sparse_matvec(a, &p);

        let alpha = rsold / dot(&p, &ap);

        // x_{k+1} = x_k + alpha * p_k
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r_{k+1} = r_k - alpha * Ap
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let rsnew = dot(&r, &r);
        let residual_rel = rsnew.sqrt() / b_norm;

        if residual_rel < tol {
            return Some(x);
        }

        // beta = rsnew / rsold
        let beta = rsnew / rsold;

        // p_{k+1} = r_{k+1} + beta * p_k
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    None
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
            sum += val * x[col_idx];
        }
        y[row_idx] = sum;
    }
    y
}

/// 向量内积。
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 向量 2-范数。
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
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
}
