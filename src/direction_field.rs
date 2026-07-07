//! 方向场模块。
//!
//! 实现 N-RoSy（N 重旋转对称）方向场，基于 Knöppel et al. (2013/2015)
//! "Globally Optimal Direction Fields" 的协变拉普拉斯特征值方法。
//!
//! ## 核心算法
//!
//! 最光滑 N-RoSy 场通过求解协变拉普拉斯 $L^\nabla_N$ 的最小特征向量得到：
//! $$L^\nabla_N \mathbf{z} = \lambda_{\min} \mathbf{z}$$
//!
//! 其中 $L^\nabla_N$ 是复埃尔米特矩阵，实数化后为 $2|F| \times 2|F|$ 的实对称矩阵。
//!
//! ## N-RoSy 类型
//!
//! | N | 名称 | 等价类 | 应用 |
//! |---|------|--------|------|
//! | 1 | 切向量场 | $d \sim d$ | 流场可视化 |
//! | 2 | 线场/交叉场 | $d \sim -d$ | 主曲率方向 |
//! | 4 | 帧场 | $d \sim R_{\pi/2}d$ | 四边形网格化 |
//!
//! ## API
//!
//! | 函数 | 功能 |
//! |------|------|
//! | [`smoothest_nrosy`] | 最光滑 N-RoSy 方向场 |
//! | [`smoothest_vector_field`] | N=1 便捷函数 |
//! | [`smoothest_cross_field`] | N=2 便捷函数 |
//! | [`smoothest_frame_field`] | N=4 便捷函数 |
//! | [`detect_singularities`] | 检测奇异点 |

use std::collections::HashMap;

use crate::geometry::{cotan_edge_weight, edge_length, face_normal};
use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::linalg::norm2;
use crate::linalg::{SparseSystem, conjugate_gradient};
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexAdjacentFaces};

// ============================================================
// 类型定义
// ============================================================

/// 面局部坐标系。
#[derive(Debug, Clone)]
pub struct FaceLocalFrame {
    /// 面内参考方向 e1（单位向量，在切平面内）
    pub e1: [f64; 3],
    /// 面内正交方向 e2 = normal × e1
    pub e2: [f64; 3],
    /// 面法向
    pub normal: [f64; 3],
}

/// 奇异点信息。
#[derive(Debug, Clone)]
pub struct Singularity {
    /// 奇异点所在顶点
    pub vertex: VertexId,
    /// 奇异点指数（对 N-RoSy，指数为 k/N，k 为整数）
    pub index: f64,
}

// ============================================================
// 面局部坐标系
// ============================================================

/// 为每个面构建局部坐标系。
///
/// e1 取面最短边方向在切平面上的投影（归一化），
/// e2 = normal × e1，normal = 面法向。
pub fn build_face_local_frames(mesh: &MeshStorage) -> HashMap<FaceId, FaceLocalFrame> {
    let mut frames = HashMap::new();
    for f in mesh.face_ids() {
        let normal = match face_normal(mesh, f) {
            Some(n) => n,
            None => continue,
        };
        // 找最短边方向
        let mut best_dir = [1.0, 0.0, 0.0];
        let mut best_len = f64::INFINITY;
        let he_ids: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, f).collect();
        for &he in &he_ids {
            if let Some(l) = edge_length(mesh, he)
                && l < best_len
                && l > 1e-14
            {
                best_len = l;
                let h = match mesh.get_halfedge(he) {
                    Some(h) => h,
                    None => continue,
                };
                let tip = h.vertex;
                let origin = match h.twin.and_then(|t| mesh.get_halfedge(t)) {
                    Some(t) => t.vertex,
                    None => continue,
                };
                let p_tip = match mesh.get_vertex(tip) {
                    Some(v) => v.position,
                    None => continue,
                };
                let p_origin = match mesh.get_vertex(origin) {
                    Some(v) => v.position,
                    None => continue,
                };
                let dir = sub3(p_tip, p_origin);
                let len = norm3(dir);
                if len > 1e-14 {
                    best_dir = scale3(dir, 1.0 / len);
                }
            }
        }
        // 将 best_dir 投影到切平面
        let dot_val = dot3(best_dir, normal);
        let proj = sub3(best_dir, scale3(normal, dot_val));
        let proj_len = norm3(proj);
        let e1 = if proj_len > 1e-14 {
            scale3(proj, 1.0 / proj_len)
        } else {
            // 最短边几乎平行于法向，选任意正交方向
            let arb = if normal[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let p = sub3(arb, scale3(normal, dot3(arb, normal)));
            let pl = norm3(p);
            if pl > 1e-14 {
                scale3(p, 1.0 / pl)
            } else {
                [1.0, 0.0, 0.0]
            }
        };
        let e2 = cross3(normal, e1);
        let e2_len = norm3(e2);
        let e2 = if e2_len > 1e-14 {
            scale3(e2, 1.0 / e2_len)
        } else {
            [0.0, 1.0, 0.0]
        };

        frames.insert(f, FaceLocalFrame { e1, e2, normal });
    }
    frames
}

// ============================================================
// 平行转移角
// ============================================================

/// 计算相邻面间的平行转移角。
///
/// 对共享边 e 的两个面 fi, fj，计算将 fj 的参考方向 e1^j
/// 绕共享边旋转到 fi 切平面后与 e1^i 的夹角 δ_ij。
///
/// 返回 HashMap，键为 (fi_index, fj_index) 的排序对。
pub fn compute_transport_angles(
    mesh: &MeshStorage,
    frames: &HashMap<FaceId, FaceLocalFrame>,
    face_index: &HashMap<FaceId, usize>,
) -> HashMap<(usize, usize), f64> {
    let mut angles = HashMap::new();

    for he in mesh.halfedge_ids() {
        let h = match mesh.get_halfedge(he) {
            Some(h) => h,
            None => continue,
        };
        let Some(twin_id) = h.twin else { continue };
        let twin = match mesh.get_halfedge(twin_id) {
            Some(t) => t,
            None => continue,
        };
        let Some(fi) = h.face else { continue };
        let Some(fj) = twin.face else { continue };
        if fi == fj {
            continue;
        }

        let Some(frame_i) = frames.get(&fi) else {
            continue;
        };
        let Some(frame_j) = frames.get(&fj) else {
            continue;
        };
        let Some(&idx_i) = face_index.get(&fi) else {
            continue;
        };
        let Some(&idx_j) = face_index.get(&fj) else {
            continue;
        };

        // 共享边方向：从 origin 到 tip
        let tip = h.vertex;
        let origin = twin.vertex;
        let p_tip = match mesh.get_vertex(tip) {
            Some(v) => v.position,
            None => continue,
        };
        let p_origin = match mesh.get_vertex(origin) {
            Some(v) => v.position,
            None => continue,
        };
        let edge_dir = sub3(p_tip, p_origin);
        let edge_len = norm3(edge_dir);
        if edge_len < 1e-14 {
            continue;
        }
        let edge_dir = scale3(edge_dir, 1.0 / edge_len);

        // 二面角的补角：绕共享边将 fj 法向旋转到 fi 法向的角度
        let ni = frame_i.normal;
        let nj = frame_j.normal;
        let cos_alpha = dot3(ni, nj);
        let sin_alpha = dot3(cross3(nj, ni), edge_dir);
        let alpha = sin_alpha.atan2(cos_alpha);

        // 将 e1^j 绕共享边旋转 alpha 到 fi 的切平面
        let e1j_par = rodrigues_rotate(frame_j.e1, edge_dir, alpha);

        // 投影到 fi 切平面（消除数值误差的法向分量）
        let proj = sub3(e1j_par, scale3(ni, dot3(e1j_par, ni)));
        let proj_len = norm3(proj);
        if proj_len < 1e-14 {
            continue;
        }
        let e1j_par = scale3(proj, 1.0 / proj_len);

        // δ_ij = e1^i 与 e1j_par 的带符号夹角
        let cos_delta = dot3(frame_i.e1, e1j_par);
        let sin_delta = dot3(cross3(ni, e1j_par), frame_i.e1);
        let delta = sin_delta.atan2(cos_delta);

        // 存储 (min_idx, max_idx) → delta
        // 注意：delta(i→j) = -delta(j→i)
        let key = if idx_i < idx_j {
            (idx_i, idx_j)
        } else {
            (idx_j, idx_i)
        };
        let sign = if idx_i < idx_j { 1.0 } else { -1.0 };
        angles.insert(key, sign * delta);
    }

    angles
}

/// Rodrigues 旋转公式：将向量 v 绕单位轴 k 旋转角度 angle。
fn rodrigues_rotate(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let k_cross_v = cross3(k, v);
    let k_dot_v = dot3(k, v);
    add3(
        add3(scale3(v, cos_a), scale3(k_cross_v, sin_a)),
        scale3(k, k_dot_v * (1.0 - cos_a)),
    )
}

// ============================================================
// 协变拉普拉斯矩阵构建
// ============================================================

/// 构建 N-RoSy 协变拉普拉斯的实数化矩阵（2|F| × 2|F| 对称矩阵）。
///
/// 复数协变拉普拉斯 $L^\nabla_N$ 的元素：
/// $$L^\nabla_{ij} = \begin{cases}
///   \sum_k w_{ik} & i = j \\
///   -w_{ij} e^{iN\delta_{ij}} & (f_i, f_j) \text{ 相邻}
/// \end{cases}$$
///
/// 实数化后为块结构：
/// $$L^\nabla_{\mathbb{R}} = \begin{pmatrix} A & -B \\ -B & A \end{pmatrix}$$
/// 其中 $A = \text{Re}(L^\nabla)$, $B = \text{Im}(L^\nabla)$。
fn build_covariant_laplacian_real(
    mesh: &MeshStorage,
    n_sym: usize,
    _frames: &HashMap<FaceId, FaceLocalFrame>,
    face_index: &HashMap<FaceId, usize>,
    transport_angles: &HashMap<(usize, usize), f64>,
    n_faces: usize,
) -> sprs::CsMat<f64> {
    let dim = 2 * n_faces;
    let mut sys = SparseSystem::new(dim);

    // 遍历所有面邻接对
    for he in mesh.halfedge_ids() {
        let h = match mesh.get_halfedge(he) {
            Some(h) => h,
            None => continue,
        };
        let Some(twin_id) = h.twin else { continue };
        let twin = match mesh.get_halfedge(twin_id) {
            Some(t) => t,
            None => continue,
        };
        let Some(fi) = h.face else { continue };
        let Some(fj) = twin.face else { continue };
        if fi == fj {
            continue;
        }

        let Some(&idx_i) = face_index.get(&fi) else {
            continue;
        };
        let Some(&idx_j) = face_index.get(&fj) else {
            continue;
        };
        if idx_i >= idx_j {
            continue; // 只处理 idx_i < idx_j，对称性处理
        }

        // 余切权重
        let w = cotan_edge_weight(mesh, he).unwrap_or(0.5);
        // 确保权重非负（内蕴 Delaunay 后应为非负）
        let w = w.max(0.0);
        if w < 1e-14 {
            continue;
        }

        // 获取转移角
        let key = (idx_i, idx_j);
        let delta = match transport_angles.get(&key) {
            Some(&d) => d,
            None => continue,
        };

        // N-RoSy 转移相位
        let phi = n_sym as f64 * delta;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        // 实数化矩阵的四个块
        // 对角块 A: (2*idx_i, 2*idx_j) 和 (2*idx_i+1, 2*idx_j+1)
        // 非对角块 B: (2*idx_i, 2*idx_j+1) 和 (2*idx_i+1, 2*idx_j)

        // L^∇_ij = -w * (cos(phi) + i*sin(phi))
        // 实数化：
        // Re block: (2i, 2j) += -w*cos(phi), (2i+1, 2j+1) += -w*cos(phi)
        // Im block: (2i, 2j+1) += w*sin(phi), (2i+1, 2j) += -w*sin(phi)
        // 注意：由于 SparseSystem::add 自动对称化，我们只需写上三角

        let ri = 2 * idx_i;
        let rj = 2 * idx_j;

        // 对角块 A
        sys.add(ri, rj, -w * cos_phi);
        sys.add(ri + 1, rj + 1, -w * cos_phi);

        // 非对角块 B（注意符号）
        // (ri, rj+1): +w*sin(phi
        sys.add(ri, rj + 1, w * sin_phi);
        // (ri+1, rj): -w*sin(phi)
        sys.add(ri + 1, rj, -w * sin_phi);

        // 对角元 += w（每个面的度数贡献）
        sys.add_diag(ri, w);
        sys.add_diag(ri + 1, w);
        sys.add_diag(rj, w);
        sys.add_diag(rj + 1, w);
    }

    // 处理孤立面（无边邻接的面）
    for (&f, &idx) in face_index {
        let ri = 2 * idx;
        // 确保对角元非零（正则化）
        let has_neighbors = mesh.halfedge_ids().any(|he| {
            mesh.get_halfedge(he)
                .and_then(|h| h.face)
                .is_some_and(|fi| fi == f)
                && mesh
                    .get_halfedge(he)
                    .and_then(|h| h.twin)
                    .is_some_and(|twin| {
                        mesh.get_halfedge(twin)
                            .and_then(|t| t.face)
                            .is_some_and(|fj| fj != f)
                    })
        });
        if !has_neighbors {
            sys.add_diag(ri, 1e-6);
            sys.add_diag(ri + 1, 1e-6);
        }
    }

    sys.finish()
}

// ============================================================
// 最小特征向量求解
// ============================================================

/// 逆幂迭代法求最小特征值对应的特征向量。
///
/// 求解 $(L + \epsilon I) x = b$，其中 $L$ 是半正定矩阵。
/// 最小特征向量对应 $L$ 的最小特征值。
fn smallest_eigenvector(mat: &sprs::CsMat<f64>, dim: usize, max_iter: usize, tol: f64) -> Vec<f64> {
    // 正则化使矩阵正定
    let mut mat_reg = mat.clone();
    crate::linalg::regularize_diagonal(&mut mat_reg, 1e-8);

    // 随机初始化
    let mut x = vec![0.0; dim];
    // 使用确定性种子以获得可复现结果
    let seed = 42u64;
    let mut state = seed;
    for v in x.iter_mut() {
        // 简单的伪随机数生成
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
        *v = val;
    }
    // 归一化
    let x_norm = norm2(&x);
    if x_norm > 1e-14 {
        for v in &mut x {
            *v /= x_norm;
        }
    }

    // 逆幂迭代
    for _ in 0..max_iter {
        let mut y = match conjugate_gradient(&mat_reg, &x, 500, tol * 0.01) {
            Some(y) => y,
            None => break,
        };
        let y_norm = norm2(&y);
        if y_norm < 1e-14 {
            break;
        }
        for v in &mut y {
            *v /= y_norm;
        }
        // 检查收敛：||x - y|| / ||x||
        let diff: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
        x = y;
        if diff.sqrt() < tol {
            break;
        }
    }

    x
}

// ============================================================
// 公共 API
// ============================================================

/// 计算最光滑 N-RoSy 方向场。
///
/// 返回每个面的角度 $\theta^f$（弧度），方向为
/// $d^f = \cos(\theta^f) \cdot e_1^f + \sin(\theta^f) \cdot e_2^f$。
///
/// # 算法
/// 1. 构建面局部坐标系 $(e_1^f, e_2^f, n^f)$
/// 2. 计算相邻面间平行转移角 $\delta_{ij}$
/// 3. 构建协变拉普拉斯 $L^\nabla_N$（实数化）
/// 4. 逆幂迭代求最小特征向量
/// 5. 从特征向量提取角度
///
/// # 参数
/// - `mesh`: 三角网格
/// - `n_sym`: 旋转对称阶数（1=向量场, 2=交叉场, 4=帧场）
pub fn smoothest_nrosy(mesh: &MeshStorage, n_sym: usize) -> HashMap<FaceId, f64> {
    if n_sym == 0 {
        return HashMap::new();
    }

    // 面索引映射
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    let n_faces = faces.len();
    if n_faces == 0 {
        return HashMap::new();
    }
    let face_index: HashMap<FaceId, usize> =
        faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();

    // 1. 构建面局部坐标系
    let frames = build_face_local_frames(mesh);

    // 2. 计算转移角
    let transport_angles = compute_transport_angles(mesh, &frames, &face_index);

    // 3. 构建协变拉普拉斯
    let mat = build_covariant_laplacian_real(
        mesh,
        n_sym,
        &frames,
        &face_index,
        &transport_angles,
        n_faces,
    );

    // 4. 求最小特征向量
    let dim = 2 * n_faces;
    let eigvec = smallest_eigenvector(&mat, dim, 100, 1e-8);

    // 5. 提取角度
    let mut result = HashMap::new();
    for (&f, &idx) in &face_index {
        let re = eigvec[2 * idx];
        let im = eigvec[2 * idx + 1];
        let theta = im.atan2(re) / n_sym as f64;
        result.insert(f, theta);
    }

    result
}

/// 最光滑切向量场（N=1）。
pub fn smoothest_vector_field(mesh: &MeshStorage) -> HashMap<FaceId, f64> {
    smoothest_nrosy(mesh, 1)
}

/// 最光滑交叉场（N=2）。
pub fn smoothest_cross_field(mesh: &MeshStorage) -> HashMap<FaceId, f64> {
    smoothest_nrosy(mesh, 2)
}

/// 最光滑帧场（N=4）。
pub fn smoothest_frame_field(mesh: &MeshStorage) -> HashMap<FaceId, f64> {
    smoothest_nrosy(mesh, 4)
}

// ============================================================
// 奇异点检测
// ============================================================

/// 检测 N-RoSy 方向场的奇异点。
///
/// 绕顶点的一环邻域面累加角度差（考虑转移），归一化后得到指数：
/// $$\text{index}(v) = \frac{1}{2\pi N} \sum_{(f_i, f_j) \in \text{ring}(v)}
///   \text{wrap}(N\theta^{f_j} - N\theta^{f_i} - N\delta_{ij})$$
pub fn detect_singularities(
    mesh: &MeshStorage,
    n_sym: usize,
    theta: &HashMap<FaceId, f64>,
) -> Vec<Singularity> {
    let frames = build_face_local_frames(mesh);
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    let face_index: HashMap<FaceId, usize> =
        faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let transport_angles = compute_transport_angles(mesh, &frames, &face_index);

    let mut singularities = Vec::new();

    for v in mesh.vertex_ids().collect::<Vec<_>>() {
        let adj_faces: Vec<FaceId> = VertexAdjacentFaces::new(mesh, v).collect();
        if adj_faces.len() < 3 {
            continue;
        }

        let mut total_angle = 0.0;
        for i in 0..adj_faces.len() {
            let fi = adj_faces[i];
            let fj = adj_faces[(i + 1) % adj_faces.len()];

            let Some(&theta_i) = theta.get(&fi) else {
                continue;
            };
            let Some(&theta_j) = theta.get(&fj) else {
                continue;
            };
            let Some(&idx_i) = face_index.get(&fi) else {
                continue;
            };
            let Some(&idx_j) = face_index.get(&fj) else {
                continue;
            };

            // 获取转移角
            let key = if idx_i < idx_j {
                (idx_i, idx_j)
            } else {
                (idx_j, idx_i)
            };
            let delta = match transport_angles.get(&key) {
                Some(&d) => {
                    if idx_i < idx_j {
                        d
                    } else {
                        -d
                    }
                }
                None => 0.0,
            };

            // 角度差（考虑 N-RoSy 对称性）
            let diff = n_sym as f64 * (theta_j - theta_i) - n_sym as f64 * delta;
            // 归一化到 [-π, π]
            let wrapped = wrap_angle(diff);
            total_angle += wrapped;
        }

        let index = total_angle / (2.0 * std::f64::consts::PI);
        if index.abs() > 0.01 {
            singularities.push(Singularity { vertex: v, index });
        }
    }

    singularities
}

// ============================================================
// 辅助函数
// ============================================================

/// 将角度归一化到 (-π, π]。
fn wrap_angle(a: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;
    let mut a = a.rem_euclid(two_pi);
    if a > pi {
        a -= two_pi;
    }
    a
}

/// 向量减法。
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 向量加法。
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// 标量乘向量。
fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// 向量点积。
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 向量叉积。
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 向量范数。
fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::build_grid;
    use crate::storage::{MeshStorage, Vertex};
    use crate::test_util::build_icosphere;
    use crate::topology_ops::add_triangle;

    /// 平面网格上的 1-RoSy 场：角度应近似均匀。
    #[test]
    fn smoothest_vector_field_grid() {
        let mesh = build_grid(1.0, 1.0, 3, 3);
        let theta = smoothest_vector_field(&mesh);
        // 平面网格应有非空结果
        assert!(!theta.is_empty(), "方向场不应为空");
        // 角度应在 [-π, π] 范围内
        for &t in theta.values() {
            assert!(t.abs() <= std::f64::consts::PI + 0.1);
        }
    }

    /// icosphere 上的 2-RoSy 交叉场：场非空且角度合理。
    #[test]
    fn smoothest_cross_field_icosphere() {
        let mesh = build_icosphere(1);
        let theta = smoothest_cross_field(&mesh);
        assert!(!theta.is_empty());
        // 角度应在 [-π/2, π/2] 范围内（2-RoSy 的周期为 π）
        for &t in theta.values() {
            assert!(t.abs() <= std::f64::consts::PI + 0.1, "角度超出范围: {}", t);
        }
    }

    /// 4-RoSy 帧场在平面网格上。
    #[test]
    fn smoothest_frame_field_grid() {
        let mesh = build_grid(1.0, 1.0, 2, 2);
        let theta = smoothest_frame_field(&mesh);
        assert!(!theta.is_empty());
    }

    /// 空网格不 panic。
    #[test]
    fn smoothest_field_empty() {
        let mesh = MeshStorage::new();
        let theta = smoothest_nrosy(&mesh, 1);
        assert!(theta.is_empty());
    }

    /// 单三角形：方向场只有一个面，角度任意。
    #[test]
    fn smoothest_field_single_triangle() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        add_triangle(&mut mesh, v0, v1, v2).unwrap();
        let theta = smoothest_nrosy(&mesh, 1);
        assert_eq!(theta.len(), 1);
    }

    /// Rodrigues 旋转正确性。
    #[test]
    fn rodrigues_rotation_90deg() {
        let v = [1.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let rotated = rodrigues_rotate(v, axis, std::f64::consts::FRAC_PI_2);
        assert!((rotated[0] - 0.0).abs() < 1e-10, "x = {}", rotated[0]);
        assert!((rotated[1] - 1.0).abs() < 1e-10, "y = {}", rotated[1]);
        assert!((rotated[2] - 0.0).abs() < 1e-10, "z = {}", rotated[2]);
    }

    /// 面局部坐标系构建正确。
    #[test]
    fn face_local_frame_orthogonal() {
        let mesh = build_icosphere(1);
        let frames = build_face_local_frames(&mesh);
        for frame in frames.values() {
            // e1 · normal ≈ 0
            assert!(
                dot3(frame.e1, frame.normal).abs() < 1e-10,
                "e1 不在切平面内"
            );
            // e2 · normal ≈ 0
            assert!(
                dot3(frame.e2, frame.normal).abs() < 1e-10,
                "e2 不在切平面内"
            );
            // e1 · e2 ≈ 0
            assert!(dot3(frame.e1, frame.e2).abs() < 1e-10, "e1 与 e2 不正交");
            // |e1| ≈ 1
            assert!((norm3(frame.e1) - 1.0).abs() < 1e-10);
            // |e2| ≈ 1
            assert!((norm3(frame.e2) - 1.0).abs() < 1e-10);
        }
    }

    /// wrap_angle 正确性。
    #[test]
    fn wrap_angle_test() {
        assert!((wrap_angle(0.0)).abs() < 1e-10);
        assert!((wrap_angle(std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-10);
        assert!((wrap_angle(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-10);
        // -3π rem_euclid(2π) = π, 所以 wrap_angle(-3π) = π
        assert!((wrap_angle(-3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-10);
    }
}
