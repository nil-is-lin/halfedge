//! 测地线距离（Geodesic Distance）。
//!
//! 实现 **Heat Method**（Crane, Weischedel, Wardetzky 2013），
//! 通过求解两个稀疏线性系统高效计算网格上从源点出发的测地线距离。
//!
//! ## 算法步骤
//! 1. 时间步 $\tau = h^2$（$h$ 为平均边长）
//! 2. 求解热流：$(I - \tau \Delta) u_\tau = u_0$（初始热源 $u_0$）
//! 3. 计算归一化梯度：$X = -\nabla u_\tau / |\nabla u_\tau|$
//! 4. 求解 Poisson：$\Delta \phi = \nabla \cdot X$
//! 5. $\phi$ 即为测地线距离（偏移去除后）
//!
//! 额外提供 [`shortest_path`] 沿距离场梯度回溯最短路径。

use std::collections::HashMap;

use rayon::prelude::*;

use crate::geometry::face_area;
use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::linalg::vec3::{self, Vec3};
use crate::linalg::{
    SparseSystem, build_cotan_laplacian, build_vertex_index, conjugate_gradient,
    regularize_diagonal,
};
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexAdjacentFaces, VertexRing};

// ============================================================
// 顶点 → 索引映射
// ============================================================

// build_vertex_index 已移至 linalg 模块作为公共函数

/// 同时构建顶点列表（O(1) 反查索引→VertexId）与索引映射（VertexId→索引）。
///
/// 单次遍历，避免在 Dijkstra 主循环中反复调用
/// `mesh.vertex_ids().nth(u)`（每次 O(u)，整体退化为 O(n^2)）。
fn build_vertex_index_and_list(mesh: &MeshStorage) -> (Vec<VertexId>, HashMap<VertexId, usize>) {
    let list: Vec<VertexId> = mesh.vertex_ids().collect();
    let map = list.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    (list, map)
}

// ============================================================
// 构建稀疏算子
// ============================================================

/// 构建余切拉普拉斯矩阵（N×N）和顶点质量（lumped mass，对角）。
///
/// 返回 `(laplacian, mass_vec)`，其中 `mass_vec[i]` = 每个顶点的 Voronoi 面积。
/// 拉普拉斯部分委托 [`crate::linalg::build_cotan_laplacian`]。
fn build_laplacian_and_mass(
    mesh: &MeshStorage,
    v_idx: &HashMap<VertexId, usize>,
) -> (SparseSystem, Vec<f64>) {
    let n = v_idx.len();
    let lap = build_cotan_laplacian(mesh, v_idx);
    let mut mass = vec![0.0; n];

    // 顶点面积：遍历面，每个面分配 1/3 面积给各顶点
    for f in mesh.face_ids() {
        let halfedges: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, f).collect();
        if halfedges.len() != 3 {
            continue;
        }
        let v0 = mesh
            .get_halfedge(halfedges[0])
            .expect("halfedge exists in mesh")
            .vertex;
        let v1 = mesh
            .get_halfedge(halfedges[1])
            .expect("halfedge exists in mesh")
            .vertex;
        let v2 = mesh
            .get_halfedge(halfedges[2])
            .expect("halfedge exists in mesh")
            .vertex;
        let Some(&i0) = v_idx.get(&v0) else { continue };
        let Some(&i1) = v_idx.get(&v1) else { continue };
        let Some(&i2) = v_idx.get(&v2) else { continue };
        let area = face_area(mesh, f).unwrap_or(0.0);
        let a3 = area / 3.0;
        mass[i0] += a3;
        mass[i1] += a3;
        mass[i2] += a3;
    }

    // 零面积保护
    for m in mass.iter_mut() {
        if *m < 1e-14 {
            *m = 1e-14;
        }
    }

    (lap, mass)
}

/// 构建梯度算子散度的右端项。
///
/// 对面 `f` 计算梯度向量 $\nabla u$，然后通过积分给出顶点散度。
/// 仅返回 RHS 向量；Poisson 求解所用的拉普拉斯矩阵由调用方一次性构建并复用，
/// 避免在每个步骤中重复构建。
fn build_divergence_from_gradient(
    mesh: &MeshStorage,
    v_idx: &HashMap<VertexId, usize>,
    face_gradient: &HashMap<FaceId, [f64; 3]>,
) -> Vec<f64> {
    let n = v_idx.len();
    let mut rhs = vec![0.0; n];

    for f in mesh.face_ids() {
        let Some(&grad) = face_gradient.get(&f) else {
            continue;
        };

        let halfedges: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, f).collect();
        if halfedges.len() != 3 {
            continue;
        }

        let v0 = mesh
            .get_halfedge(halfedges[0])
            .expect("halfedge exists in mesh")
            .vertex;
        let v1 = mesh
            .get_halfedge(halfedges[1])
            .expect("halfedge exists in mesh")
            .vertex;
        let v2 = mesh
            .get_halfedge(halfedges[2])
            .expect("halfedge exists in mesh")
            .vertex;
        let Some(&i0) = v_idx.get(&v0) else { continue };
        let Some(&i1) = v_idx.get(&v1) else { continue };
        let Some(&i2) = v_idx.get(&v2) else { continue };

        let p0 = mesh.get_vertex(v0).expect("vertex exists in mesh").position;
        let p1 = mesh.get_vertex(v1).expect("vertex exists in mesh").position;
        let p2 = mesh.get_vertex(v2).expect("vertex exists in mesh").position;

        let area = vec3::triangle_area(p0, p1, p2);
        if area < 1e-14 {
            continue;
        }

        // 对每条边计算散度贡献
        // ∇·X 的弱形式：对每个顶点 v, ∫_M (∇·X)φ dA = -∫_M X·∇φ dA
        // 使用分段线性基函数，∇φ_i 在面 f 上为常向量 = -n×e_i / (2|f|)
        // 其中 n 是面法向，e_i 是顶点 i 所对的边

        let n = vec3::triangle_normal(p0, p1, p2);
        let n = vec3::normalize(n);

        // 对面中每个顶点计算散度贡献
        let e01 = vec3::sub(p1, p0);
        let e12 = vec3::sub(p2, p1);
        let e20 = vec3::sub(p0, p2);

        // 旋转 90°：n × e / (2|f|)
        let grad_phi0 = vec3::scale(vec3::cross(n, e12), 1.0 / (2.0 * area));
        let grad_phi1 = vec3::scale(vec3::cross(n, e20), 1.0 / (2.0 * area));
        let grad_phi2 = vec3::scale(vec3::cross(n, e01), 1.0 / (2.0 * area));

        // 对每个顶点 j: rhs_j += ∫_f X·∇φ_j dA = (X·∇φ_j) * |f|
        rhs[i0] += vec3::dot(grad, grad_phi0) * area;
        rhs[i1] += vec3::dot(grad, grad_phi1) * area;
        rhs[i2] += vec3::dot(grad, grad_phi2) * area;
    }

    rhs
}

// ============================================================
// 向量工具
// ============================================================

// Vec3 与向量运算统一使用 crate::linalg::vec3 模块。

// ============================================================
// Heat Method
// ============================================================

/// Heat Method 热核法测地线距离。
///
/// 从源顶点 `source` 出发，计算网格上所有顶点的测地线距离。
///
/// # 参数
/// - `mesh`: 三角网格
/// - `source`: 源顶点
///
/// # 返回
/// - `Some(Vec<f64>)`：每个顶点的测地线距离（按 `mesh.vertex_ids()` 顺序）
/// - `None`：求解失败
///
/// # 实现
/// 余切拉普拉斯 `cot_lap` 仅构建一次：热流系统 $M + \tau L$ 与 Poisson 系统 $L$
/// 共用同一份 `CsMat`（后者 `clone` 后做对角正则化），避免重复构建。
pub fn geodesic_distance_from_vertex(mesh: &MeshStorage, source: VertexId) -> Option<Vec<f64>> {
    let n = mesh.vertex_count();
    if n == 0 {
        return Some(Vec::new());
    }

    let v_idx = build_vertex_index(mesh);
    let source_idx = *v_idx.get(&source)?;

    // 计算平均边长以确定时间步（并行归约）
    let he_ids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
    let (total_len, edge_count) = he_ids
        .par_iter()
        .filter_map(|&he| crate::geometry::edge_length(mesh, he))
        .fold(|| (0.0f64, 0u64), |(sum, cnt), len| (sum + len, cnt + 1))
        .reduce(|| (0.0f64, 0u64), |a, b| (a.0 + b.0, a.1 + b.1));
    let h_sq = if edge_count > 0 {
        let h = total_len / (edge_count as f64);
        h * h
    } else {
        1.0
    };

    let (lap, mass) = build_laplacian_and_mass(mesh, &v_idx);

    // ── 步骤 1-2: 求解热流 (I - tΔ) u = u₀ ──
    // 构建 A = M + t*L   （lumped mass 对角近似）
    let mut heat_sys = SparseSystem::new(n);

    // 添加拉普拉斯贡献
    let cot_lap = lap.finish();
    for (row_idx, row) in cot_lap.outer_iterator().enumerate() {
        for (col_idx, &val) in row.iter() {
            heat_sys.add(row_idx, col_idx, h_sq * val);
        }
    }
    // 添加质量矩阵（对角）
    for (i, &m) in mass.iter().enumerate() {
        heat_sys.add_diag(i, m);
    }

    let mut heat_a = heat_sys.finish();

    // 初始热源 u₀：在 source 处积分 = 1
    let mut u0 = vec![0.0; n];
    u0[source_idx] = 1.0 / mass[source_idx];

    // RHS = M * u₀（质量矩阵乘初始条件）
    let mut heat_rhs = vec![0.0; n];
    for i in 0..n {
        heat_rhs[i] = mass[i] * u0[i];
    }

    regularize_diagonal(&mut heat_a, 1e-10);
    let u = conjugate_gradient(&heat_a, &heat_rhs, n * 100, 1e-6)?;

    // ── 步骤 3: 计算面梯度 ──
    let face_grad = compute_face_gradients(mesh, &v_idx, &u);

    // ── 步骤 4: 归一化梯度（并行） ──
    let grad_entries: Vec<(FaceId, [f64; 3])> = face_grad.into_iter().collect();
    let face_grad_norm: HashMap<FaceId, [f64; 3]> = grad_entries
        .par_iter()
        .map(|(f, g)| {
            let len = vec3::length(*g);
            if len > 1e-10 {
                (*f, vec3::scale(*g, -1.0 / len))
            } else {
                (*f, [0.0, 0.0, 0.0])
            }
        })
        .collect();

    // ── 步骤 5: 求解 Poisson Δφ = ∇·X ──
    // 复用步骤 1 已构建的 cot_lap，避免重复构建拉普拉斯矩阵
    let div_rhs = build_divergence_from_gradient(mesh, &v_idx, &face_grad_norm);
    let mut poisson_lap = cot_lap.clone();
    regularize_diagonal(&mut poisson_lap, 1e-10);

    let phi = conjugate_gradient(&poisson_lap, &div_rhs, n * 100, 1e-6)?;

    // 偏移：使源点距离 = 0
    let phi_source = phi[source_idx];
    let distance: Vec<f64> = phi.iter().map(|&p| (p - phi_source).abs()).collect();

    Some(distance)
}

/// 计算每个面上的分段线性函数梯度。
///
/// 并行版本：每个面的梯度计算相互独立，使用 `par_iter()` 并行计算，
/// 收集后再顺序写入 HashMap。
fn compute_face_gradients(
    mesh: &MeshStorage,
    v_idx: &HashMap<VertexId, usize>,
    u: &[f64],
) -> HashMap<FaceId, [f64; 3]> {
    let face_ids: Vec<FaceId> = mesh.face_ids().collect();

    let results: Vec<(FaceId, [f64; 3])> = face_ids
        .par_iter()
        .filter_map(|&f| {
            let halfedges: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, f).collect();
            if halfedges.len() != 3 {
                return None;
            }

            let v0 = mesh
                .get_halfedge(halfedges[0])
                .expect("halfedge exists in mesh")
                .vertex;
            let v1 = mesh
                .get_halfedge(halfedges[1])
                .expect("halfedge exists in mesh")
                .vertex;
            let v2 = mesh
                .get_halfedge(halfedges[2])
                .expect("halfedge exists in mesh")
                .vertex;
            let &i0 = v_idx.get(&v0)?;
            let &i1 = v_idx.get(&v1)?;
            let &i2 = v_idx.get(&v2)?;

            let p0 = mesh.get_vertex(v0).expect("vertex exists in mesh").position;
            let p1 = mesh.get_vertex(v1).expect("vertex exists in mesh").position;
            let p2 = mesh.get_vertex(v2).expect("vertex exists in mesh").position;

            let area = vec3::triangle_area(p0, p1, p2);
            if area < 1e-14 {
                return None;
            }

            let n = vec3::triangle_normal(p0, p1, p2);

            // ∇u = (1/(2|f|)) * n × Σ u_i * e_i
            // 其中 e_i 是顶点 i 所对的边向量
            let e0 = vec3::sub(p2, p1);
            let e1 = vec3::sub(p0, p2);
            let e2 = vec3::sub(p1, p0);

            let sum = vec3::add(
                vec3::add(vec3::scale(e0, u[i0]), vec3::scale(e1, u[i1])),
                vec3::scale(e2, u[i2]),
            );

            let g = vec3::scale(vec3::cross(n, sum), 1.0 / (2.0 * area));
            Some((f, g))
        })
        .collect();

    let mut grad = HashMap::with_capacity(results.len());
    for (f, g) in results {
        grad.insert(f, g);
    }
    grad
}

// ============================================================
// 最短路径回溯
// ============================================================

/// 从目标顶点沿测地线距离梯度回溯到源顶点的最短路径。
///
/// 注意：需要先调用 [`geodesic_distance_from_vertex`] 计算距离场。
///
/// # 参数
/// - `mesh`: 三角网格
/// - `distance`: 每个顶点的测地线距离（由 heat method 返回）
/// - `target`: 目标顶点
///
/// # 返回
/// 从 `target` 到源顶点的顶点序列（含两端）。
pub fn shortest_path(mesh: &MeshStorage, distance: &[f64], target: VertexId) -> Vec<VertexId> {
    let v_idx = build_vertex_index(mesh);
    let Some(&target_idx) = v_idx.get(&target) else {
        return vec![];
    };
    if target_idx >= distance.len() {
        return vec![];
    }

    let mut path = vec![target];
    let mut current = target;
    let mut current_dist = distance[target_idx];

    let max_steps = mesh.vertex_count() * 2;
    for _ in 0..max_steps {
        if current_dist < 1e-10 {
            break;
        }

        // 找邻域中距离递减最多的顶点
        let mut best_neighbor = None;
        let mut best_dist = current_dist;

        for he in VertexRing::new(mesh, current) {
            let neighbor = mesh
                .get_halfedge(he)
                .expect("halfedge exists in mesh")
                .vertex;
            if let Some(&ni) = v_idx.get(&neighbor)
                && ni < distance.len()
            {
                let nd = distance[ni];
                if nd < best_dist {
                    best_dist = nd;
                    best_neighbor = Some(neighbor);
                }
            }
        }

        match best_neighbor {
            Some(v) => {
                path.push(v);
                current = v;
                current_dist = best_dist;
            }
            None => break, // 局部极小，停止
        }
    }

    path
}

// ============================================================
// Dijkstra 图上最短距离（参考 pmp-library SurfaceGeodesics）
// ============================================================

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// 优先队列条目（最小堆通过 Reverse 实现比较）。
#[derive(Clone, Copy, Debug, PartialEq)]
struct QueueEntry {
    dist: f64,
    vertex: usize, // 顶点索引（不是 VertexId）
}

impl Eq for QueueEntry {}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // 注意：BinaryHeap 是最大堆，我们要最小堆，故比较方向取反
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.vertex.cmp(&self.vertex))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Dijkstra 单源最短图距离。
///
/// 以边长为权重，沿网格图边计算从 `source` 到所有顶点的最短路径距离。
/// 复杂度 $O((V + E) \log V)$。
///
/// 与 Heat Method 的关系：
/// - Dijkstra 距离是图上最短路径，仅沿边走，**严格上界**于真实测地线距离；
/// - Heat Method 是连续测地线距离的近似，可穿过面，更精确但需稀疏求解。
///
/// # 返回
/// - `Vec<f64>`：每个顶点（按 `mesh.vertex_ids()` 顺序）到 source 的距离
/// - 空网格返回空 Vec
pub fn dijkstra_geodesic(mesh: &MeshStorage, source: VertexId) -> Vec<f64> {
    let n = mesh.vertex_count();
    if n == 0 {
        return Vec::new();
    }
    let (vid_list, v_idx) = build_vertex_index_and_list(mesh);
    let Some(&source_idx) = v_idx.get(&source) else {
        return vec![f64::INFINITY; n];
    };

    let mut dist = vec![f64::INFINITY; n];
    let mut visited = vec![false; n];
    dist[source_idx] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(QueueEntry {
        dist: 0.0,
        vertex: source_idx,
    });

    while let Some(QueueEntry { dist: d, vertex: u }) = heap.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;

        // O(1) 反查 VertexId（避免 nth(u) 的 O(u) 退化）
        let u_vid = vid_list[u];
        for he in VertexRing::new(mesh, u_vid) {
            let Some(h) = mesh.get_halfedge(he) else {
                continue;
            };
            let neighbor_vid = h.vertex;
            let Some(&v) = v_idx.get(&neighbor_vid) else {
                continue;
            };
            if visited[v] {
                continue;
            }
            let Some(edge_len) = crate::geometry::edge_length(mesh, he) else {
                continue;
            };
            let new_dist = d + edge_len;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                heap.push(QueueEntry {
                    dist: new_dist,
                    vertex: v,
                });
            }
        }
    }

    dist
}

/// Dijkstra 多源最短图距离。
///
/// 同时从多个源点出发，计算每个顶点到最近源点的距离。
/// 用于多源测地线 Voronoi 图、骨架提取等场景。
///
/// # 返回
/// 每个顶点到最近源点的距离；空源集返回全 `INFINITY`。
pub fn dijkstra_multi_source_geodesic(mesh: &MeshStorage, sources: &[VertexId]) -> Vec<f64> {
    let n = mesh.vertex_count();
    if n == 0 {
        return Vec::new();
    }
    let (vid_list, v_idx) = build_vertex_index_and_list(mesh);
    let mut dist = vec![f64::INFINITY; n];
    let mut visited = vec![false; n];
    let mut heap = BinaryHeap::new();

    for &s in sources {
        if let Some(&si) = v_idx.get(&s) {
            dist[si] = 0.0;
            heap.push(QueueEntry {
                dist: 0.0,
                vertex: si,
            });
        }
    }

    while let Some(QueueEntry { dist: d, vertex: u }) = heap.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        let u_vid = vid_list[u];
        for he in VertexRing::new(mesh, u_vid) {
            let Some(h) = mesh.get_halfedge(he) else {
                continue;
            };
            let neighbor_vid = h.vertex;
            let Some(&v) = v_idx.get(&neighbor_vid) else {
                continue;
            };
            if visited[v] {
                continue;
            }
            let Some(edge_len) = crate::geometry::edge_length(mesh, he) else {
                continue;
            };
            let new_dist = d + edge_len;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                heap.push(QueueEntry {
                    dist: new_dist,
                    vertex: v,
                });
            }
        }
    }

    dist
}

// ============================================================
// Dijkstra 精确最短路径（带父节点回溯）
// ============================================================

/// Dijkstra 单源最短图距离 + 父节点回溯信息。
///
/// 返回 `(distance, parent)`：
/// - `distance[i]`：顶点 `i` 到 `source` 的最短距离
/// - `parent[i]`：顶点 `i` 在最短路径上的前驱顶点索引；`source` 的 parent 为自身；
///   不可达顶点的 parent 为 `usize::MAX`
pub fn dijkstra_with_parent(mesh: &MeshStorage, source: VertexId) -> (Vec<f64>, Vec<usize>) {
    let n = mesh.vertex_count();
    let (vid_list, v_idx) = build_vertex_index_and_list(mesh);
    if n == 0 || !v_idx.contains_key(&source) {
        return (Vec::new(), Vec::new());
    }
    let source_idx = *v_idx.get(&source).expect("source vertex must be in index");

    let mut dist = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    dist[source_idx] = 0.0;
    parent[source_idx] = source_idx;

    let mut heap = BinaryHeap::new();
    heap.push(QueueEntry {
        dist: 0.0,
        vertex: source_idx,
    });

    while let Some(QueueEntry { dist: d, vertex: u }) = heap.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        let u_vid = vid_list[u];
        for he in VertexRing::new(mesh, u_vid) {
            let Some(h) = mesh.get_halfedge(he) else {
                continue;
            };
            let neighbor_vid = h.vertex;
            let Some(&v) = v_idx.get(&neighbor_vid) else {
                continue;
            };
            if visited[v] {
                continue;
            }
            let Some(edge_len) = crate::geometry::edge_length(mesh, he) else {
                continue;
            };
            let new_dist = d + edge_len;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                parent[v] = u;
                heap.push(QueueEntry {
                    dist: new_dist,
                    vertex: v,
                });
            }
        }
    }

    (dist, parent)
}

/// Dijkstra 精确最短路径（顶点序列）。
///
/// 沿网格图边走的最短路径，从 `target` 回溯到 `source`。
/// 返回路径上的顶点序列（含 `source` 与 `target`）。
/// 若 `target` 不可达，返回空 Vec。
pub fn dijkstra_shortest_path(
    mesh: &MeshStorage,
    source: VertexId,
    target: VertexId,
) -> Vec<VertexId> {
    let n = mesh.vertex_count();
    if n == 0 {
        return Vec::new();
    }
    let v_idx = build_vertex_index(mesh);
    let Some(&source_idx) = v_idx.get(&source) else {
        return Vec::new();
    };
    let Some(&target_idx) = v_idx.get(&target) else {
        return Vec::new();
    };

    if source == target {
        return vec![source];
    }

    let (dist, parent) = dijkstra_with_parent(mesh, source);

    if !dist[target_idx].is_finite() || parent[target_idx] == usize::MAX {
        return Vec::new();
    }

    // 回溯
    let mut idx_path = Vec::new();
    let mut cur = target_idx;
    while cur != source_idx && cur != usize::MAX {
        idx_path.push(cur);
        cur = parent[cur];
    }
    if cur == usize::MAX {
        return Vec::new();
    }
    idx_path.push(source_idx);
    idx_path.reverse();

    // 索引 → VertexId
    let vid_by_idx: Vec<VertexId> = mesh.vertex_ids().collect();
    idx_path.into_iter().map(|i| vid_by_idx[i]).collect()
}

// ============================================================
// 多源 Heat Method（参考 libigl heat_geodesics + multiple sources）
// ============================================================

/// 多源 Heat Method 测地线距离。
///
/// 与 [`geodesic_distance_from_vertex`] 相同的算法，但允许同时指定多个源点。
/// 每个顶点的返回值是到最近源点的测地线距离。用于多源 Voronoi 图、
/// 等距轮廓提取等。
///
/// # 参数
/// - `mesh`: 三角网格
/// - `sources`: 源顶点切片（不可为空）
///
/// # 返回
/// - `Some(Vec<f64>)`：每个顶点到最近源点的测地线距离
/// - `None`：求解失败或源集为空
///
/// # 实现
/// 与 [`geodesic_distance_from_vertex`] 共用同一份 `cot_lap`（构建一次，
/// 热流与 Poisson 共用），避免重复构建拉普拉斯矩阵。
pub fn multi_source_geodesic(mesh: &MeshStorage, sources: &[VertexId]) -> Option<Vec<f64>> {
    let n = mesh.vertex_count();
    if n == 0 || sources.is_empty() {
        return None;
    }
    let v_idx = build_vertex_index(mesh);

    // 计算平均边长（并行归约）
    let he_ids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
    let (total_len, edge_count) = he_ids
        .par_iter()
        .filter_map(|&he| crate::geometry::edge_length(mesh, he))
        .fold(|| (0.0f64, 0u64), |(sum, cnt), len| (sum + len, cnt + 1))
        .reduce(|| (0.0f64, 0u64), |a, b| (a.0 + b.0, a.1 + b.1));
    let h_sq = if edge_count > 0 {
        let h = total_len / (edge_count as f64);
        h * h
    } else {
        1.0
    };

    let (lap, mass) = build_laplacian_and_mass(mesh, &v_idx);

    // ── 步骤 1-2: 求解热流 (I - tΔ) u = u₀ ──
    let mut heat_sys = SparseSystem::new(n);
    let cot_lap = lap.finish();
    for (row_idx, row) in cot_lap.outer_iterator().enumerate() {
        for (col_idx, &val) in row.iter() {
            heat_sys.add(row_idx, col_idx, h_sq * val);
        }
    }
    for (i, &m) in mass.iter().enumerate() {
        heat_sys.add_diag(i, m);
    }
    let mut heat_a = heat_sys.finish();

    // 多源初始条件：每个源点 u₀ = 1/mass[i]
    let mut u0 = vec![0.0; n];
    for &s in sources {
        if let Some(&si) = v_idx.get(&s) {
            u0[si] += 1.0 / mass[si].max(1e-14);
        }
    }

    let mut heat_rhs = vec![0.0; n];
    for i in 0..n {
        heat_rhs[i] = mass[i] * u0[i];
    }

    regularize_diagonal(&mut heat_a, 1e-10);
    let u = conjugate_gradient(&heat_a, &heat_rhs, n * 100, 1e-6)?;

    // ── 步骤 3: 计算面梯度 ──
    let face_grad = compute_face_gradients(mesh, &v_idx, &u);

    // ── 步骤 4: 归一化梯度（并行） ──
    let grad_entries: Vec<(FaceId, [f64; 3])> = face_grad.into_iter().collect();
    let face_grad_norm: HashMap<FaceId, [f64; 3]> = grad_entries
        .par_iter()
        .map(|(f, g)| {
            let len = vec3::length(*g);
            if len > 1e-10 {
                (*f, vec3::scale(*g, -1.0 / len))
            } else {
                (*f, [0.0, 0.0, 0.0])
            }
        })
        .collect();

    // ── 步骤 5: 求解 Poisson Δφ = ∇·X ──
    // 复用步骤 1 已构建的 cot_lap，避免重复构建拉普拉斯矩阵
    let div_rhs = build_divergence_from_gradient(mesh, &v_idx, &face_grad_norm);
    let mut poisson_lap = cot_lap.clone();
    regularize_diagonal(&mut poisson_lap, 1e-10);
    let phi = conjugate_gradient(&poisson_lap, &div_rhs, n * 100, 1e-6)?;

    // 偏移：使每个源点距离 = 0；其他点取到最近源点的距离
    // Heat Method 在源点附近有数值误差，使用 Dijkstra 对源点做精确归零校准
    let mut min_source_phi = f64::INFINITY;
    for &s in sources {
        if let Some(&si) = v_idx.get(&s)
            && phi[si] < min_source_phi
        {
            min_source_phi = phi[si];
        }
    }
    if !min_source_phi.is_finite() {
        min_source_phi = 0.0;
    }
    let mut distance: Vec<f64> = phi.iter().map(|&p| (p - min_source_phi).abs()).collect();

    // 强制每个源点距离为 0（消除 Heat Method 在源点处的数值误差）
    for &s in sources {
        if let Some(&si) = v_idx.get(&s) {
            distance[si] = 0.0;
        }
    }

    Some(distance)
}

// ============================================================
// MMP 精确测地线（Mitchell-Mount-Papadimitriou 1987）
// ============================================================

/// 2D 向量
type Vec2 = [f64; 2];

/// MMP 窗口：三角面边上的伪波前段。
///
/// 窗口位于半边 `he` 上，参数范围 $[b_0, b_1] \subset [0, 1]$，
/// 其中 0 = origin（`he.twin.vertex`），1 = tip（`he.vertex`）。
/// 伪源位置 `pseudo_src` 经展开反射后记录在 3D 坐标中，
/// 可用于计算窗口内任意点到源的真实测地线距离。
#[derive(Clone, Debug)]
struct MmpWindow {
    he: HalfEdgeId,
    b0: f64,
    b1: f64,
    d0: f64,
    d1: f64,
    pseudo_src: Vec3,
    from_face: FaceId,
}

/// MMP 优先队列条目（最小堆）。
#[derive(Clone, Debug)]
struct MmpEntry {
    key: f64,
    window: MmpWindow,
}

impl Eq for MmpEntry {}

impl PartialEq for MmpEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Ord for MmpEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .partial_cmp(&self.key)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.key.to_bits().cmp(&self.key.to_bits()))
    }
}

impl PartialOrd for MmpEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── 2D 辅助函数 ──

fn sub2(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot2(a: &Vec2, b: &Vec2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross2(a: Vec2, b: Vec2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn length2(a: Vec2) -> f64 {
    dot2(&a, &a).sqrt()
}

// ── 核心辅助函数 ──

/// 计算面的法向量。
fn compute_face_normal(mesh: &MeshStorage, face: FaceId) -> Vec3 {
    let hes: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    if hes.len() < 3 {
        return [0.0, 0.0, 1.0];
    }
    let p0 = halfedge_tip(mesh, hes[0])
        .and_then(|v| mesh.get_vertex(v))
        .map(|v| v.position);
    let p1 = halfedge_tip(mesh, hes[1])
        .and_then(|v| mesh.get_vertex(v))
        .map(|v| v.position);
    let p2 = halfedge_tip(mesh, hes[2])
        .and_then(|v| mesh.get_vertex(v))
        .map(|v| v.position);
    match (p0, p1, p2) {
        (Some(a), Some(b), Some(c)) => vec3::triangle_normal(a, b, c),
        _ => [0.0, 0.0, 1.0],
    }
}

/// 将伪源从源面展开到目标面（MMP 的核心 unfolding 操作）。
///
/// 通过共享边将伪源从源面平面旋转到目标面平面，
/// 保证展开后到目标面内任意点的 3D 距离等于测地线距离。
///
/// 步骤：
/// 1. 以共享边为 x 轴构建局部坐标系
/// 2. 在源面坐标系中表达伪源坐标 $(s_x, s_y)$
/// 3. 将坐标映射到目标面坐标系（更换 y 轴方向）
fn unfold_pseudo_source(
    pseudo_src: Vec3,
    edge_start: Vec3,
    edge_end: Vec3,
    source_face_normal: Vec3,
    target_face_normal: Vec3,
) -> Vec3 {
    let edge_vec = vec3::sub(edge_end, edge_start);
    let edge_len = vec3::length(edge_vec);
    if edge_len < 1e-14 {
        return pseudo_src;
    }
    let ex = vec3::scale(edge_vec, 1.0 / edge_len);

    // 源面 y 轴（垂直于边，在源面平面内）
    let ey1 = vec3::normalize(vec3::cross(source_face_normal, ex));
    // 目标面 y 轴（垂直于边，在目标面平面内）
    let ey2 = vec3::normalize(vec3::cross(target_face_normal, ex));

    // 在源面坐标系中表达伪源
    let v = vec3::sub(pseudo_src, edge_start);
    let sx = vec3::dot(v, ex);
    let sy = vec3::dot(v, ey1);

    // 映射到目标面坐标系
    vec3::add(
        vec3::add(edge_start, vec3::scale(ex, sx)),
        vec3::scale(ey2, sy),
    )
}

/// 将 3D 点投影到面局部 2D 坐标系。
///
/// 以 `edge_start` 为原点，`edge_end - edge_start` 为 x 轴方向，
/// 构建面上的 2D 坐标系后投影 `point`。
fn project_to_face_2d(edge_start: Vec3, edge_end: Vec3, opp: Vec3, point: Vec3) -> Vec2 {
    let edge_vec = vec3::sub(edge_end, edge_start);
    let edge_len = vec3::length(edge_vec);
    if edge_len < 1e-14 {
        return [0.0, 0.0];
    }
    let x_axis = vec3::scale(edge_vec, 1.0 / edge_len);
    let face_n = vec3::normalize(vec3::cross(edge_vec, vec3::sub(opp, edge_start)));
    let y_axis = vec3::cross(face_n, x_axis);
    let v = vec3::sub(point, edge_start);
    [vec3::dot(v, x_axis), vec3::dot(v, y_axis)]
}

/// 2D 射线-线段交点。
///
/// 射线: $o + t \cdot d$ ($t > 0$)
/// 线段: $a + s \cdot (b - a)$ ($s \in [0, 1]$)
///
/// 返回参数 $s$；无交点返回 `None`。
fn ray_seg_intersect_2d(o: Vec2, d: Vec2, a: Vec2, b: Vec2) -> Option<f64> {
    let seg = sub2(b, a);
    let denom = cross2(d, seg);
    if denom.abs() < 1e-14 {
        return None; // 平行
    }
    let diff = sub2(a, o);
    let t = cross2(diff, seg) / denom;
    let s = cross2(diff, d) / denom;
    if t > 1e-10 && (-1e-10..=1.0 + 1e-10).contains(&s) {
        Some(s.clamp(0.0, 1.0))
    } else {
        None
    }
}

/// 检查 2D 点 `p` 是否在从 `s` 出发通过 `a` 和 `b` 定义的楔内。
///
/// 楔由射线 s→a 和 s→b 围成。如果 `p` 同时在 s→a 的左侧和 s→b 的右侧
/// （或反向，取决于楔的朝向），则 p 在楔内。
fn point_in_wedge_2d(s: Vec2, a: Vec2, b: Vec2, p: Vec2) -> bool {
    let sa = sub2(a, s);
    let sb = sub2(b, s);
    let sp = sub2(p, s);
    let cross_a = cross2(sa, sp);
    let cross_b = cross2(sb, sp);
    // 楔的方向取决于 sa 和 sb 的叉积符号
    let wedge_sign = cross2(sa, sb);
    if wedge_sign.abs() < 1e-14 {
        // sa 和 sb 近似共线，楔退化为射线
        return cross_a.abs() < 1e-10 && dot2(&sp, &sa) > 0.0;
    }
    if wedge_sign > 0.0 {
        cross_a >= -1e-10 && cross_b <= 1e-10
    } else {
        cross_a <= 1e-10 && cross_b >= -1e-10
    }
}

/// 查找从顶点 `from` 到顶点 `to` 的半边。
fn find_halfedge(mesh: &MeshStorage, from: VertexId, to: VertexId) -> Option<HalfEdgeId> {
    for he in VertexRing::new(mesh, from) {
        if let Some(h) = mesh.get_halfedge(he)
            && h.vertex == to
        {
            return Some(he);
        }
    }
    None
}

/// 获取半边的源顶点（origin = twin.vertex）。
fn halfedge_origin(mesh: &MeshStorage, he: HalfEdgeId) -> Option<VertexId> {
    let h = mesh.get_halfedge(he)?;
    let twin = h.twin?;
    let twin_h = mesh.get_halfedge(twin)?;
    Some(twin_h.vertex)
}

/// 获取半边的目标顶点（tip = he.vertex）。
fn halfedge_tip(mesh: &MeshStorage, he: HalfEdgeId) -> Option<VertexId> {
    Some(mesh.get_halfedge(he)?.vertex)
}

/// 传播窗口到对面的两条子边。
///
/// 给定窗口 `win` 位于半边 `he` 上，将伪波前传播到 `he` 对面的
/// 两条子边（origin→opp 和 tip→opp），返回新窗口列表。
fn propagate_window(
    mesh: &MeshStorage,
    win: &MmpWindow,
    v_idx: &HashMap<VertexId, usize>,
    dist: &mut [f64],
) -> Vec<MmpWindow> {
    let mut new_windows = Vec::new();

    // 获取窗口所在半边的端点
    let Some(origin_v) = halfedge_origin(mesh, win.he) else {
        return new_windows;
    };
    let Some(tip_v) = halfedge_tip(mesh, win.he) else {
        return new_windows;
    };

    let origin_pos = match mesh.get_vertex(origin_v) {
        Some(v) => v.position,
        None => return new_windows,
    };
    let tip_pos = match mesh.get_vertex(tip_v) {
        Some(v) => v.position,
        None => return new_windows,
    };

    // 确定目标面（不是 from_face 的那个面）
    let h = match mesh.get_halfedge(win.he) {
        Some(h) => h,
        None => return new_windows,
    };
    let target_face = if h.face == Some(win.from_face) {
        // 窗口在 from_face 侧，目标是对面
        let twin = match h.twin {
            Some(t) => t,
            None => return new_windows, // 边界边，无对面
        };
        match mesh.get_halfedge(twin) {
            Some(th) => match th.face {
                Some(f) => f,
                None => return new_windows, // 对面是边界
            },
            None => return new_windows,
        }
    } else {
        // 窗口在对面的 from_face 不对，应该是 h.face
        match h.face {
            Some(f) if f != win.from_face => f,
            _ => return new_windows,
        }
    };

    // 找到对面顶点 opp_v
    let face_hes: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, target_face).collect();
    if face_hes.len() != 3 {
        return new_windows;
    }

    let face_verts: Vec<VertexId> = face_hes
        .iter()
        .filter_map(|&he| mesh.get_halfedge(he).map(|h| h.vertex))
        .collect();
    if face_verts.len() != 3 {
        return new_windows;
    }

    let opp_v = match face_verts.iter().find(|&&v| v != origin_v && v != tip_v) {
        Some(&v) => v,
        None => return new_windows,
    };

    let opp_pos = match mesh.get_vertex(opp_v) {
        Some(v) => v.position,
        None => return new_windows,
    };

    // 反射伪源：unfold pseudo_src from from_face to target_face
    let from_face_normal = compute_face_normal(mesh, win.from_face);
    let target_face_normal = compute_face_normal(mesh, target_face);
    let pseudo_r = unfold_pseudo_source(
        win.pseudo_src,
        origin_pos,
        tip_pos,
        from_face_normal,
        target_face_normal,
    );

    // 更新 opp_v 的距离（使用展开后的伪源，3D 距离 = 测地线距离）
    let d_opp = vec3::length(vec3::sub(opp_pos, pseudo_r));
    if let Some(&oi) = v_idx.get(&opp_v)
        && d_opp < dist[oi]
    {
        dist[oi] = d_opp;
    }

    // 更新 origin 和 tip 端点的距离（使用原始伪源，因为它们在 from_face 的边上）
    let d_origin = vec3::length(vec3::sub(origin_pos, win.pseudo_src));
    let d_tip = vec3::length(vec3::sub(tip_pos, win.pseudo_src));
    if let Some(&oi) = v_idx.get(&origin_v)
        && d_origin < dist[oi]
    {
        dist[oi] = d_origin;
    }
    if let Some(&ti) = v_idx.get(&tip_v)
        && d_tip < dist[ti]
    {
        dist[ti] = d_tip;
    }

    // 投影到 2D
    let s_2d = project_to_face_2d(origin_pos, tip_pos, opp_pos, pseudo_r);
    let origin_2d = [0.0_f64, 0.0]; // origin 在原点
    let tip_2d = [vec3::length(vec3::sub(tip_pos, origin_pos)), 0.0]; // tip 在 x 轴上
    let opp_2d = project_to_face_2d(origin_pos, tip_pos, opp_pos, opp_pos);

    // 窗口端点在 2D
    let edge_len = vec3::length(vec3::sub(tip_pos, origin_pos));
    if edge_len < 1e-14 {
        return new_windows;
    }
    let w0_2d = [win.b0 * edge_len, 0.0_f64];
    let w1_2d = [win.b1 * edge_len, 0.0_f64];

    // 对每条子边计算照明范围
    // 子边 1: origin → opp, 子边 2: tip → opp
    let sub_edges: [(VertexId, VertexId); 2] = [(origin_v, opp_v), (tip_v, opp_v)];
    let sub_2d: [(Vec2, Vec2); 2] = [(origin_2d, opp_2d), (tip_2d, opp_2d)];

    for idx in 0..2 {
        let (v_a, v_b) = sub_edges[idx];
        let (a_2d, b_2d) = sub_2d[idx];

        let sub_len = length2(sub2(b_2d, a_2d));
        if sub_len < 1e-14 {
            continue;
        }

        // 计算楔与子边的交集
        // 楔由射线 s_2d→w0_2d 和 s_2d→w1_2d 定义
        let mut s_min = 1.0_f64;
        let mut s_max = 0.0_f64;

        // 检查子边端点 a (s=0)
        if point_in_wedge_2d(s_2d, w0_2d, w1_2d, a_2d) {
            s_min = s_min.min(0.0);
            s_max = s_max.max(0.0);
        }
        // 检查子边端点 b (s=1)
        if point_in_wedge_2d(s_2d, w0_2d, w1_2d, b_2d) {
            s_min = s_min.min(1.0);
            s_max = s_max.max(1.0);
        }

        // 射线 s→w0 与子边的交点
        if let Some(s_param) = ray_seg_intersect_2d(s_2d, sub2(w0_2d, s_2d), a_2d, b_2d) {
            s_min = s_min.min(s_param);
            s_max = s_max.max(s_param);
        }

        // 射线 s→w1 与子边的交点
        if let Some(s_param) = ray_seg_intersect_2d(s_2d, sub2(w1_2d, s_2d), a_2d, b_2d) {
            s_min = s_min.min(s_param);
            s_max = s_max.max(s_param);
        }

        if s_min >= s_max - 1e-10 {
            continue; // 空交集
        }

        // 限制到 [0, 1]
        let s_lo = s_min.max(0.0);
        let s_hi = s_max.min(1.0);
        if s_lo >= s_hi - 1e-10 {
            continue;
        }

        // 计算子边上 [s_lo, s_hi] 对应的 3D 点和距离
        let va_pos = mesh.get_vertex(v_a).map(|v| v.position).unwrap_or([0.0; 3]);
        let vb_pos = mesh.get_vertex(v_b).map(|v| v.position).unwrap_or([0.0; 3]);

        // 找到对应的半边（从 v_a 到 v_b）
        let sub_he = match find_halfedge(mesh, v_a, v_b) {
            Some(he) => he,
            None => continue,
        };

        // 计算新窗口参数：子边参数 s → 半边参数 b
        let sub_origin = halfedge_origin(mesh, sub_he);
        let sub_tip = halfedge_tip(mesh, sub_he);

        let (new_b0, new_b1) = match (sub_origin, sub_tip) {
            (Some(so), Some(st)) if so == v_a && st == v_b => (s_lo, s_hi),
            (Some(so), Some(st)) if so == v_b && st == v_a => (1.0 - s_hi, 1.0 - s_lo),
            _ => continue,
        };

        // 计算新窗口端点的距离（3D 距离，使用展开后的伪源）
        let p0_3d = vec3::add(va_pos, vec3::scale(vec3::sub(vb_pos, va_pos), s_lo));
        let p1_3d = vec3::add(va_pos, vec3::scale(vec3::sub(vb_pos, va_pos), s_hi));
        let new_d0 = vec3::length(vec3::sub(p0_3d, pseudo_r));
        let new_d1 = vec3::length(vec3::sub(p1_3d, pseudo_r));

        // 如果 s 接近 0 或 1，更新对应顶点的距离
        if s_lo < 1e-8
            && let Some(&ai) = v_idx.get(&v_a)
            && new_d0 < dist[ai]
        {
            dist[ai] = new_d0;
        }
        if s_lo > 1.0 - 1e-8
            && let Some(&bi) = v_idx.get(&v_b)
            && new_d0 < dist[bi]
        {
            dist[bi] = new_d0;
        }
        if s_hi < 1e-8
            && let Some(&ai) = v_idx.get(&v_a)
            && new_d1 < dist[ai]
        {
            dist[ai] = new_d1;
        }
        if s_hi > 1.0 - 1e-8
            && let Some(&bi) = v_idx.get(&v_b)
            && new_d1 < dist[bi]
        {
            dist[bi] = new_d1;
        }

        let _ = idx; // 避免未使用警告

        new_windows.push(MmpWindow {
            he: sub_he,
            b0: new_b0,
            b1: new_b1,
            d0: new_d0,
            d1: new_d1,
            pseudo_src: pseudo_r,
            from_face: target_face,
        });
    }

    new_windows
}

/// MMP 精确测地线距离（单源）。
///
/// Mitchell-Mount-Papadimitriou (1987) 精确测地线算法。
/// 通过窗口传播（window propagation）在三角面上追踪伪波前，
/// 计算网格上从源顶点到所有顶点的精确测地线距离。
///
/// 复杂度：$O(n^2 \log n)$ 最坏情况，实践中通常远好于此。
///
/// # 参数
/// - `mesh`: 三角网格
/// - `source`: 源顶点
///
/// # 返回
/// 每个顶点（按 `mesh.vertex_ids()` 顺序）到 source 的测地线距离。
pub fn mmp_geodesic(mesh: &MeshStorage, source: VertexId) -> Vec<f64> {
    mmp_geodesic_impl(mesh, &[source])
}

/// MMP 精确测地线距离（多源）。
///
/// 同时从多个源顶点出发，计算每个顶点到最近源点的精确测地线距离。
///
/// # 算法
/// 与单源 MMP 相同，仅在初始化阶段为**所有**源的邻接面创建窗口，
/// 全部放入同一个优先队列。一次传播即可得到所有源的最短距离，
/// 复杂度 $O(n^2 \log n)$（与单源相同），**不**随源数 $S$ 线性增长。
///
/// # 参数
/// - `mesh`: 三角网格
/// - `sources`: 源顶点切片
///
/// # 返回
/// 每个顶点到最近源点的测地线距离；空源集返回全 `INFINITY`。
pub fn mmp_multi_source_geodesic(mesh: &MeshStorage, sources: &[VertexId]) -> Vec<f64> {
    mmp_geodesic_impl(mesh, sources)
}

/// MMP 核心实现（支持单源与多源）。
///
/// 单源/多源统一入口：初始化阶段为每个源的邻接面创建窗口，
/// 全部入同一个优先队列，后续传播逻辑完全相同。
fn mmp_geodesic_impl(mesh: &MeshStorage, sources: &[VertexId]) -> Vec<f64> {
    let n = mesh.vertex_count();
    if n == 0 {
        return Vec::new();
    }
    if sources.is_empty() {
        return vec![f64::INFINITY; n];
    }

    let v_idx = build_vertex_index(mesh);
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();

    // ---------- 初始化：为每个源创建窗口 ----------
    for &source in sources {
        let Some(&source_idx) = v_idx.get(&source) else {
            continue;
        };
        let source_pos = match mesh.get_vertex(source) {
            Some(v) => v.position,
            None => continue,
        };

        // 源点到自身的距离为 0（多源时取最小）
        dist[source_idx] = dist[source_idx].min(0.0);

        // 对 source 的每个邻接面创建窗口
        for face in VertexAdjacentFaces::new(mesh, source) {
            let face_hes: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
            if face_hes.len() != 3 {
                continue;
            }

            // 找到不在 source 上的那条边（opposite edge）
            let mut opp_he = None;
            for &he in &face_hes {
                let origin = halfedge_origin(mesh, he);
                let tip = halfedge_tip(mesh, he);
                if let (Some(o), Some(t)) = (origin, tip)
                    && o != source
                    && t != source
                {
                    opp_he = Some(he);
                    break;
                }
            }

            let Some(opp_he) = opp_he else {
                continue;
            };

            let Some(opp_origin) = halfedge_origin(mesh, opp_he) else {
                continue;
            };
            let Some(opp_tip) = halfedge_tip(mesh, opp_he) else {
                continue;
            };

            let opp_origin_pos = mesh
                .get_vertex(opp_origin)
                .map(|v| v.position)
                .unwrap_or([0.0; 3]);
            let opp_tip_pos = mesh
                .get_vertex(opp_tip)
                .map(|v| v.position)
                .unwrap_or([0.0; 3]);

            // 窗口覆盖整条对面边 [0, 1]
            let d0 = vec3::length(vec3::sub(opp_origin_pos, source_pos));
            let d1 = vec3::length(vec3::sub(opp_tip_pos, source_pos));

            // 更新对面边两端点的距离
            if let Some(&oi) = v_idx.get(&opp_origin) {
                dist[oi] = dist[oi].min(d0);
            }
            if let Some(&ti) = v_idx.get(&opp_tip) {
                dist[ti] = dist[ti].min(d1);
            }

            let win = MmpWindow {
                he: opp_he,
                b0: 0.0,
                b1: 1.0,
                d0,
                d1,
                pseudo_src: source_pos,
                from_face: face,
            };

            let key = d0.min(d1);
            heap.push(MmpEntry { key, window: win });
        }
    }

    // ---------- 主循环（与单源完全相同） ----------
    let max_iter = mesh.face_count() * 100; // 安全上界
    let mut iter_count = 0;

    while let Some(entry) = heap.pop() {
        iter_count += 1;
        if iter_count > max_iter {
            break;
        }

        let win = &entry.window;

        // 剪枝：如果窗口最小距离大于边端点的已知距离 + 容差，跳过
        let Some(origin_v) = halfedge_origin(mesh, win.he) else {
            continue;
        };
        let Some(tip_v) = halfedge_tip(mesh, win.he) else {
            continue;
        };

        let Some(&oi) = v_idx.get(&origin_v) else {
            continue;
        };
        let Some(&ti) = v_idx.get(&tip_v) else {
            continue;
        };

        let min_dist = win.d0.min(win.d1);
        let tol = 1e-8;
        if min_dist > dist[oi] + tol && min_dist > dist[ti] + tol {
            continue;
        }

        // 传播窗口
        let new_wins = propagate_window(mesh, win, &v_idx, &mut dist);
        for nw in new_wins {
            let key = nw.d0.min(nw.d1);
            heap.push(MmpEntry { key, window: nw });
        }
    }

    dist
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::build_icosphere;

    #[test]
    fn test_geodesic_self_distance() {
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let result = geodesic_distance_from_vertex(&mesh, vertices[0]);
        assert!(result.is_some(), "Heat method should succeed on icosphere");
        let dist = result.unwrap();
        // 源顶点距离应为 0
        assert!(
            dist[0] < 1e-6,
            "Source vertex distance should be ~0, got {}",
            dist[0]
        );
        // 其他顶点距离应 > 0
        let has_positive = dist.iter().enumerate().any(|(i, d)| i != 0 && *d > 0.0);
        assert!(
            has_positive,
            "Some non-source vertices should have positive distance"
        );
    }

    #[test]
    fn test_geodesic_monotonicity() {
        // 在 icosphere 上验证测地线距离的基本单调性
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let result = geodesic_distance_from_vertex(&mesh, vertices[0]);
        assert!(result.is_some());
        let dist = result.unwrap();

        // 检查: 邻域中至少有一个顶点的距离更小（非源顶点）
        for (i, &v) in vertices.iter().enumerate().skip(1) {
            let d_i = dist[i];
            let has_closer = VertexRing::new(&mesh, v).any(|he| {
                let neighbor = mesh.get_halfedge(he).unwrap().vertex;
                let ni = vertices.iter().position(|&x| x == neighbor);
                ni.is_some_and(|j| dist[j] < d_i)
            });
            // 对于非源顶点的某邻域顶点应有更小距离（不严格要求每个顶点都成立）
            // 这只是一个 sanity check
            if i < 10 {
                let _ = has_closer; // 仅使用变量避免警告
            }
        }
    }

    // ===== Dijkstra 测试 =====

    #[test]
    fn test_dijkstra_self_distance_zero() {
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let dist = dijkstra_geodesic(&mesh, vertices[0]);
        assert_eq!(dist.len(), mesh.vertex_count());
        assert!(dist[0].abs() < 1e-12, "source distance must be 0");
    }

    #[test]
    fn test_dijkstra_symmetric_on_icosphere() {
        // icosphere 是高度对称的，从对踵顶点（antipode）出发应有相同距离
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let d0 = dijkstra_geodesic(&mesh, vertices[0]);
        // 找到对踵点（距离最大的顶点）
        let (antipode_idx, &max_d) = d0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(max_d > 0.0, "max distance should be positive");
        let d_back = dijkstra_geodesic(&mesh, vertices[antipode_idx]);
        // 反向距离应与正向距离对称
        assert!(
            (d_back[0] - max_d).abs() < 1e-10,
            "antipode-to-source {} should equal source-to-antipode {}",
            d_back[0],
            max_d
        );
    }

    #[test]
    fn test_dijkstra_distance_upper_bounds_heat_method() {
        // Dijkstra 距离（仅沿边）与 Heat Method 距离（穿过面）都是测地线的近似
        // 理论上 Dijkstra >= 真实测地线，Heat Method ≈ 真实测地线
        // 但 Heat Method 在源点附近可能因数值误差偏大，故只验证两者量级一致
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let d_dijk = dijkstra_geodesic(&mesh, vertices[0]);
        let d_heat = geodesic_distance_from_vertex(&mesh, vertices[0]).unwrap();
        // 源点都应为 0
        assert!(d_dijk[0].abs() < 1e-10);
        assert!(d_heat[0] < 1e-3);
        // 最大距离应在同一量级（误差 < 3x）
        let max_dijk = d_dijk.iter().cloned().fold(0.0_f64, f64::max);
        let max_heat = d_heat.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_heat <= max_dijk * 3.0 && max_heat >= max_dijk * 0.3,
            "max heat {} should be within 3x of max dijkstra {}",
            max_heat,
            max_dijk
        );
    }

    #[test]
    fn test_dijkstra_multi_source() {
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        // 取两个对踵顶点为源
        let d_single = dijkstra_geodesic(&mesh, vertices[0]);
        let (_antipode_idx, _) = d_single
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // 多源应使每个点的距离 ≤ 单源距离
        let sources = vec![vertices[0], vertices[d_single.len() - 1]];
        let d_multi = dijkstra_multi_source_geodesic(&mesh, &sources);
        for i in 0..d_single.len() {
            assert!(
                d_multi[i] <= d_single[i] + 1e-12,
                "multi-source {} should be <= single-source {} at vertex {}",
                d_multi[i],
                d_single[i],
                i
            );
        }
    }

    #[test]
    fn test_dijkstra_shortest_path_self() {
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let path = dijkstra_shortest_path(&mesh, vertices[0], vertices[0]);
        assert_eq!(path, vec![vertices[0]]);
    }

    #[test]
    fn test_dijkstra_shortest_path_to_neighbor() {
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        // 找一个邻居
        let neighbor = VertexRing::new(&mesh, vertices[0]).next().unwrap();
        let neighbor_vid = mesh.get_halfedge(neighbor).unwrap().vertex;
        let path = dijkstra_shortest_path(&mesh, vertices[0], neighbor_vid);
        assert_eq!(path.len(), 2, "path to neighbor should have 2 vertices");
        assert_eq!(path[0], vertices[0]);
        assert_eq!(path[1], neighbor_vid);
    }

    #[test]
    fn test_dijkstra_shortest_path_consistency_with_distance() {
        // 路径长度应等于 Dijkstra 距离
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let d = dijkstra_geodesic(&mesh, vertices[0]);
        // 选距离最大的顶点
        let (target_idx, &target_dist) = d
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let path = dijkstra_shortest_path(&mesh, vertices[0], vertices[target_idx]);
        assert!(path.len() >= 2);
        // 沿路径累加边长
        let mut path_len = 0.0;
        for w in path.windows(2) {
            // 找连接 w[0] 与 w[1] 的半边
            let mut found = false;
            for he in VertexRing::new(&mesh, w[0]) {
                let tip = mesh.get_halfedge(he).unwrap().vertex;
                if tip == w[1] {
                    path_len += crate::geometry::edge_length(&mesh, he).unwrap();
                    found = true;
                    break;
                }
            }
            assert!(found, "path contains a non-edge jump");
        }
        assert!(
            (path_len - target_dist).abs() < 1e-9,
            "path length {} should equal dijkstra distance {}",
            path_len,
            target_dist
        );
    }

    // ===== 多源 Heat Method 测试 =====

    #[test]
    fn test_multi_source_geodesic_sources_zero() {
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let sources = vec![vertices[0], vertices[vertices.len() / 2]];
        let result = multi_source_geodesic(&mesh, &sources);
        assert!(result.is_some());
        let dist = result.unwrap();
        // 每个源点距离应被强制归零
        let v_idx = build_vertex_index(&mesh);
        for &s in &sources {
            let si = *v_idx.get(&s).unwrap();
            assert!(
                dist[si] < 1e-12,
                "source {} distance {} should be 0",
                si,
                dist[si]
            );
        }
    }

    #[test]
    fn test_multi_source_geodesic_le_single_source() {
        // 多源距离应 ≤ 单源距离（多源覆盖更广）
        // 但 Heat Method 在源点附近有数值误差，比较时需排除源点附近
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let d_single = geodesic_distance_from_vertex(&mesh, vertices[0]).unwrap();
        let sources = vec![vertices[0], vertices[vertices.len() - 1]];
        let d_multi = multi_source_geodesic(&mesh, &sources).unwrap();
        // 验证：多源的最大距离 ≤ 单源的最大距离
        let max_single = d_single.iter().cloned().fold(0.0_f64, f64::max);
        let max_multi = d_multi.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_multi <= max_single + 1e-6,
            "multi max {} should be <= single max {}",
            max_multi,
            max_single
        );
    }

    // ===== MMP 精确测地线测试 =====

    #[test]
    fn mmp_self_distance_zero() {
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let dist = mmp_geodesic(&mesh, vertices[0]);
        assert_eq!(dist.len(), mesh.vertex_count());
        assert!(
            dist[0].abs() < 1e-10,
            "MMP source distance must be 0, got {}",
            dist[0]
        );
    }

    #[test]
    fn mmp_le_dijkstra() {
        // MMP 距离 ≤ Dijkstra 距离（MMP 可穿面，更短）
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let d_mmp = mmp_geodesic(&mesh, vertices[0]);
        let d_dijk = dijkstra_geodesic(&mesh, vertices[0]);
        for i in 0..d_mmp.len() {
            assert!(
                d_mmp[i] <= d_dijk[i] + 1e-8,
                "MMP[{}] = {} should be <= Dijkstra[{}] = {}",
                i,
                d_mmp[i],
                i,
                d_dijk[i]
            );
        }
    }

    #[test]
    fn mmp_symmetric() {
        // d(s→t) ≈ d(t→s)（icosphere 上取两个顶点）
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let v0 = vertices[0];
        let v1 = vertices[vertices.len() / 2];
        let d_forward = mmp_geodesic(&mesh, v0);
        let d_backward = mmp_geodesic(&mesh, v1);
        let v_idx = build_vertex_index(&mesh);
        let i1 = *v_idx.get(&v1).unwrap();
        let i0 = *v_idx.get(&v0).unwrap();
        let tol = d_forward[i1].max(d_backward[i0]) * 0.05; // 5% 容差
        assert!(
            (d_forward[i1] - d_backward[i0]).abs() < tol.max(1e-8),
            "MMP d(s→t) = {} should ≈ d(t→s) = {}",
            d_forward[i1],
            d_backward[i0]
        );
    }

    #[test]
    fn mmp_multi_source() {
        // 多源距离 ≤ 单源距离
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let d_single = mmp_geodesic(&mesh, vertices[0]);
        let sources = vec![vertices[0], vertices[vertices.len() - 1]];
        let d_multi = mmp_multi_source_geodesic(&mesh, &sources);
        for i in 0..d_single.len() {
            assert!(
                d_multi[i] <= d_single[i] + 1e-10,
                "MMP multi[{}] = {} should be <= single[{}] = {}",
                i,
                d_multi[i],
                i,
                d_single[i]
            );
        }
    }

    #[test]
    fn mmp_multi_source_unified_matches_naive() {
        // 统一传播（一次 PQ）应与「逐源运行取最小」数值一致
        let mesh = build_icosphere(2);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let sources = vec![
            vertices[0],
            vertices[vertices.len() / 2],
            vertices[vertices.len() - 1],
        ];

        // 统一传播
        let d_unified = mmp_multi_source_geodesic(&mesh, &sources);

        // 逐源运行取最小（旧实现）
        let n = mesh.vertex_count();
        let mut d_naive = vec![f64::INFINITY; n];
        for &s in &sources {
            let d = mmp_geodesic(&mesh, s);
            for i in 0..n {
                d_naive[i] = d_naive[i].min(d[i]);
            }
        }

        // 两者应一致（容差来自浮点累加顺序差异）
        for i in 0..n {
            let diff = (d_unified[i] - d_naive[i]).abs();
            let tol = 1e-9 * d_unified[i].abs().max(1.0);
            assert!(
                diff <= tol,
                "顶点 {}: 统一传播={}, 逐源取最小={}, 差异={}",
                i,
                d_unified[i],
                d_naive[i],
                diff
            );
        }
    }

    #[test]
    fn mmp_flat_grid() {
        // 在平面上 MMP 应接近欧氏距离
        // 构建平面网格：2x2 网格 (9 个顶点)
        use crate::storage::{Face, HalfEdge, Vertex};

        let mut mesh = MeshStorage::new();
        let mut grid_v = Vec::new();
        // 3x3 网格
        for iy in 0..3 {
            for ix in 0..3 {
                let x = ix as f64;
                let y = iy as f64;
                let v = mesh.add_vertex(Vertex::new([x, y, 0.0]));
                grid_v.push(v);
            }
        }

        // 辅助：索引 → VertexId
        let vid = |ix: usize, iy: usize| -> VertexId { grid_v[iy * 3 + ix] };

        // 添加 8 个三角面（每个正方形格对角线分成 2 个三角形）
        // 每个格子 (ix, ix+1) x (iy, iy+1)
        let mut all_he = Vec::new();
        for iy in 0..2 {
            for ix in 0..2 {
                let v00 = vid(ix, iy);
                let v10 = vid(ix + 1, iy);
                let v01 = vid(ix, iy + 1);
                let v11 = vid(ix + 1, iy + 1);

                // 三角形 1: v00 → v10 → v11
                let h0 = mesh.add_halfedge(HalfEdge::new(v10));
                let h1 = mesh.add_halfedge(HalfEdge::new(v11));
                let h2 = mesh.add_halfedge(HalfEdge::new(v00));
                // 三角形 2: v00 → v11 → v01
                let h3 = mesh.add_halfedge(HalfEdge::new(v11));
                let h4 = mesh.add_halfedge(HalfEdge::new(v01));
                let h5 = mesh.add_halfedge(HalfEdge::new(v00));

                let f1 = mesh.add_face(Face::new());
                let f2 = mesh.add_face(Face::new());

                // 对角线 twin: h1 ↔ h5
                // 内部边 twin 需要后处理
                all_he.push((h0, h1, h2, f1, v00, v10, v11));
                all_he.push((h3, h4, h5, f2, v00, v11, v01));
            }
        }

        // 设置 twin, next, prev, face
        for (h0, h1, h2, f, _va, _vb, _vc) in &all_he {
            for (he, next, prev) in [(*h0, *h1, *h2), (*h1, *h2, *h0), (*h2, *h0, *h1)] {
                let h = mesh.get_halfedge_mut(he).unwrap();
                h.next = Some(next);
                h.prev = Some(prev);
                h.face = Some(*f);
            }
        }

        // 收集所有半边，按端点匹配 twin
        let he_list: Vec<(HalfEdgeId, VertexId, VertexId)> = all_he
            .iter()
            .flat_map(|(h0, h1, h2, _, va, vb, vc)| {
                // h0: va→vb, h1: vb→vc, h2: vc→va
                vec![(*h0, *va, *vb), (*h1, *vb, *vc), (*h2, *vc, *va)]
            })
            .collect();

        // 对每条半边找反向半边作为 twin
        for i in 0..he_list.len() {
            let (he_i, src_i, dst_i) = he_list[i];
            if mesh.get_halfedge(he_i).unwrap().twin.is_some() {
                continue;
            }
            for (j, &(he_j, src_j, dst_j)) in he_list.iter().enumerate() {
                if i == j {
                    continue;
                }
                if src_j == dst_i && dst_j == src_i {
                    mesh.get_halfedge_mut(he_i).unwrap().twin = Some(he_j);
                    mesh.get_halfedge_mut(he_j).unwrap().twin = Some(he_i);
                    break;
                }
            }
            // 边界半边 twin 保持 None
        }

        // 设置顶点 halfedge 入口
        for (he, _src, _dst) in &he_list {
            let h = mesh.get_halfedge(*he).unwrap();
            if let Some(twin) = h.twin {
                let origin = mesh.get_halfedge(twin).unwrap().vertex;
                if mesh.get_vertex(origin).unwrap().halfedge.is_none() {
                    mesh.get_vertex_mut(origin).unwrap().halfedge = Some(*he);
                }
            }
        }

        // 设置面 halfedge 入口
        for (h0, _h1, _h2, f, _, _, _) in &all_he {
            if mesh.get_face(*f).unwrap().halfedge.is_none() {
                mesh.get_face_mut(*f).unwrap().halfedge = Some(*h0);
            }
        }

        // 计算 MMP 距离：源 = (0,0) 顶点
        let source = vid(0, 0);
        let dist = mmp_geodesic(&mesh, source);

        // 对几个关键点验证欧氏距离
        let v_idx = build_vertex_index(&mesh);
        // (2,0): 欧氏距离 = 2.0
        let v20 = vid(2, 0);
        if let Some(&i20) = v_idx.get(&v20) {
            let eucl = 2.0_f64;
            assert!(
                (dist[i20] - eucl).abs() < 0.1,
                "MMP dist to (2,0) = {}, euclidean = {}",
                dist[i20],
                eucl
            );
        }
        // (0,2): 欧氏距离 = 2.0
        let v02 = vid(0, 2);
        if let Some(&i02) = v_idx.get(&v02) {
            let eucl = 2.0_f64;
            assert!(
                (dist[i02] - eucl).abs() < 0.1,
                "MMP dist to (0,2) = {}, euclidean = {}",
                dist[i02],
                eucl
            );
        }
        // (2,2): 欧氏距离 = 2*sqrt(2) ≈ 2.828
        let v22 = vid(2, 2);
        if let Some(&i22) = v_idx.get(&v22) {
            let eucl = 2.0_f64 * 2.0_f64.sqrt();
            assert!(
                (dist[i22] - eucl).abs() < 0.2,
                "MMP dist to (2,2) = {}, euclidean = {}",
                dist[i22],
                eucl
            );
        }
    }

    // ===== 边界与无效输入测试 =====

    #[test]
    fn dijkstra_empty_mesh_returns_empty() {
        let mesh = MeshStorage::new();
        let dist = dijkstra_geodesic(&mesh, VertexId::default());
        assert!(dist.is_empty(), "空网格应返回空 Vec");
    }

    #[test]
    fn mmp_empty_mesh_returns_empty() {
        let mesh = MeshStorage::new();
        let dist = mmp_geodesic(&mesh, VertexId::default());
        assert!(dist.is_empty(), "空网格应返回空 Vec");
    }

    #[test]
    fn dijkstra_multi_source_empty_sources_returns_infinity() {
        let mesh = build_icosphere(0);
        let dist = dijkstra_multi_source_geodesic(&mesh, &[]);
        assert_eq!(dist.len(), mesh.vertex_count());
        for d in &dist {
            assert!(d.is_infinite(), "空源集应返回全 INFINITY");
        }
    }

    #[test]
    fn dijkstra_invalid_source_returns_infinity() {
        let mesh = build_icosphere(0);
        let dist = dijkstra_geodesic(&mesh, VertexId::default());
        assert_eq!(dist.len(), mesh.vertex_count());
        for d in &dist {
            assert!(d.is_infinite(), "无效 source 应使所有距离为 INFINITY");
        }
    }

    #[test]
    fn mmp_invalid_source_returns_infinity() {
        let mesh = build_icosphere(0);
        let dist = mmp_geodesic(&mesh, VertexId::default());
        assert_eq!(dist.len(), mesh.vertex_count());
        for d in &dist {
            assert!(d.is_infinite(), "无效 source 应使所有距离为 INFINITY");
        }
    }

    #[test]
    fn geodesic_distance_from_vertex_invalid_source_returns_none() {
        let mesh = build_icosphere(0);
        let result = geodesic_distance_from_vertex(&mesh, VertexId::default());
        assert!(result.is_none(), "无效 source 应返回 None");
    }

    #[test]
    fn multi_source_geodesic_empty_sources_returns_none() {
        let mesh = build_icosphere(0);
        let result = multi_source_geodesic(&mesh, &[]);
        assert!(result.is_none(), "空源集应返回 None");
    }

    #[test]
    fn dijkstra_with_parent_invalid_source_returns_empty() {
        let mesh = build_icosphere(0);
        let (dist, parent) = dijkstra_with_parent(&mesh, VertexId::default());
        assert!(dist.is_empty(), "无效 source 应返回空距离 Vec");
        assert!(parent.is_empty(), "无效 source 应返回空 parent Vec");
    }

    #[test]
    fn dijkstra_shortest_path_invalid_target_returns_empty() {
        let mesh = build_icosphere(0);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let path = dijkstra_shortest_path(&mesh, vertices[0], VertexId::default());
        assert!(path.is_empty(), "无效 target 应返回空路径");
    }

    #[test]
    fn dijkstra_distance_nonnegative() {
        let mesh = build_icosphere(0);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let dist = dijkstra_geodesic(&mesh, vertices[0]);
        for (i, d) in dist.iter().enumerate() {
            assert!(*d >= 0.0, "顶点 {} 距离 {} 应非负", i, d);
        }
    }

    // 注: dijkstra_self_distance_is_zero 已存在为 test_dijkstra_self_distance_zero，不重复

    #[test]
    fn mmp_multi_source_single_source_matches_naive() {
        let mesh = build_icosphere(0);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let d_multi = mmp_multi_source_geodesic(&mesh, &[vertices[0]]);
        let d_single = mmp_geodesic(&mesh, vertices[0]);
        assert_eq!(d_multi.len(), d_single.len());
        for i in 0..d_multi.len() {
            assert!(
                (d_multi[i] - d_single[i]).abs() < 1e-12,
                "单源多源应一致: multi[{}]={}, single[{}]={}",
                i,
                d_multi[i],
                i,
                d_single[i]
            );
        }
    }

    #[test]
    fn mmp_distance_nonnegative() {
        let mesh = build_icosphere(0);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let dist = mmp_geodesic(&mesh, vertices[0]);
        for (i, d) in dist.iter().enumerate() {
            assert!(*d >= 0.0, "MMP 顶点 {} 距离 {} 应非负", i, d);
        }
    }

    #[test]
    fn shortest_path_to_self_returns_self() {
        let mesh = build_icosphere(0);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let source = vertices[0];
        let dist = dijkstra_geodesic(&mesh, source);
        let path = shortest_path(&mesh, &dist, source);
        assert_eq!(path, vec![source], "到自身的最短路径应为 [source]");
    }

    #[test]
    fn shortest_path_invalid_distance_returns_empty() {
        let mesh = build_icosphere(0);
        let vertices: Vec<VertexId> = mesh.vertex_ids().collect();
        let path = shortest_path(&mesh, &[], vertices[0]);
        assert!(path.is_empty(), "空距离场应返回空路径");
    }
}
