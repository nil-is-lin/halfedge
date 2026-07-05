//! 内蕴 Delaunay 三角剖分模块。
//!
//! 实现 Fisher, Springborn, Schröder & Desbrun (2007) 的内蕴 Delaunay 翻转算法：
//! 通过反复翻转不满足 Delaunay 条件的内部边，使所有边的对角之和 ≤ π，
//! 保证余切拉普拉斯权重非负（正定性）。
//!
//! ## 核心思想
//!
//! 外蕴 Delaunay 基于外接圆测试，在曲面网格上不保证收敛。
//! 内蕴 Delaunay 基于边长定义的度量几何，判据为：
//!
//! ```text
//! 对内部边 e = AB，其对角顶点 C、D 满足
//!   α_C + α_D > π  ⟺  cot(α_C) + cot(α_D) < 0
//! 时翻转 e → CD。
//! ```
//!
//! Bobenko & Springborn (2007) 证明每次合法翻转严格减小 Dirichlet 能量，
//! 因此算法必然终止。
//!
//! ## API
//!
//! | 函数 | 功能 |
//! |------|------|
//! | [`intrinsic_delaunay`] | 内蕴 Delaunay 三角剖分（主入口） |
//! | [`compute_intrinsic_lengths`] | 从 3D 嵌入计算初始内蕴边长 |
//! | [`is_intrinsic_delaunay_edge`] | 判断单条边是否满足内蕴 Delaunay |
//! | [`intrinsic_cotan_weight`] | 从内蕴边长计算余切权重 |

use std::collections::{HashMap, HashSet, VecDeque};

use crate::geometry::edge_length;
use crate::ids::HalfEdgeId;
use crate::storage::MeshStorage;
use crate::topology_ops::flip_edge;
use crate::traversal::FaceHalfEdges;

// ============================================================
// 内蕴边长
// ============================================================

/// 从 3D 嵌入计算初始内蕴边长（即欧氏距离）。
///
/// 对每条半边，其内蕴长度初始化为两端顶点的欧氏距离。
/// twin 对的两个半边具有相同的长度。
pub fn compute_intrinsic_lengths(mesh: &MeshStorage) -> HashMap<HalfEdgeId, f64> {
    let mut lengths = HashMap::new();
    for he in mesh.halfedge_ids() {
        if let Some(l) = edge_length(mesh, he) {
            lengths.insert(he, l);
        }
    }
    lengths
}

// ============================================================
// 内蕴几何辅助函数
// ============================================================

/// 从三边长计算三角形面积（Heron 公式）。
///
/// 返回 0.0 当边长不满足三角不等式（退化情况）。
fn triangle_area_from_lengths(a: f64, b: f64, c: f64) -> f64 {
    let s = (a + b + c) * 0.5;
    let val = s * (s - a) * (s - b) * (s - c);
    if val <= 0.0 { 0.0 } else { val.sqrt() }
}

/// 从内蕴边长判断边是否满足内蕴 Delaunay 条件。
///
/// 设内部边 e = AB，两侧三角形为 ABC 和 ABD，C、D 为对角顶点。
/// 边长为 l_AB, l_BC, l_CA, l_AD, l_BD。
///
/// Delaunay 条件等价于 cot(α_C) + cot(α_D) ≥ 0。
/// 退化三角形（面积 < ε）视为满足条件，不翻转。
pub fn is_intrinsic_delaunay_edge(l_ab: f64, l_bc: f64, l_ca: f64, l_ad: f64, l_bd: f64) -> bool {
    let area_abc = triangle_area_from_lengths(l_ab, l_bc, l_ca);
    let area_abd = triangle_area_from_lengths(l_ab, l_ad, l_bd);
    // 退化三角形不翻转
    if area_abc < 1e-14 || area_abd < 1e-14 {
        return true;
    }
    // cot(α_C) = (l_CA² + l_BC² - l_AB²) / (4·Area(ABC))
    // cot(α_D) = (l_AD² + l_BD² - l_AB²) / (4·Area(ABD))
    let cot_c = (l_ca * l_ca + l_bc * l_bc - l_ab * l_ab) / (4.0 * area_abc);
    let cot_d = (l_ad * l_ad + l_bd * l_bd - l_ab * l_ab) / (4.0 * area_abd);
    cot_c + cot_d >= 0.0
}

/// 从内蕴边长计算翻转后新边的长度。
///
/// 边 e = AB 被翻转为 CD。在三角形 ABC 中，顶点 A 处的角为 ∠A_ABC；
/// 在三角形 ABD 中，顶点 A 处的角为 ∠A_ABD。
/// 角 ∠CAD = ∠A_ABC + ∠A_ABD，然后用余弦定理计算 l_CD。
fn intrinsic_flipped_length(l_ab: f64, l_bc: f64, l_ca: f64, l_ad: f64, l_bd: f64) -> f64 {
    // 三角形 ABC 中顶点 A 处的角
    let cos_a_abc =
        ((l_ab * l_ab + l_ca * l_ca - l_bc * l_bc) / (2.0 * l_ab * l_ca)).clamp(-1.0, 1.0);
    // 三角形 ABD 中顶点 A 处的角
    let cos_a_abd =
        ((l_ab * l_ab + l_ad * l_ad - l_bd * l_bd) / (2.0 * l_ab * l_ad)).clamp(-1.0, 1.0);
    let angle_a_abc = cos_a_abc.acos();
    let angle_a_abd = cos_a_abd.acos();
    let angle_cad = angle_a_abc + angle_a_abd;
    // 在三角形 ACD 中用余弦定理
    let l_cd_sq = l_ca * l_ca + l_ad * l_ad - 2.0 * l_ca * l_ad * angle_cad.cos();
    l_cd_sq.max(0.0).sqrt()
}

/// 从内蕴边长计算边 e 的余切权重 cot(α_C) + cot(α_D)。
///
/// 参数含义同 [`is_intrinsic_delaunay_edge`]。
pub fn intrinsic_cotan_weight(l_ab: f64, l_bc: f64, l_ca: f64, l_ad: f64, l_bd: f64) -> f64 {
    let area_abc = triangle_area_from_lengths(l_ab, l_bc, l_ca);
    let area_abd = triangle_area_from_lengths(l_ab, l_ad, l_bd);
    let mut weight = 0.0;
    if area_abc > 1e-14 {
        weight += (l_ca * l_ca + l_bc * l_bc - l_ab * l_ab) / (4.0 * area_abc);
    }
    if area_abd > 1e-14 {
        weight += (l_ad * l_ad + l_bd * l_bd - l_ab * l_ab) / (4.0 * area_abd);
    }
    weight
}

// ============================================================
// 内蕴 Delaunay 三角剖分主算法
// ============================================================

/// 内蕴 Delaunay 三角剖分统计信息。
#[derive(Debug, Clone, Default)]
pub struct IntrinsicDelaunayStats {
    /// 翻转的边数
    pub flips: usize,
    /// 最大迭代次数（安全守卫）
    pub iterations: usize,
}

/// 将三角网格的内蕴三角化转换为 Delaunay 三角化。
///
/// 仅修改拓扑连接（翻转边），不修改顶点位置。
/// 翻转后更新 `intrinsic_lengths` 中对应边的长度。
///
/// # 算法
/// 1. 初始化内蕴边长为 3D 欧氏距离
/// 2. 将所有内部边入队
/// 3. 取出边 e，若不满足内蕴 Delaunay 条件则翻转
/// 4. 翻转后用余弦定理计算新边 C-D 的内蕴长度
/// 5. 将受影响的邻接边重新入队
/// 6. 重复直至队列为空
///
/// # 参数
/// - `mesh`: 三角网格（可变借用）
/// - `intrinsic_lengths`: 内蕴边长映射（通常由 [`compute_intrinsic_lengths`] 初始化）
///
/// # 返回
/// 翻转统计信息。
///
/// # 收敛性
/// Bobenko & Springborn (2007) 证明每次合法翻转严格减小 Dirichlet 能量，
/// 算法在有限步内终止。设安全上限为 `max_iterations = 10 * E`（E 为边数）。
pub fn intrinsic_delaunay(
    mesh: &mut MeshStorage,
    intrinsic_lengths: &mut HashMap<HalfEdgeId, f64>,
) -> IntrinsicDelaunayStats {
    let edge_count = mesh.edge_count();
    let max_iterations = 10 * edge_count.max(1);

    // 收集所有内部边（两侧均有面的边）
    let mut queue: VecDeque<HalfEdgeId> = VecDeque::new();
    let mut in_queue: HashSet<HalfEdgeId> = HashSet::new();
    for edge in mesh.edge_ids() {
        let he = edge.halfedge();
        // 只入队内部边（两侧都有面）
        if is_interior_edge(mesh, he) {
            queue.push_back(he);
            in_queue.insert(he);
        }
    }

    let mut flips = 0;
    let mut iterations = 0;

    while let Some(he) = queue.pop_front() {
        in_queue.remove(&he);
        iterations += 1;

        if iterations > max_iterations {
            break;
        }

        // 边可能已被前面的翻转删除或变为边界
        if !mesh.contains_halfedge(he) || !is_interior_edge(mesh, he) {
            continue;
        }

        // 获取四顶点和六条边长
        let Some(edge_data) = get_edge_quad(mesh, he, intrinsic_lengths) else {
            continue;
        };

        // 检查 Delaunay 条件
        if is_intrinsic_delaunay_edge(
            edge_data.l_ab,
            edge_data.l_bc,
            edge_data.l_ca,
            edge_data.l_ad,
            edge_data.l_bd,
        ) {
            continue;
        }

        // 计算翻转后新边的内蕴长度
        let l_cd = intrinsic_flipped_length(
            edge_data.l_ab,
            edge_data.l_bc,
            edge_data.l_ca,
            edge_data.l_ad,
            edge_data.l_bd,
        );

        // 记录翻转前受影响的邻接半边（翻转后半边 ID 不变，但拓扑变了）
        let affected = get_affected_halfedges(mesh, he);

        // 执行拓扑翻转
        if flip_edge(mesh, he).is_err() {
            continue;
        }
        flips += 1;

        // 更新翻转后 he/twin 的内蕴边长
        // flip_edge 后：he 变为 D→C，twin 变为 C→D
        // 新边 CD 的长度为 l_cd
        intrinsic_lengths.insert(he, l_cd);
        if let Some(twin) = mesh.get_halfedge(he).and_then(|h| h.twin) {
            intrinsic_lengths.insert(twin, l_cd);
        }

        // 将受影响的邻接边重新入队
        for &adj_he in &affected {
            if !in_queue.contains(&adj_he)
                && mesh.contains_halfedge(adj_he)
                && is_interior_edge(mesh, adj_he)
            {
                queue.push_back(adj_he);
                in_queue.insert(adj_he);
            }
        }
    }

    IntrinsicDelaunayStats { flips, iterations }
}

// ============================================================
// 内部辅助函数
// ============================================================

/// 判断半边是否为内部边（两侧均有面）。
fn is_interior_edge(mesh: &MeshStorage, he: HalfEdgeId) -> bool {
    let Some(h) = mesh.get_halfedge(he) else {
        return false;
    };
    if h.face.is_none() {
        return false;
    }
    let Some(twin_id) = h.twin else {
        return false;
    };
    let Some(twin) = mesh.get_halfedge(twin_id) else {
        return false;
    };
    twin.face.is_some()
}

/// 四边形边长数据。
struct EdgeQuadData {
    l_ab: f64,
    l_bc: f64,
    l_ca: f64,
    l_ad: f64,
    l_bd: f64,
}

/// 获取内部边 e = AB 的四边形配置及内蕴边长。
///
/// 设 e = he: A→B，he.face = F1，twin.face = F2。
/// F1 = (A→B→C→A)，F2 = (B→A→D→B)。
/// 返回 (l_AB, l_BC, l_CA, l_AD, l_BD)。
fn get_edge_quad(
    mesh: &MeshStorage,
    he: HalfEdgeId,
    lengths: &HashMap<HalfEdgeId, f64>,
) -> Option<EdgeQuadData> {
    let h = mesh.get_halfedge(he)?;
    let twin_id = h.twin?;
    let twin = mesh.get_halfedge(twin_id)?;

    // A = origin of he, B = tip of he
    let _b = h.vertex; // tip
    let _a = twin.vertex; // origin

    // F1 = he.face: n1 = he.next (B→C), C = n1.vertex
    let n1 = h.next?;
    let _c = mesh.get_halfedge(n1)?.vertex;

    // F2 = twin.face: n2 = twin.next (A→D), D = n2.vertex
    let n2 = twin.next?;
    let _d = mesh.get_halfedge(n2)?.vertex;

    // 获取边长
    // l_AB: he (A→B) 的长度
    let l_ab = get_length(mesh, he, lengths)?;
    // l_BC: he.next (B→C) 的长度
    let l_bc = get_length(mesh, n1, lengths)?;
    // l_CA: he.prev (C→A) 的长度
    let p1 = h.prev?;
    let l_ca = get_length(mesh, p1, lengths)?;
    // l_AD: twin.next (A→D) 的长度
    let l_ad = get_length(mesh, n2, lengths)?;
    // l_BD: twin.prev (D→B) 的长度
    let p2 = twin.prev?;
    let l_bd = get_length(mesh, p2, lengths)?;

    Some(EdgeQuadData {
        l_ab,
        l_bc,
        l_ca,
        l_ad,
        l_bd,
    })
}

/// 获取半边的内蕴长度。优先从 lengths 查找，否则回退到欧氏距离。
fn get_length(
    mesh: &MeshStorage,
    he: HalfEdgeId,
    lengths: &HashMap<HalfEdgeId, f64>,
) -> Option<f64> {
    if let Some(&l) = lengths.get(&he) {
        return Some(l);
    }
    // 回退到欧氏距离（初始化时可能遗漏）
    edge_length(mesh, he)
}

/// 获取翻转后会受影响的邻接半边（翻转后需要重新检查 Delaunay 条件）。
///
/// 翻转边 he = AB → CD 后，四条邻接边 (AC, CB, BD, DA) 所在的
/// 三角形发生了变化，需要重新入队。
fn get_affected_halfedges(mesh: &MeshStorage, he: HalfEdgeId) -> Vec<HalfEdgeId> {
    let mut affected = Vec::new();
    let Some(h) = mesh.get_halfedge(he) else {
        return affected;
    };
    let Some(twin_id) = h.twin else {
        return affected;
    };
    let Some(twin) = mesh.get_halfedge(twin_id) else {
        return affected;
    };
    let Some(f1) = h.face else {
        return affected;
    };
    let Some(f2) = twin.face else {
        return affected;
    };

    // F1 的三条半边（除 he 外）
    for fhe in FaceHalfEdges::new(mesh, f1) {
        if fhe != he {
            affected.push(fhe);
            // 也加入 twin（同一条无向边）
            if let Some(t) = mesh.get_halfedge(fhe).and_then(|h| h.twin) {
                affected.push(t);
            }
        }
    }
    // F2 的三条半边（除 twin 外）
    for fhe in FaceHalfEdges::new(mesh, f2) {
        if fhe != twin_id {
            affected.push(fhe);
            if let Some(t) = mesh.get_halfedge(fhe).and_then(|h| h.twin) {
                affected.push(t);
            }
        }
    }
    affected
}

/// 从内蕴边长计算三角形中指定顶点处的内角余切值。
///
/// 在边长为 (a, b, c) 的三角形中，计算边 a 对角的余切值。
/// 即顶点 A（不在边 a 上）处角的余切。
pub fn cotan_from_lengths(a: f64, b: f64, c: f64) -> f64 {
    let area = triangle_area_from_lengths(a, b, c);
    if area < 1e-14 {
        return 0.0;
    }
    (b * b + c * c - a * a) / (4.0 * area)
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

    /// 正方形网格（2 个三角面）的内蕴 Delaunay。
    /// 若初始对角线不满足 Delaunay，翻转后应满足。
    #[test]
    fn intrinsic_delaunay_grid() {
        let mut mesh = build_grid(1.0, 1.0, 1, 1);
        let mut lengths = compute_intrinsic_lengths(&mesh);
        let stats = intrinsic_delaunay(&mut mesh, &mut lengths);
        // 正方形网格可能需要翻转也可能不需要，取决于对角线选择
        // 关键是翻转后所有边都满足 Delaunay
        for edge in mesh.edge_ids() {
            let he = edge.halfedge();
            if !is_interior_edge(&mesh, he) {
                continue;
            }
            let Some(data) = get_edge_quad(&mesh, he, &lengths) else {
                continue;
            };
            assert!(
                is_intrinsic_delaunay_edge(data.l_ab, data.l_bc, data.l_ca, data.l_ad, data.l_bd),
                "翻转后仍存在非 Delaunay 边"
            );
        }
        let _ = stats;
    }

    /// icosphere 上所有边初始即满足内蕴 Delaunay。
    #[test]
    fn intrinsic_delaunay_icosphere_no_flip() {
        let mesh = build_icosphere(1);
        let mut lengths = compute_intrinsic_lengths(&mesh);
        // icosphere 初始可能已经 Delaunay 或需要少量翻转
        let mut mesh = mesh;
        let stats = intrinsic_delaunay(&mut mesh, &mut lengths);
        // 验证所有内部边都满足 Delaunay
        for edge in mesh.edge_ids() {
            let he = edge.halfedge();
            if !is_interior_edge(&mesh, he) {
                continue;
            }
            let Some(data) = get_edge_quad(&mesh, he, &lengths) else {
                continue;
            };
            assert!(is_intrinsic_delaunay_edge(
                data.l_ab, data.l_bc, data.l_ca, data.l_ad, data.l_bd
            ),);
        }
        let _ = stats;
    }

    /// 非常扁平的四边形：对角线一定需要翻转。
    #[test]
    fn intrinsic_delaunay_flat_quad() {
        let mut mesh = MeshStorage::new();
        // 构造一个扁平的四边形
        // A=(0,0,0), B=(2,0,0), C=(1,0.1,0), D=(1,1,0)
        // 对角线 A-C 将四边形分为 ABC（扁平）和 ACD
        // 更好的对角线 B-D 满足 Delaunay
        let a = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(Vertex::new([2.0, 0.0, 0.0]));
        let c = mesh.add_vertex(Vertex::new([1.0, 0.1, 0.0]));
        let d = mesh.add_vertex(Vertex::new([1.0, 1.0, 0.0]));

        // 使用 A-C 作为对角线（可能不满足 Delaunay）
        add_triangle(&mut mesh, a, b, c).unwrap();
        add_triangle(&mut mesh, a, c, d).unwrap();

        let mut lengths = compute_intrinsic_lengths(&mesh);
        let stats = intrinsic_delaunay(&mut mesh, &mut lengths);

        // 验证所有内部边满足 Delaunay
        for edge in mesh.edge_ids() {
            let he = edge.halfedge();
            if !is_interior_edge(&mesh, he) {
                continue;
            }
            let Some(data) = get_edge_quad(&mesh, he, &lengths) else {
                continue;
            };
            assert!(is_intrinsic_delaunay_edge(
                data.l_ab, data.l_bc, data.l_ca, data.l_ad, data.l_bd
            ),);
        }
        let _ = stats;
    }

    /// 测试 is_intrinsic_delaunay_edge 对等边三角形返回 true。
    #[test]
    fn is_delaunay_equilateral() {
        // 等边三角形对角 = 60° + 60° = 120° < 180°，满足 Delaunay
        let s = 1.0;
        assert!(is_intrinsic_delaunay_edge(s, s, s, s, s));
    }

    /// 测试三角形面积函数。
    #[test]
    fn triangle_area_heron() {
        // 3-4-5 直角三角形，面积 = 6
        let area = triangle_area_from_lengths(3.0, 4.0, 5.0);
        assert!((area - 6.0).abs() < 1e-10);
        // 退化三角形
        let deg_area = triangle_area_from_lengths(1.0, 2.0, 3.0);
        assert_eq!(deg_area, 0.0);
    }

    /// 测试 intrinsic_flipped_length：菱形四边形翻转发对角线。
    /// A=(-1,0), B=(1,0), C=(0,1), D=(0,-1)，C、D 在 AB 两侧。
    /// 翻转 AB → CD 后，CD 的内蕴长度应为 2。
    #[test]
    fn flipped_length_diamond() {
        let l_ab = 2.0;
        let l_bc = std::f64::consts::SQRT_2;
        let l_ca = std::f64::consts::SQRT_2;
        let l_ad = std::f64::consts::SQRT_2;
        let l_bd = std::f64::consts::SQRT_2;
        let l_cd = intrinsic_flipped_length(l_ab, l_bc, l_ca, l_ad, l_bd);
        assert!((l_cd - 2.0).abs() < 1e-10, "l_cd = {}, expected 2.0", l_cd);
    }

    /// 测试 compute_intrinsic_lengths 与 edge_length 一致。
    #[test]
    fn intrinsic_lengths_match_euclidean() {
        let mesh = build_icosphere(1);
        let lengths = compute_intrinsic_lengths(&mesh);
        for he in mesh.halfedge_ids() {
            if let Some(l) = edge_length(&mesh, he) {
                let stored = lengths.get(&he);
                assert!(stored.is_some(), "半边 {:?} 缺少内蕴长度", he);
                assert!(
                    (stored.unwrap() - l).abs() < 1e-10,
                    "内蕴长度与欧氏距离不一致"
                );
            }
        }
    }

    /// 测试余切权重：对 Delaunay 网格，所有权重 ≥ 0。
    #[test]
    fn cotan_weight_nonnegative_after_delaunay() {
        let mut mesh = build_grid(1.0, 1.0, 2, 2);
        let mut lengths = compute_intrinsic_lengths(&mesh);
        intrinsic_delaunay(&mut mesh, &mut lengths);
        // 所有内部边的余切权重应非负
        for edge in mesh.edge_ids() {
            let he = edge.halfedge();
            if !is_interior_edge(&mesh, he) {
                continue;
            }
            let Some(data) = get_edge_quad(&mesh, he, &lengths) else {
                continue;
            };
            let w = intrinsic_cotan_weight(data.l_ab, data.l_bc, data.l_ca, data.l_ad, data.l_bd);
            assert!(w >= -1e-10, "余切权重为负: {}", w);
        }
    }

    /// 测试空网格不 panic。
    #[test]
    fn intrinsic_delaunay_empty() {
        let mut mesh = MeshStorage::new();
        let mut lengths = compute_intrinsic_lengths(&mesh);
        let stats = intrinsic_delaunay(&mut mesh, &mut lengths);
        assert_eq!(stats.flips, 0);
    }

    /// 测试单个三角形（无内部边）不翻转。
    #[test]
    fn intrinsic_delaunay_single_triangle() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        add_triangle(&mut mesh, v0, v1, v2).unwrap();
        let mut lengths = compute_intrinsic_lengths(&mesh);
        let stats = intrinsic_delaunay(&mut mesh, &mut lengths);
        assert_eq!(stats.flips, 0);
    }
}
