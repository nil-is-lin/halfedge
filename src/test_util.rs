//! 测试夹具模块
//!
//! 提供 [`build_icosphere`]：生成单位 icosphere 测试网格，用于快速调试
//! 几何算法、IO、校验等功能。
//!
//! ## 算法
//! 1. **基础 icosahedron**：12 顶点 + 20 三角面，使用黄金比例
//!    $\varphi = \frac{1 + \sqrt{5}}{2}$ 构造，所有顶点归一化到单位球面；
//! 2. **可选细分**：每次细分将每个三角面分裂为 4 个小三角形
//!    （在每条边中点插入新顶点），所有新顶点再次归一化到单位球面。
//!
//! 细分次数 $n$ 与网格规模：
//! $$
//! V_n = 10 \cdot 4^n + 2, \quad F_n = 20 \cdot 4^n, \quad E_n = 30 \cdot 4^n
//! $$
//!
//! ## 用途
//! 调试 [`crate::geometry`]、[`crate::validate`]、[`crate::io`]、
//! [`crate::export`] 等模块的算法正确性。

use std::collections::HashMap;

use crate::io::build_mesh_from_vertices_and_faces;
use crate::storage::MeshStorage;

// ============================================================
// icosphere 生成
// ============================================================

/// 黄金比例。
const PHI: f64 = 1.618_033_988_749_895;

/// icosahedron 的 12 个顶点（未归一化）。
const ICOSA_VERTICES: [[f64; 3]; 12] = [
    [-1.0, PHI, 0.0],
    [1.0, PHI, 0.0],
    [-1.0, -PHI, 0.0],
    [1.0, -PHI, 0.0],
    [0.0, -1.0, PHI],
    [0.0, 1.0, PHI],
    [0.0, -1.0, -PHI],
    [0.0, 1.0, -PHI],
    [PHI, 0.0, -1.0],
    [PHI, 0.0, 1.0],
    [-PHI, 0.0, -1.0],
    [-PHI, 0.0, 1.0],
];

/// icosahedron 的 20 个三角面（CCW 朝向外法向）。
const ICOSA_FACES: [[u32; 3]; 20] = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

/// 构造一个单位 icosphere（球心在原点，半径 1）。
///
/// # 参数
/// - `subdivisions`：细分次数。`0` = 基础 icosahedron（12 顶点 20 面）；
///   `n` = 每个面分裂为 $4^n$ 个小三角形。
///
/// # 复杂度
/// 顶点数 $O(4^n)$，构建时间 $O(4^n)$。
///
/// # 用途
/// 仅用于测试与调试。生产场景请使用更高效的网格生成器。
pub fn build_icosphere(subdivisions: usize) -> MeshStorage {
    let mut vertices: Vec<[f64; 3]> = ICOSA_VERTICES.iter().map(|p| normalize(*p)).collect();
    let mut faces: Vec<[u32; 3]> = ICOSA_FACES.to_vec();

    for _ in 0..subdivisions {
        let (new_verts, new_faces) = subdivide_once(&vertices, &faces);
        vertices = new_verts;
        faces = new_faces;
    }

    build_mesh_from_vertices_and_faces(&vertices, &faces)
}

/// 单次细分：每个三角面分裂为 4 个小三角形，新顶点归一化到单位球面。
fn subdivide_once(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut new_vertices: Vec<[f64; 3]> = vertices.to_vec();
    let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);

    for face in faces {
        let [a, b, c] = *face;
        let ab = midpoint(&mut new_vertices, &mut midpoint_cache, a, b);
        let bc = midpoint(&mut new_vertices, &mut midpoint_cache, b, c);
        let ca = midpoint(&mut new_vertices, &mut midpoint_cache, c, a);
        new_faces.push([a, ab, ca]);
        new_faces.push([b, bc, ab]);
        new_faces.push([c, ca, bc]);
        new_faces.push([ab, bc, ca]);
    }

    (new_vertices, new_faces)
}

/// 取边 (a, b) 的中点（缓存以避免重复创建）。
/// 新中点会被归一化到单位球面。
fn midpoint(
    vertices: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let pa = vertices[a as usize];
    let pb = vertices[b as usize];
    let mid = [
        (pa[0] + pb[0]) * 0.5,
        (pa[1] + pb[1]) * 0.5,
        (pa[2] + pb[2]) * 0.5,
    ];
    let mid_normalized = normalize(mid);
    let idx = vertices.len() as u32;
    vertices.push(mid_normalized);
    cache.insert(key, idx);
    idx
}

#[inline]
fn normalize(p: [f64; 3]) -> [f64; 3] {
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if len < 1e-12 {
        return p;
    }
    [p[0] / len, p[1] / len, p[2] / len]
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::check_topology;
    use crate::{export, geometry};

    #[test]
    fn icosphere_subdiv0_basic() {
        let mesh = build_icosphere(0);
        assert_eq!(mesh.vertex_count(), 12);
        assert_eq!(mesh.face_count(), 20);
        // 30 条边 * 2 半边 = 60
        assert_eq!(mesh.halfedge_count(), 60);
    }

    #[test]
    fn icosphere_subdiv0_passes_validation() {
        let mesh = build_icosphere(0);
        assert!(
            check_topology(&mesh).is_ok(),
            "icosphere(0) 应通过完整校验: {:?}",
            check_topology(&mesh)
        );
    }

    #[test]
    fn icosphere_subdiv1_basic() {
        let mesh = build_icosphere(1);
        // V=42, F=80, E=120 → 半边 240
        assert_eq!(mesh.vertex_count(), 42);
        assert_eq!(mesh.face_count(), 80);
        assert_eq!(mesh.halfedge_count(), 240);
    }

    #[test]
    fn icosphere_subdiv1_passes_validation() {
        let mesh = build_icosphere(1);
        assert!(
            check_topology(&mesh).is_ok(),
            "icosphere(1) 应通过完整校验: {:?}",
            check_topology(&mesh)
        );
    }

    #[test]
    fn icosphere_subdiv2_passes_validation() {
        let mesh = build_icosphere(2);
        assert!(
            check_topology(&mesh).is_ok(),
            "icosphere(2) 应通过完整校验: {:?}",
            check_topology(&mesh)
        );
        // V=162, F=320
        assert_eq!(mesh.vertex_count(), 162);
        assert_eq!(mesh.face_count(), 320);
    }

    #[test]
    fn icosphere_vertices_on_unit_sphere() {
        let mesh = build_icosphere(0);
        for v_id in mesh.vertex_ids() {
            let p = mesh.get_vertex(v_id).unwrap().position;
            let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            assert!(
                (r2 - 1.0).abs() < 1e-9,
                "顶点应在单位球面上，|p|²={} 偏离 1",
                r2
            );
        }
    }

    #[test]
    fn icosphere_face_normals_outward() {
        let mesh = build_icosphere(0);
        let mut all_outward = true;
        for f_id in mesh.face_ids() {
            let Some(n) = geometry::face_normal(&mesh, f_id) else {
                all_outward = false;
                break;
            };
            // 面法向应与面重心方向（即外法向）一致
            let verts: Vec<_> = crate::traversal::FaceHalfEdges::new(&mesh, f_id)
                .filter_map(|he| mesh.get_halfedge(he))
                .map(|h| h.vertex)
                .filter_map(|v| mesh.get_vertex(v))
                .map(|v| v.position)
                .collect();
            if verts.len() != 3 {
                all_outward = false;
                break;
            }
            let centroid = [
                (verts[0][0] + verts[1][0] + verts[2][0]) / 3.0,
                (verts[0][1] + verts[1][1] + verts[2][1]) / 3.0,
                (verts[0][2] + verts[1][2] + verts[2][2]) / 3.0,
            ];
            let dot = n[0] * centroid[0] + n[1] * centroid[1] + n[2] * centroid[2];
            if dot <= 0.0 {
                all_outward = false;
                break;
            }
        }
        assert!(all_outward, "所有面法向应朝外");
    }

    #[test]
    fn icosphere_export_buffers_consistent() {
        let mesh = build_icosphere(1);
        let (vb, ib) = export::mesh_to_vertex_index_buffers(&mesh);
        assert_eq!(vb.len(), 42);
        assert_eq!(ib.len(), 240); // 80 面 * 3
    }
}
