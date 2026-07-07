//! 几何导出模块
//!
//! 提供 [`mesh_to_vertex_index_buffers`]：将半边网格导出为
//! `Vec<[[f32; 3]]>` 顶点 + `Vec<u32>` 索引的扁平形式，可直接传入
//! wgpu 创建 `Buffer`。
//!
//! ## 输出约定
//! - **顶点缓冲**：每个顶点占 3 个 `f32`（位置 x/y/z）；
//! - **索引缓冲**：每个三角面输出 3 个 `u32`（CCW 朝向，与原面边界环一致）；
//! - 顶点索引按 `vertex_ids()` 顺序重新编号为 0-based。
//!
//! ## 兼容性
//! 仅依赖 `f32` 与 `u32`（wgpu 默认索引类型），无外部依赖。
//! 顶点位置从 `f64` 截断为 `f32`，精度足够大多数渲染场景。
//!
//! ## 复杂度
//! $O(V + F)$：每个顶点与每个面常数时间。

use std::collections::HashMap;

use crate::ids::VertexId;
use crate::storage::MeshStorage;
use crate::traversal::FaceHalfEdges;

// ============================================================
// 几何缓冲导出
// ============================================================

/// 将网格导出为顶点缓冲（`Vec<[f32; 3]>`）与索引缓冲（`Vec<u32>`）。
///
/// # 返回
/// `(vertex_buffer, index_buffer)`：
/// - `vertex_buffer`：长度 = `vertex_count() * 3`；
/// - `index_buffer`：长度 = `face_count() * 3`，每三个构成一个三角形。
///
/// # 跳过项
/// - 非三角面（边界环长度 ≠ 3）被跳过；
/// - 顶点缺失（已删除）被跳过。
pub fn mesh_to_vertex_index_buffers(mesh: &MeshStorage) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut v_index: HashMap<VertexId, u32> = HashMap::new();
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(mesh.vertex_count());
    let mut next_idx = 0u32;
    for v_id in mesh.vertex_ids() {
        let Some(v) = mesh.get_vertex(v_id) else {
            continue;
        };
        let pos = v.position;
        vertices.push([pos[0] as f32, pos[1] as f32, pos[2] as f32]);
        v_index.insert(v_id, next_idx);
        next_idx += 1;
    }

    let mut indices: Vec<u32> = Vec::with_capacity(mesh.face_count() * 3);
    let mut skipped: u32 = 0;
    for f_id in mesh.face_ids() {
        let verts: Vec<u32> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| v_index.get(&v).copied())
            .collect();
        if verts.len() != 3 {
            skipped += 1;
            continue; // 跳过非三角面
        }
        indices.extend_from_slice(&verts);
    }
    if skipped > 0 {
        log::warn!("[halfedge::mesh_to_vertex_index_buffers] 警告：跳过 {skipped} 个非三角面");
    }

    (vertices, indices)
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::build_mesh_from_vertices_and_faces;
    use crate::validate::check_topology;

    #[test]
    fn export_basic_quad() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();

        let (vb, ib) = mesh_to_vertex_index_buffers(&mesh);
        assert_eq!(vb.len(), 4);
        assert_eq!(ib.len(), 6);

        // 验证位置正确
        assert_eq!(vb[0], [0.0, 0.0, 0.0]);
        assert_eq!(vb[1], [1.0, 0.0, 0.0]);
        assert_eq!(vb[2], [1.0, 1.0, 0.0]);
        assert_eq!(vb[3], [0.0, 1.0, 0.0]);

        // 检查每个三角形的索引范围
        for i in &ib {
            assert!(*i < 4, "索引 {} 越界", i);
        }
    }

    #[test]
    fn export_tetrahedron() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let (vb, ib) = mesh_to_vertex_index_buffers(&mesh);
        assert_eq!(vb.len(), 4);
        assert_eq!(ib.len(), 12); // 4 面 * 3 索引
    }

    #[test]
    fn export_preserves_ccw_winding() {
        // 确保导出的索引顺序与原面 CCW 一致
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let (vb, ib) = mesh_to_vertex_index_buffers(&mesh);

        // 取出第一个三角形的三个顶点，验证法向为 +z（CCW）
        let a = vb[ib[0] as usize];
        let b = vb[ib[1] as usize];
        let c = vb[ib[2] as usize];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        assert!(n[2] > 0.0, "CCW 朝向 +z 应有正法向 z 分量，实际: {:?}", n);
    }

    #[test]
    fn export_empty_mesh() {
        let mesh = MeshStorage::new();
        let (vb, ib) = mesh_to_vertex_index_buffers(&mesh);
        assert!(vb.is_empty());
        assert!(ib.is_empty());
    }

    #[test]
    fn export_roundtrip_validation() {
        // 导出后再用 io builder 重建，应仍通过校验
        // 注意：面绕向必须一致（每条边在两个面中方向相反），否则 build_mesh 会生成无 twin 的悬空半边
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let (vb, ib) = mesh_to_vertex_index_buffers(&mesh);
        let verts_f64: Vec<[f64; 3]> = vb
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let faces_u32: Vec<[u32; 3]> = ib.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let mesh2 = build_mesh_from_vertices_and_faces(&verts_f64, &faces_u32).unwrap();
        assert!(check_topology(&mesh2).is_ok(), "导出后重建的网格应通过校验");
        assert_eq!(mesh2.vertex_count(), mesh.vertex_count());
        assert_eq!(mesh2.face_count(), mesh.face_count());
    }
}
