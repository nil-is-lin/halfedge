//! 几何图元生成模块
//!
//! 提供常用三维几何体的程序化生成，返回完整的 `MeshStorage`。

use std::f64::consts::TAU;

use crate::ids::VertexId;
use crate::storage::{MeshStorage, Vertex};
use crate::topology_ops::add_triangle;

// ============================================================
// 立方体
// ============================================================

/// 生成立方体，中心在原点，边长 `size`。
///
/// 顶点按右手系排列，所有面的法向朝外（CCW）。
pub fn build_cube(size: f64) -> MeshStorage {
    let h = size / 2.0;
    let verts = vec![
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ];
    let faces: Vec<[u32; 3]> = vec![
        [0, 3, 2],
        [0, 2, 1], // -z
        [4, 5, 6],
        [4, 6, 7], // +z
        [0, 1, 5],
        [0, 5, 4], // -y
        [3, 7, 6],
        [3, 6, 2], // +y
        [0, 4, 7],
        [0, 7, 3], // -x
        [1, 2, 6],
        [1, 6, 5], // +x
    ];
    crate::io::build_mesh_from_vertices_and_faces(&verts, &faces)
}

// ============================================================
// UV 球体
// ============================================================

/// 生成 UV 球体（经线×纬线），中心在原点，半径 `radius`。
///
/// `slices`：经线分段数（绕 Y 轴）；`stacks`：纬线分段数（从顶到底）。
/// 最小为 3。
pub fn build_uv_sphere(radius: f64, slices: usize, stacks: usize) -> MeshStorage {
    let slices = slices.max(3);
    let stacks = stacks.max(2);

    let mut mesh = MeshStorage::new();
    let cap = (slices + 1) * (stacks + 1);
    mesh.reserve(cap, cap * 6, slices * stacks * 2);

    // 生成顶点（顶→底）
    let mut v_ids: Vec<Vec<VertexId>> = Vec::with_capacity(stacks + 1);
    for i in 0..=stacks {
        let phi = std::f64::consts::PI * i as f64 / stacks as f64; // [0, π]
        let y = radius * phi.cos();
        let r = radius * phi.sin();
        let mut row = Vec::with_capacity(slices + 1);
        for j in 0..=slices {
            let theta = TAU * j as f64 / slices as f64;
            let x = r * theta.cos();
            let z = r * theta.sin();
            row.push(mesh.add_vertex(Vertex::new([x, y, z])));
        }
        v_ids.push(row);
    }

    // 生成三角形
    for i in 0..stacks {
        for j in 0..slices {
            let a = v_ids[i][j];
            let b = v_ids[i][j + 1];
            let c = v_ids[i + 1][j];
            let d = v_ids[i + 1][j + 1];
            let _ = add_triangle(&mut mesh, a, b, c);
            let _ = add_triangle(&mut mesh, c, b, d);
        }
    }

    mesh
}

// ============================================================
// 圆柱体
// ============================================================

/// 生成圆柱体，沿 Y 轴从 `-height/2` 到 `+height/2`，半径 `radius`。
///
/// `slices`：圆周分段数（最小 3）。两端各加一个中心顶点形成顶面/底面。
pub fn build_cylinder(radius: f64, height: f64, slices: usize) -> MeshStorage {
    let slices = slices.max(3);
    let h = height / 2.0;

    let mut mesh = MeshStorage::new();
    mesh.reserve(slices * 2 + 2, slices * 12, slices * 4);

    // 顶面中心、底面中心
    let top_c = mesh.add_vertex(Vertex::new([0.0, h, 0.0]));
    let bot_c = mesh.add_vertex(Vertex::new([0.0, -h, 0.0]));

    // 环上顶点
    let mut top_ring = Vec::with_capacity(slices);
    let mut bot_ring = Vec::with_capacity(slices);
    for i in 0..slices {
        let theta = TAU * i as f64 / slices as f64;
        let x = radius * theta.cos();
        let z = radius * theta.sin();
        top_ring.push(mesh.add_vertex(Vertex::new([x, h, z])));
        bot_ring.push(mesh.add_vertex(Vertex::new([x, -h, z])));
    }

    // 侧面 + 顶底面
    for i in 0..slices {
        let j = (i + 1) % slices;
        // 侧面（两个三角形）
        let _ = add_triangle(&mut mesh, top_ring[i], bot_ring[i], bot_ring[j]);
        let _ = add_triangle(&mut mesh, top_ring[i], bot_ring[j], top_ring[j]);
        // 顶面
        let _ = add_triangle(&mut mesh, top_c, top_ring[j], top_ring[i]);
        // 底面
        let _ = add_triangle(&mut mesh, bot_c, bot_ring[i], bot_ring[j]);
    }

    mesh
}

// ============================================================
// 圆环体
// ============================================================

/// 生成圆环体（torus），中心在原点，位于 XZ 平面。
///
/// `major_radius`：环半径（中心到管中心的距离）；
/// `minor_radius`：管半径；
/// `major_segments`：环分段数（最小 3）；
/// `minor_segments`：管截面分段数（最小 3）。
pub fn build_torus(
    major_radius: f64,
    minor_radius: f64,
    major_segments: usize,
    minor_segments: usize,
) -> MeshStorage {
    let mj = major_segments.max(3);
    let mn = minor_segments.max(3);

    let mut mesh = MeshStorage::new();
    mesh.reserve(mj * mn, mj * mn * 6, mj * mn * 2);

    // 生成环上顶点
    let mut rings: Vec<Vec<VertexId>> = Vec::with_capacity(mj);
    for i in 0..mj {
        let theta = TAU * i as f64 / mj as f64;
        let _cx = major_radius * theta.cos();
        let _cz = major_radius * theta.sin();
        let mut ring = Vec::with_capacity(mn);
        for j in 0..mn {
            let phi = TAU * j as f64 / mn as f64;
            let r = major_radius + minor_radius * phi.cos();
            let x = r * theta.cos();
            let z = r * theta.sin();
            let y = minor_radius * phi.sin();
            ring.push(mesh.add_vertex(Vertex::new([x, y, z])));
        }
        rings.push(ring);
    }

    for i in 0..mj {
        let ni = (i + 1) % mj;
        for j in 0..mn {
            let nj = (j + 1) % mn;
            let a = rings[i][j];
            let b = rings[ni][j];
            let c = rings[ni][nj];
            let d = rings[i][nj];
            let _ = add_triangle(&mut mesh, a, b, c);
            let _ = add_triangle(&mut mesh, a, c, d);
        }
    }

    mesh
}

// ============================================================
// 圆锥体
// ============================================================

/// 生成圆锥体，顶点在 `(0, height/2, 0)`，底面在 `y = -height/2`。
///
/// `slices`：底面圆周分段数（最小 3）。
pub fn build_cone(radius: f64, height: f64, slices: usize) -> MeshStorage {
    let slices = slices.max(3);
    let h = height / 2.0;

    let mut mesh = MeshStorage::new();
    mesh.reserve(slices + 2, slices * 6, slices * 2);

    let apex = mesh.add_vertex(Vertex::new([0.0, h, 0.0]));
    let base_c = mesh.add_vertex(Vertex::new([0.0, -h, 0.0]));

    let mut ring = Vec::with_capacity(slices);
    for i in 0..slices {
        let theta = TAU * i as f64 / slices as f64;
        ring.push(mesh.add_vertex(Vertex::new([
            radius * theta.cos(),
            -h,
            radius * theta.sin(),
        ])));
    }

    for i in 0..slices {
        let j = (i + 1) % slices;
        // 侧面
        let _ = add_triangle(&mut mesh, apex, ring[j], ring[i]);
        // 底面
        let _ = add_triangle(&mut mesh, base_c, ring[i], ring[j]);
    }

    mesh
}

// ============================================================
// 网格平面
// ============================================================

/// 生成立方体 XZ 平面网格（Y=0），中心在原点。
///
/// `width`×`depth` 的矩形区域，细分为 `segments_x`×`segments_z` 个单元，
/// 每个单元由两个三角形组成。
pub fn build_grid(width: f64, depth: f64, segments_x: usize, segments_z: usize) -> MeshStorage {
    let sx = segments_x.max(1);
    let sz = segments_z.max(1);

    let mut mesh = MeshStorage::new();
    let nv = (sx + 1) * (sz + 1);
    mesh.reserve(nv, sx * sz * 6, sx * sz * 2);

    let mut v_ids: Vec<Vec<VertexId>> = Vec::with_capacity(sz + 1);
    for iz in 0..=sz {
        let z = -depth / 2.0 + depth * iz as f64 / sz as f64;
        let mut row = Vec::with_capacity(sx + 1);
        for ix in 0..=sx {
            let x = -width / 2.0 + width * ix as f64 / sx as f64;
            row.push(mesh.add_vertex(Vertex::new([x, 0.0, z])));
        }
        v_ids.push(row);
    }

    for iz in 0..sz {
        for ix in 0..sx {
            let a = v_ids[iz][ix];
            let b = v_ids[iz][ix + 1];
            let c = v_ids[iz + 1][ix];
            let d = v_ids[iz + 1][ix + 1];
            let _ = add_triangle(&mut mesh, a, b, c);
            let _ = add_triangle(&mut mesh, c, b, d);
        }
    }

    mesh
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_is_closed() {
        let cube = build_cube(2.0);
        assert_eq!(cube.vertex_count(), 8);
        assert_eq!(cube.face_count(), 12);
        crate::topology_ops::validate_mesh(&cube).unwrap();
    }

    #[test]
    fn uv_sphere_is_closed() {
        let sphere = build_uv_sphere(1.0, 16, 8);
        assert!(sphere.face_count() > 0);
        crate::topology_ops::validate_mesh(&sphere).unwrap();
    }

    #[test]
    fn cylinder_counts() {
        let cyl = build_cylinder(1.0, 2.0, 16);
        assert_eq!(cyl.vertex_count(), 34); // 16*2 + 2 centers
        assert_eq!(cyl.face_count(), 64); // 16*4
        crate::topology_ops::validate_mesh(&cyl).unwrap();
    }

    #[test]
    fn torus_counts() {
        let torus = build_torus(2.0, 0.5, 16, 8);
        assert_eq!(torus.vertex_count(), 128);
        assert_eq!(torus.face_count(), 256);
        crate::topology_ops::validate_mesh(&torus).unwrap();
    }

    #[test]
    fn cone_counts() {
        let cone = build_cone(1.0, 2.0, 16);
        assert_eq!(cone.vertex_count(), 18);
        assert_eq!(cone.face_count(), 32);
        crate::topology_ops::validate_mesh(&cone).unwrap();
    }

    #[test]
    fn grid_counts() {
        let grid = build_grid(2.0, 2.0, 4, 4);
        assert_eq!(grid.vertex_count(), 25); // 5*5
        assert_eq!(grid.face_count(), 32); // 4*4*2
    }

    #[test]
    fn primitives_pass_validation() {
        for mesh in [
            build_cube(1.0),
            build_uv_sphere(1.0, 8, 4),
            build_cylinder(0.5, 1.0, 8),
            build_torus(2.0, 0.3, 8, 4),
            build_cone(1.0, 1.0, 8),
            build_grid(1.0, 1.0, 2, 2),
        ] {
            crate::topology_ops::validate_mesh(&mesh).unwrap();
        }
    }
}
