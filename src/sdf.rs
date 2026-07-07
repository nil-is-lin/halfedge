//! SDF（有符号距离函数）与 Marching Cubes 模块。
//!
//! 提供：
//! - [`Sdf`] trait：有符号距离函数接口
//! - SDF 图元：球体、立方体、胶囊体、圆环体
//! - CSG 操作：并集、交集、差集、平滑变体
//! - [`march_cubes`]：Marching Cubes 等值面提取
//! - [`march_sdf`]：从 SDF 生成三角网格
//!
//! ## Marching Cubes
//!
//! Lorensen & Cline (1987) 的经典算法，在三维标量场中提取等值面。
//! 对每个体素，根据 8 个角点值生成 8-bit 案例码，查表得到三角面拓扑，
//! 线性插值求边上交点。
//!
//! ## SDF 图元公式
//!
//! | 图元 | SDF |
//! |------|-----|
//! | 球体 | $\|\mathbf{x}-\mathbf{c}\|-r$ |
//! | 立方体 | $\|\max(\mathbf{q},\mathbf{0})\|+\min(\max(q_x,\max(q_y,q_z)),0)$ |
//! | 胶囊体 | $\|\text{closest\_on\_segment}-\mathbf{x}\|-r$ |
//! | 圆环体 | $\|(\sqrt{x^2+z^2}-R, y)\|-r$ |

use std::collections::HashMap;

use crate::ids::VertexId;
use crate::storage::{MeshStorage, Vertex};
use crate::topology_ops::add_triangle;

// ============================================================
// Sdf trait
// ============================================================

/// 有符号距离函数 trait。
///
/// 实现 `eval` 方法返回点 `p` 处的 SDF 值：
/// - 负值 = 物体内部
/// - 零值 = 表面上
/// - 正值 = 物体外部
pub trait Sdf {
    /// 计算 (x,y,z) 处的 SDF 值。
    fn eval(&self, p: [f64; 3]) -> f64;

    /// 用中心差分估算梯度。
    fn gradient(&self, p: [f64; 3]) -> [f64; 3] {
        let eps = 1e-6;
        let dx = (self.eval([p[0] + eps, p[1], p[2]]) - self.eval([p[0] - eps, p[1], p[2]]))
            / (2.0 * eps);
        let dy = (self.eval([p[0], p[1] + eps, p[2]]) - self.eval([p[0], p[1] - eps, p[2]]))
            / (2.0 * eps);
        let dz = (self.eval([p[0], p[1], p[2] + eps]) - self.eval([p[0], p[1], p[2] - eps]))
            / (2.0 * eps);
        [dx, dy, dz]
    }
}

// ============================================================
// SDF 图元
// ============================================================

/// 球体 SDF：$f(\mathbf{x}) = \|\mathbf{x}-\mathbf{c}\|-r$。
#[derive(Debug, Clone)]
pub struct SdfSphere {
    pub center: [f64; 3],
    pub radius: f64,
}

impl Sdf for SdfSphere {
    fn eval(&self, p: [f64; 3]) -> f64 {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt() - self.radius
    }
}

/// 轴对齐立方体 SDF（中心在原点，半尺寸 half）。
///
/// $f(\mathbf{x}) = \|\max(\mathbf{q},\mathbf{0})\| + \min(\max(q_x,\max(q_y,q_z)),0)$，
/// 其中 $\mathbf{q} = |\mathbf{x}| - \mathbf{b}$。
#[derive(Debug, Clone)]
pub struct SdfBox {
    pub half: [f64; 3],
}

impl Sdf for SdfBox {
    fn eval(&self, p: [f64; 3]) -> f64 {
        let qx = p[0].abs() - self.half[0];
        let qy = p[1].abs() - self.half[1];
        let qz = p[2].abs() - self.half[2];
        let outside = [qx.max(0.0), qy.max(0.0), qz.max(0.0)];
        let outside_len =
            (outside[0] * outside[0] + outside[1] * outside[1] + outside[2] * outside[2]).sqrt();
        let inside = qx.max(qy).max(qz).min(0.0);
        outside_len + inside
    }
}

/// 胶囊体 SDF（线段 ab，半径 r）。
#[derive(Debug, Clone)]
pub struct SdfCapsule {
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub radius: f64,
}

impl Sdf for SdfCapsule {
    fn eval(&self, p: [f64; 3]) -> f64 {
        let pa = [p[0] - self.a[0], p[1] - self.a[1], p[2] - self.a[2]];
        let ba = [
            self.b[0] - self.a[0],
            self.b[1] - self.a[1],
            self.b[2] - self.a[2],
        ];
        let baba = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
        if baba < 1e-14 {
            return (pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2]).sqrt() - self.radius;
        }
        let t = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) / baba).clamp(0.0, 1.0);
        let cx = pa[0] - t * ba[0];
        let cy = pa[1] - t * ba[1];
        let cz = pa[2] - t * ba[2];
        (cx * cx + cy * cy + cz * cz).sqrt() - self.radius
    }
}

/// 圆环体 SDF（主半径 R，管半径 r，在 xz 平面上）。
///
/// $f(\mathbf{x}) = \|(\sqrt{x^2+z^2}-R,\; y)\|-r$。
#[derive(Debug, Clone)]
pub struct SdfTorus {
    pub major_radius: f64,
    pub minor_radius: f64,
}

impl Sdf for SdfTorus {
    fn eval(&self, p: [f64; 3]) -> f64 {
        let xz = (p[0] * p[0] + p[2] * p[2]).sqrt();
        let qx = xz - self.major_radius;
        let qy = p[1];
        (qx * qx + qy * qy).sqrt() - self.minor_radius
    }
}

// ============================================================
// CSG 组合操作
// ============================================================

/// SDF 并集：$\min(f_A, f_B)$。
#[derive(Debug, Clone)]
pub struct SdfUnion<A: Sdf, B: Sdf> {
    pub a: A,
    pub b: B,
}

impl<A: Sdf, B: Sdf> Sdf for SdfUnion<A, B> {
    fn eval(&self, p: [f64; 3]) -> f64 {
        self.a.eval(p).min(self.b.eval(p))
    }
}

/// SDF 交集：$\max(f_A, f_B)$。
#[derive(Debug, Clone)]
pub struct SdfIntersection<A: Sdf, B: Sdf> {
    pub a: A,
    pub b: B,
}

impl<A: Sdf, B: Sdf> Sdf for SdfIntersection<A, B> {
    fn eval(&self, p: [f64; 3]) -> f64 {
        self.a.eval(p).max(self.b.eval(p))
    }
}

/// SDF 差集：$\max(f_A, -f_B)$。
#[derive(Debug, Clone)]
pub struct SdfDifference<A: Sdf, B: Sdf> {
    pub a: A,
    pub b: B,
}

impl<A: Sdf, B: Sdf> Sdf for SdfDifference<A, B> {
    fn eval(&self, p: [f64; 3]) -> f64 {
        self.a.eval(p).max(-self.b.eval(p))
    }
}

/// SDF 平滑并集（Inigo Quilez）。
///
/// $h = \text{clamp}(\frac{1}{2}+\frac{f_A-f_B}{2k}, 0, 1)$，
/// $f = \text{mix}(f_A, f_B, h) - k \cdot h(1-h)$。
#[derive(Debug, Clone)]
pub struct SdfSmoothUnion<A: Sdf, B: Sdf> {
    pub a: A,
    pub b: B,
    pub k: f64,
}

impl<A: Sdf, B: Sdf> Sdf for SdfSmoothUnion<A, B> {
    fn eval(&self, p: [f64; 3]) -> f64 {
        let fa = self.a.eval(p);
        let fb = self.b.eval(p);
        let h = (0.5 + 0.5 * (fa - fb) / self.k).clamp(0.0, 1.0);
        fa * (1.0 - h) + fb * h - self.k * h * (1.0 - h)
    }
}

/// SDF 平移。
#[derive(Debug, Clone)]
pub struct SdfTranslate<S: Sdf> {
    pub sdf: S,
    pub offset: [f64; 3],
}

impl<S: Sdf> Sdf for SdfTranslate<S> {
    fn eval(&self, p: [f64; 3]) -> f64 {
        self.sdf.eval([
            p[0] - self.offset[0],
            p[1] - self.offset[1],
            p[2] - self.offset[2],
        ])
    }
}

// ============================================================
// Marching Cubes 查找表
// ============================================================

/// 边表：EDGE_TABLE[i] 的 12-bit 指示案例 i 中哪些边有交点。
const EDGE_TABLE: [u16; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6,
    0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69,
    0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c,
    0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
    0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3,
    0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a,
    0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
    0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];

/// 三角表：TRI_TABLE[i] 列出案例 i 的三角面顶点（边编号），-1 终止。
/// 每行最多 5 个三角形（15 个边编号 + 1 个 -1）。
const TRI_TABLE: [[i16; 16]; 256] = include!("marching_table.in");

// ============================================================
// Marching Cubes 核心
// ============================================================

/// Marching Cubes 参数。
#[derive(Debug, Clone)]
pub struct McParams {
    /// 采样包围盒起点（最小角）
    pub origin: [f64; 3],
    /// 单个体素的尺寸
    pub cell_size: [f64; 3],
    /// xyz 方向体素数
    pub resolution: [usize; 3],
    /// 等值面阈值（SDF 下通常为 0.0）
    pub isolevel: f64,
}

/// 从 SDF 生成三角网格。
///
/// 在均匀网格上采样 SDF 值，然后调用 Marching Cubes 提取等值面。
pub fn march_sdf(sdf: &dyn Sdf, params: &McParams) -> MeshStorage {
    let [nx, ny, nz] = params.resolution;
    let n_verts_x = nx + 1;
    let n_verts_y = ny + 1;
    let n_verts_z = nz + 1;

    // 采样 SDF 值
    let mut field = vec![0.0; n_verts_x * n_verts_y * n_verts_z];
    for iz in 0..n_verts_z {
        for iy in 0..n_verts_y {
            for ix in 0..n_verts_x {
                let x = params.origin[0] + ix as f64 * params.cell_size[0];
                let y = params.origin[1] + iy as f64 * params.cell_size[1];
                let z = params.origin[2] + iz as f64 * params.cell_size[2];
                let idx = iz * n_verts_y * n_verts_x + iy * n_verts_x + ix;
                field[idx] = sdf.eval([x, y, z]);
            }
        }
    }

    march_field(
        &field,
        n_verts_x,
        n_verts_y,
        n_verts_z,
        params.origin,
        params.cell_size,
        params.isolevel,
    )
}

/// 从三维标量场数组生成三角网格（Marching Cubes）。
///
/// # 参数
/// - `field`: 标量场，长度 (nx)*(ny)*(nz)，按 z-y-x 优先排列
/// - `nx, ny, nz`: 采样点数（体素数 + 1）
/// - `origin`: 包围盒起点
/// - `cell_size`: 体素尺寸
/// - `isolevel`: 等值面阈值
pub fn march_field(
    field: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    origin: [f64; 3],
    cell_size: [f64; 3],
    isolevel: f64,
) -> MeshStorage {
    let mut mesh = MeshStorage::new();

    if nx < 2 || ny < 2 || nz < 2 || field.len() < nx * ny * nz {
        return mesh;
    }

    // 顶点去重映射：key = 体素索引 + 边编号 → VertexId
    let mut vertex_map: HashMap<(usize, usize, usize, usize), VertexId> = HashMap::new();
    let mut failed_tris: u32 = 0;

    // 体素角点编号（Lorensen 约定）
    // 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
    // 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
    let corner_offsets: [(usize, usize, usize); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];

    // 边端点：edge_endpoints[e] = (corner_a, corner_b)
    let edge_endpoints: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (3, 2),
        (0, 3), // 底面
        (4, 5),
        (5, 6),
        (7, 6),
        (4, 7), // 顶面
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // 竖边
    ];

    // 遍历所有体素
    for iz in 0..nz - 1 {
        for iy in 0..ny - 1 {
            for ix in 0..nx - 1 {
                // 采样 8 个角点值
                let mut values = [0.0; 8];
                let mut case_index = 0u8;
                for (ci, (dx, dy, dz)) in corner_offsets.iter().enumerate() {
                    let jx = ix + dx;
                    let jy = iy + dy;
                    let jz = iz + dz;
                    let idx = jz * ny * nx + jy * nx + jx;
                    let v = if idx < field.len() {
                        field[idx]
                    } else {
                        isolevel + 1.0
                    };
                    values[ci] = v;
                    if v < isolevel {
                        case_index |= 1 << ci;
                    }
                }

                if case_index == 0 || case_index == 255 {
                    continue;
                }

                let edge_bits = EDGE_TABLE[case_index as usize];
                if edge_bits == 0 {
                    continue;
                }

                // 计算各边交点
                let mut edge_vertices: [Option<VertexId>; 12] = [None; 12];
                for e in 0..12 {
                    if edge_bits & (1 << e) == 0 {
                        continue;
                    }
                    let (ca, cb) = edge_endpoints[e];
                    // 查找或创建顶点
                    let key = (ix, iy, iz, e);
                    let vid = if let Some(&vid) = vertex_map.get(&key) {
                        vid
                    } else {
                        let va = values[ca];
                        let vb = values[cb];
                        let t = if (vb - va).abs() < 1e-14 {
                            0.5
                        } else {
                            ((isolevel - va) / (vb - va)).clamp(0.0, 1.0)
                        };
                        let (dxa, dya, dza) = corner_offsets[ca];
                        let (dxb, dyb, dzb) = corner_offsets[cb];
                        let x = origin[0]
                            + ((ix + dxa) as f64 * (1.0 - t) + (ix + dxb) as f64 * t)
                                * cell_size[0];
                        let y = origin[1]
                            + ((iy + dya) as f64 * (1.0 - t) + (iy + dyb) as f64 * t)
                                * cell_size[1];
                        let z = origin[2]
                            + ((iz + dza) as f64 * (1.0 - t) + (iz + dzb) as f64 * t)
                                * cell_size[2];
                        let vid = mesh.add_vertex(Vertex::new([x, y, z]));
                        vertex_map.insert(key, vid);
                        vid
                    };
                    edge_vertices[e] = Some(vid);
                }

                // 生成三角面
                let tri_row = &TRI_TABLE[case_index as usize];
                let mut ti = 0;
                while ti + 2 < 16 {
                    // -1（i16）作为终止标记
                    if tri_row[ti] < 0 || tri_row[ti + 1] < 0 || tri_row[ti + 2] < 0 {
                        break;
                    }
                    let e0 = tri_row[ti] as usize;
                    let e1 = tri_row[ti + 1] as usize;
                    let e2 = tri_row[ti + 2] as usize;
                    let v0 = match edge_vertices[e0] {
                        Some(v) => v,
                        None => {
                            ti += 3;
                            continue;
                        }
                    };
                    let v1 = match edge_vertices[e1] {
                        Some(v) => v,
                        None => {
                            ti += 3;
                            continue;
                        }
                    };
                    let v2 = match edge_vertices[e2] {
                        Some(v) => v,
                        None => {
                            ti += 3;
                            continue;
                        }
                    };
                    if add_triangle(&mut mesh, v0, v1, v2).is_err()
                        && add_triangle(&mut mesh, v2, v1, v0).is_err()
                    {
                        failed_tris += 1;
                    }
                    ti += 3;
                }
            }
        }
    }

    if failed_tris > 0 {
        log::warn!(
            "[halfedge::march_field] 警告：{failed_tris} 个三角形创建失败（拓扑冲突），已跳过"
        );
    }

    mesh
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 球体 SDF：中心处值为 -radius。
    #[test]
    fn sdf_sphere_center() {
        let sphere = SdfSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        assert!((sphere.eval([0.0, 0.0, 0.0]) - (-1.0)).abs() < 1e-10);
        assert!((sphere.eval([2.0, 0.0, 0.0]) - 1.0).abs() < 1e-10);
    }

    /// Box SDF：中心处值为负，外部为正。
    #[test]
    fn sdf_box() {
        let bx = SdfBox {
            half: [1.0, 1.0, 1.0],
        };
        assert!(bx.eval([0.0, 0.0, 0.0]) < 0.0);
        assert!(bx.eval([2.0, 0.0, 0.0]) > 0.0);
    }

    /// 圆环体 SDF。
    #[test]
    fn sdf_torus() {
        let torus = SdfTorus {
            major_radius: 1.0,
            minor_radius: 0.3,
        };
        // 在环中心（x=1,y=0,z=0）处应接近 -0.3
        assert!((torus.eval([1.0, 0.0, 0.0]) - (-0.3)).abs() < 1e-10);
    }

    /// CSG 并集。
    #[test]
    fn sdf_union() {
        let a = SdfSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let b = SdfSphere {
            center: [2.0, 0.0, 0.0],
            radius: 1.0,
        };
        let union = SdfUnion { a, b };
        // 两个球之间 (1,0,0) 处应为负（两个球在此处的值都接近 0）
        assert!(union.eval([1.0, 0.0, 0.0]) <= 0.0);
    }

    /// Marching Cubes 生成球体网格。
    #[test]
    fn march_cubes_sphere() {
        let sphere = SdfSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let params = McParams {
            origin: [-2.0, -2.0, -2.0],
            cell_size: [0.25, 0.25, 0.25],
            resolution: [16, 16, 16],
            isolevel: 0.0,
        };
        let mesh = march_sdf(&sphere, &params);
        assert!(mesh.vertex_count() > 10, "球体网格应有顶点");
        assert!(mesh.face_count() > 5, "球体网格应有面");
    }

    /// Marching Cubes 生成圆环体网格。
    #[test]
    fn march_cubes_torus() {
        let torus = SdfTorus {
            major_radius: 1.0,
            minor_radius: 0.3,
        };
        let params = McParams {
            origin: [-2.0, -2.0, -2.0],
            cell_size: [0.2, 0.2, 0.2],
            resolution: [20, 20, 20],
            isolevel: 0.0,
        };
        let mesh = march_sdf(&torus, &params);
        assert!(mesh.vertex_count() > 10);
        assert!(mesh.face_count() > 5);
    }

    /// Marching Cubes 空场景（全外部）。
    #[test]
    fn march_cubes_empty() {
        let sphere = SdfSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let params = McParams {
            origin: [10.0, 10.0, 10.0],
            cell_size: [1.0, 1.0, 1.0],
            resolution: [4, 4, 4],
            isolevel: 0.0,
        };
        let mesh = march_sdf(&sphere, &params);
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    /// SDF 梯度估算。
    #[test]
    fn sdf_gradient() {
        let sphere = SdfSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let grad = sphere.gradient([2.0, 0.0, 0.0]);
        // 梯度应指向 x 正方向（从表面向外）
        assert!(grad[0] > 0.9, "梯度 x 分量应接近 1: {}", grad[0]);
    }

    /// 平滑并集 SDF。
    #[test]
    fn sdf_smooth_union() {
        let a = SdfSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let b = SdfSphere {
            center: [1.5, 0.0, 0.0],
            radius: 1.0,
        };
        let smooth = SdfSmoothUnion { a, b, k: 0.5 };
        // 平滑并集在中间点应为负值
        assert!(smooth.eval([0.75, 0.0, 0.0]) < 0.0);
    }
}
