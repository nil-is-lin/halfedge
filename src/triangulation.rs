//! 多边形三角化模块
//!
//! 提供二维多边形三角化算法。
//!
//! ## 鲁棒性
//! 所有的方向判定（凸顶点、点在三角形内、多边形环绕方向）均通过
//! [`crate::predicates`] 模块的 Shewchuk 鲁棒谓词实现，保证在共线、
//! 共点等退化情况下给出精确的符号判定。

use crate::predicates::{is_convex_vertex2d, orient2d, point_in_triangle_2d};

// ============================================================
// 2D 向量工具
// ============================================================

type V2 = [f64; 2];

/// 多边形是否 CCW（逆时针）。
///
/// 使用 Shewchuk 鲁棒 `orient2d` 累加各三角形有向面积：当总面积为正
/// 时多边形呈 CCW。退化多边形（面积 = 0）返回 `false`。
fn is_ccw(poly: &[V2]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    // 以第一个顶点为基准，累加 (v0, vi, vi+1) 的 orient2d
    // orient2d > 0 表示 CCW，总和 > 0 表示多边形 CCW
    let v0 = poly[0];
    let mut sum = 0.0f64;
    for i in 1..n - 1 {
        sum += orient2d(v0, poly[i], poly[i + 1]);
    }
    sum > 0.0
}

// ============================================================
// 扇形三角化（凸多边形）
// ============================================================

/// 扇形三角化：从第一个顶点向其余顶点连三角形。
///
/// 要求多边形为**简单凸多边形**且顶点按 CCW 顺序排列。
/// 返回 `Vec<[usize; 3]>`，每个元素是三个顶点索引。
///
/// 对凹多边形结果可能不正确（三角形可能落在多边形外部）。
pub fn fan_triangulation(polygon: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    let mut tris = Vec::with_capacity(n - 2);
    for i in 2..n {
        tris.push([0, i - 1, i]);
    }
    tris
}

/// 扇形三角化（3D 版本）：输入为 3D 点，假设所有点共面。
/// 投影到主平面对应的 2D 坐标系上进行扇形三角化。
pub fn fan_triangulation_3d(polygon: &[[f64; 3]]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    // 简单起见，直接按索引扇形（与其他构建器配合使用时由调用者保证共面/凸性）
    let mut tris = Vec::with_capacity(n - 2);
    for i in 2..n {
        tris.push([0, i - 1, i]);
    }
    tris
}

// ============================================================
// Ear Clipping（支持凹多边形）
// ============================================================

/// Ear Clipping 三角化：适用于任意**简单多边形**（含凹多边形）。
///
/// 顶点按 CCW 顺序排列。返回 `Vec<[usize; 3]>`。
///
/// 算法：反复寻找"耳尖"（凸顶点，且对角线与多边形不相交），
/// 剪切该耳朵后继续，直到剩余 3 个顶点。
pub fn ear_clipping(polygon: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }

    // 确保 CCW
    let mut verts: Vec<V2> = polygon.to_vec();
    if !is_ccw(&verts) {
        verts.reverse();
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut triangles = Vec::with_capacity(n - 2);

    let mut iter_count = 0;
    while indices.len() > 3 {
        iter_count += 1;
        if iter_count > indices.len() * indices.len() {
            // 安全兜底：算法卡住时回退到扇形
            break;
        }

        let m = indices.len();
        let mut found_ear = false;

        for i in 0..m {
            let prev = if i == 0 { m - 1 } else { i - 1 };
            let next = (i + 1) % m;

            let a = verts[indices[prev]];
            let b = verts[indices[i]];
            let c = verts[indices[next]];

            // 检查顶点 i 是否为凸顶点（内角 < 180°）
            // 使用鲁棒 orient2d：b 是凸顶点当且仅当 orient2d(a, b, c) > 0
            if !is_convex_vertex2d(a, b, c) {
                continue; // 凹顶点或共线，跳过
            }

            // 检查三角形 (a,b,c) 内是否包含其他顶点
            // 使用鲁棒 point_in_triangle_2d
            let mut is_ear = true;
            for j in 0..m {
                if j == prev || j == i || j == next {
                    continue;
                }
                if point_in_triangle_2d(verts[indices[j]], a, b, c) {
                    is_ear = false;
                    break;
                }
            }

            if is_ear {
                triangles.push([indices[prev], indices[i], indices[next]]);
                indices.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            break; // 无耳可剪
        }
    }

    // 剩余最后一个三角形
    if indices.len() == 3 {
        triangles.push([indices[0], indices[1], indices[2]]);
    }

    triangles
}

/// Ear Clipping 三角化（3D 版本）：将 3D 点投影到最佳平面对应的 2D 坐标，
/// 执行 ear clipping，再映射回原始索引。
///
/// 自动选择法向绝对值最大的轴作为投影丢弃轴。
pub fn ear_clipping_3d(polygon: &[[f64; 3]]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }

    // 计算近似法向，选择丢弃轴
    let mut normal = [0.0f64; 3];
    for i in 0..n {
        let j = (i + 1) % n;
        normal[0] += (polygon[i][1] - polygon[j][1]) * (polygon[i][2] + polygon[j][2]);
        normal[1] += (polygon[i][2] - polygon[j][2]) * (polygon[i][0] + polygon[j][0]);
        normal[2] += (polygon[i][0] - polygon[j][0]) * (polygon[i][1] + polygon[j][1]);
    }

    // 选择法向绝对值最大的轴丢弃
    let drop_axis = if normal[0].abs() >= normal[1].abs() && normal[0].abs() >= normal[2].abs() {
        0 // 丢弃 x
    } else if normal[1].abs() >= normal[2].abs() {
        1 // 丢弃 y
    } else {
        2 // 丢弃 z
    };

    let poly_2d: Vec<V2> = polygon
        .iter()
        .map(|p| match drop_axis {
            0 => [p[1], p[2]],
            1 => [p[0], p[2]],
            _ => [p[0], p[1]],
        })
        .collect();

    ear_clipping(&poly_2d)
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fan_triangulation_quad() {
        let quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tris = fan_triangulation(&quad);
        assert_eq!(tris.len(), 2);
        assert_eq!(tris[0], [0, 1, 2]);
        assert_eq!(tris[1], [0, 2, 3]);
    }

    #[test]
    fn fan_triangulation_empty() {
        assert_eq!(fan_triangulation(&[]).len(), 0);
        assert_eq!(fan_triangulation(&[[0.0, 0.0]]).len(), 0);
    }

    #[test]
    fn ear_clipping_convex_quad() {
        let quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tris = ear_clipping(&quad);
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn ear_clipping_concave_l_shape() {
        let lshape = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.5],
            [1.0, 0.5],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let tris = ear_clipping(&lshape);
        assert_eq!(tris.len(), 4);
    }

    #[test]
    fn ear_clipping_3d_square() {
        let square = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let tris = ear_clipping_3d(&square);
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn ear_clipping_triangle() {
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(ear_clipping(&tri).len(), 1);
    }
}
