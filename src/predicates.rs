//! 鲁棒几何谓词（Shewchuk 自适应精度浮点运算）。
//!
//! 本模块基于 [`robust`](https://docs.rs/robust) crate（Jonathan Richard Shewchuk
//! 1997 论文的 Rust 直译实现），提供**精确的**几何方向谓词：
//!
//! - [`orient2d`]：2D 三点方向判定（CCW / CW / 共线）
//! - [`orient3d`]：3D 四点方向判定（在平面下方 / 上方 / 共面）
//! - [`incircle`]：2D 四点共圆判定（在圆内 / 圆外 / 共圆）
//! - [`insphere`]：3D 五点共球判定（在球内 / 球外 / 共球）
//!
//! ## 为什么需要鲁棒谓词
//!
//! 普通浮点运算在**退化情况**（共线、共面、共圆、共球）附近会因舍入误差
//! 给出错误符号，导致布尔运算、三角剖分、内外判定等算法在边界情况下
//! 失败（例如点刚好在三角形边上时被误判）。
//!
//! Shewchuk 的自适应精度算法在普通 `f64` 计算不足以确定符号时，
//! 自动扩展到扩展精度（adaptive precision），**保证返回值的符号是精确的**。
//!
//! ## 性能
//!
//! 自适应算法仅在普通计算结果接近 0 时才启用扩展精度，
//! 非退化情况下性能与朴素 `f64` 实现相当。
//!
//! ## API 约定
//!
//! 谓词返回 `f64` 而非 `bool`：
//! - 正值 → 第一类几何关系成立（CCW / 圆内 / 球内 / 平面下方）
//! - 负值 → 反向关系成立（CW / 圆外 / 球外 / 平面上方）
//! - 零  → 退化（共线 / 共圆 / 共球 / 共面）
//!
//! 便利函数 [`is_ccw2d`] / [`is_collinear2d`] / [`point_in_triangle_2d`]
//! / [`tet_signed_volume`] 提供常用 `bool` / 标量包装。
//!
//! ## 参考
//! - Shewchuk, J. R. (1997). *Adaptive Precision Floating-Point Arithmetic
//!   and Fast Robust Geometric Predicates.* Discrete & Computational Geometry.

use robust::{Coord, Coord3D};

// ============================================================
// 核心谓词
// ============================================================

/// 2D 方向谓词：返回点 `c` 相对于有向直线 `a → b` 的方向。
///
/// - 正值：`c` 在 `a → b` 的**左侧**（三点呈 CCW 顺序）
/// - 负值：`c` 在 `a → b` 的**右侧**（三点呈 CW 顺序）
/// - 零：三点**共线**
///
/// 返回值的绝对值等于三角形 `abc` 有向面积的两倍。
#[inline]
pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    robust::orient2d(
        Coord { x: a[0], y: a[1] },
        Coord { x: b[0], y: b[1] },
        Coord { x: c[0], y: c[1] },
    )
}

/// 3D 方向谓词：返回点 `d` 相对于平面 `abc` 的方向。
///
/// **符号约定**（Shewchuk 原版）：从 `abc` 呈 CCW 顺序的一侧（"上方"）看，
/// - 正值：`d` 在平面 `abc` 的**下方**
/// - 负值：`d` 在平面 `abc` 的**上方**
/// - 零：四点**共面**
///
/// 注意：此约定与"abc CCW 朝外时四面体体积为正"的几何约定**相反**。
/// 如需后者，请使用 [`tet_signed_volume`]。
///
/// 返回值的绝对值等于四面体 `abcd` 有向体积的 6 倍。
#[inline]
pub fn orient3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    robust::orient3d(
        Coord3D {
            x: a[0],
            y: a[1],
            z: a[2],
        },
        Coord3D {
            x: b[0],
            y: b[1],
            z: b[2],
        },
        Coord3D {
            x: c[0],
            y: c[1],
            z: c[2],
        },
        Coord3D {
            x: d[0],
            y: d[1],
            z: d[2],
        },
    )
}

/// 2D 共圆谓词：返回点 `d` 相对于 `a, b, c` 外接圆的位置。
///
/// **要求**：`a, b, c` 必须呈 **CCW 顺序**，否则结果符号反转。
///
/// - 正值：`d` 在外接圆**内部**
/// - 负值：`d` 在外接圆**外部**
/// - 零：四点**共圆**
#[inline]
pub fn incircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    robust::incircle(
        Coord { x: a[0], y: a[1] },
        Coord { x: b[0], y: b[1] },
        Coord { x: c[0], y: c[1] },
        Coord { x: d[0], y: d[1] },
    )
}

/// 3D 共球谓词：返回点 `e` 相对于 `a, b, c, d` 外接球的位置。
///
/// **要求**：`a, b, c, d` 必须呈**正方向**（即 `orient3d(a, b, c, d) > 0`），
/// 否则结果符号反转。
///
/// - 正值：`e` 在外接球**内部**
/// - 负值：`e` 在外接球**外部**
/// - 零：五点**共球**
#[inline]
pub fn insphere(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], e: [f64; 3]) -> f64 {
    robust::insphere(
        Coord3D {
            x: a[0],
            y: a[1],
            z: a[2],
        },
        Coord3D {
            x: b[0],
            y: b[1],
            z: b[2],
        },
        Coord3D {
            x: c[0],
            y: c[1],
            z: c[2],
        },
        Coord3D {
            x: d[0],
            y: d[1],
            z: d[2],
        },
        Coord3D {
            x: e[0],
            y: e[1],
            z: e[2],
        },
    )
}

// ============================================================
// 便利包装
// ============================================================

/// 三角形 `abc` 的 2D 有符号面积（CCW 为正）。
///
/// 等于 `0.5 * orient2d(a, b, c)`。
#[inline]
pub fn signed_area2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    0.5 * orient2d(a, b, c)
}

/// 三角形 `abc` 的 2D 无符号面积。
#[inline]
pub fn triangle_area_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    signed_area2d(a, b, c).abs()
}

/// 三点是否呈 CCW 顺序（严格左转）。
#[inline]
pub fn is_ccw2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    orient2d(a, b, c) > 0.0
}

/// 三点是否共线。
#[inline]
pub fn is_collinear2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    orient2d(a, b, c) == 0.0
}

/// 顶点 `b` 在 `a → c` 路径上是否为凸顶点（左转，即 CCW 三角形）。
///
/// 等价于 `orient2d(a, b, c) > 0`。用于 ear clipping 凸性判定。
#[inline]
pub fn is_convex_vertex2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    orient2d(a, b, c) > 0.0
}

/// 点 `p` 是否在三角形 `abc` 内部或边界上（2D）。
///
/// 使用 4 次 `orient2d` 同号判定：若 `abc` 为 CCW，则 `p` 在内部当且仅当
/// `p` 相对三条有向边 `ab`、`bc`、`ca` 均在左侧或边上（即 orient ≥ 0）。
/// 若 `abc` 为 CW，则要求 orient ≤ 0。
///
/// 退化三角形（共线）返回 `false`。
pub fn point_in_triangle_2d(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let o1 = orient2d(a, b, p);
    let o2 = orient2d(b, c, p);
    let o3 = orient2d(c, a, p);
    // 三角形 abc 退化（共线）
    let tri_sign = orient2d(a, b, c);
    if tri_sign == 0.0 {
        return false;
    }
    if tri_sign > 0.0 {
        // CCW：要求 p 在所有边的左侧或边上
        o1 >= 0.0 && o2 >= 0.0 && o3 >= 0.0
    } else {
        // CW：要求 p 在所有边的右侧或边上
        o1 <= 0.0 && o2 <= 0.0 && o3 <= 0.0
    }
}

/// 四面体 `abcd` 的有符号体积（CCW 朝外为正）。
///
/// 等于 `-orient3d(a, b, c, d) / 6`（符号取反是因为 Shewchuk 的 `orient3d`
/// 约定 "d 在 abc 下方为正"，与 "abc CCW 朝外时体积为正" 的几何约定相反）。
#[inline]
pub fn tet_signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    -orient3d(a, b, c, d) / 6.0
}

/// 四点是否共面。
#[inline]
pub fn is_coplanar(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> bool {
    orient3d(a, b, c, d) == 0.0
}

/// 3D 三角形是否退化（三顶点共线或重合）。
///
/// 算法：将三角形投影到**主轴正交**的 2D 平面（选取叉积分量最大的轴为丢弃轴，
/// 保证投影面积最大，避免再次退化），然后用 [`orient2d`] 精确判定共线性。
///
/// 与朴素浮点叉积长度阈值相比，此实现使用 Shewchuk 自适应精度，
/// 在退化边界情况下也能给出**精确**的共线判定。
pub fn is_triangle_degenerate_3d(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    // 法向各分量绝对值
    let nx = (ab[1] * ac[2] - ab[2] * ac[1]).abs();
    let ny = (ab[2] * ac[0] - ab[0] * ac[2]).abs();
    let nz = (ab[0] * ac[1] - ab[1] * ac[0]).abs();
    // 丢弃最大分量轴，投影到其他两轴
    if nx >= ny && nx >= nz {
        // 丢弃 x，投影到 yz
        orient2d([a[1], a[2]], [b[1], b[2]], [c[1], c[2]]) == 0.0
    } else if ny >= nz {
        // 丢弃 y，投影到 xz
        orient2d([a[0], a[2]], [b[0], b[2]], [c[0], c[2]]) == 0.0
    } else {
        // 丢弃 z，投影到 xy
        orient2d([a[0], a[1]], [b[0], b[1]], [c[0], c[1]]) == 0.0
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orient2d_ccw() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert!(orient2d(a, b, c) > 0.0);
        assert!(is_ccw2d(a, b, c));
    }

    #[test]
    fn orient2d_cw() {
        let a = [0.0, 0.0];
        let b = [0.0, 1.0];
        let c = [1.0, 0.0];
        assert!(orient2d(a, b, c) < 0.0);
        assert!(!is_ccw2d(a, b, c));
    }

    #[test]
    fn orient2d_collinear_exact() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [2.0, 0.0];
        assert_eq!(orient2d(a, b, c), 0.0);
        assert!(is_collinear2d(a, b, c));
    }

    #[test]
    fn orient2d_collinear_near() {
        // 朴素 f64 计算会因舍入给出非零结果，Shewchuk 自适应精度能正确识别为 0
        // 构造一个共线但浮点易错的例子
        let a = [1e20, 1.0];
        let b = [1e20 + 1.0, 1.0];
        let c = [1e20 + 2.0, 1.0];
        assert_eq!(orient2d(a, b, c), 0.0);
    }

    #[test]
    fn signed_area_2d_unit_triangle() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let area = signed_area2d(a, b, c);
        assert!((area - 0.5).abs() < 1e-15);
        assert!((triangle_area_2d(a, b, c) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn point_in_triangle_interior() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let p = [0.25, 0.25];
        assert!(point_in_triangle_2d(p, a, b, c));
    }

    #[test]
    fn point_in_triangle_vertex() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert!(point_in_triangle_2d(a, a, b, c));
        assert!(point_in_triangle_2d(b, a, b, c));
        assert!(point_in_triangle_2d(c, a, b, c));
    }

    #[test]
    fn point_in_triangle_edge() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let p = [0.5, 0.0]; // 边 ab 中点
        assert!(point_in_triangle_2d(p, a, b, c));
    }

    #[test]
    fn point_outside_triangle() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert!(!point_in_triangle_2d([0.6, 0.6], a, b, c));
        assert!(!point_in_triangle_2d([-0.1, 0.5], a, b, c));
        assert!(!point_in_triangle_2d([0.5, -0.1], a, b, c));
    }

    #[test]
    fn point_in_triangle_cw_orientation() {
        // CW 三角形：a, c, b
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let p = [0.25, 0.25];
        assert!(point_in_triangle_2d(p, a, c, b));
    }

    #[test]
    fn point_in_triangle_degenerate() {
        // 共线三角形
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [2.0, 0.0];
        let p = [0.5, 0.0];
        assert!(!point_in_triangle_2d(p, a, b, c));
    }

    #[test]
    fn orient3d_basic() {
        // abc 平面为 z=0，从 z>0 方向看 abc 呈 CCW
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // Shewchuk 约定：正值 = d 在平面下方（即 -z 侧）
        let d_below = [0.0, 0.0, -1.0];
        assert!(orient3d(a, b, c, d_below) > 0.0);
        // 负值 = d 在平面上方（即 +z 侧）
        let d_above = [0.0, 0.0, 1.0];
        assert!(orient3d(a, b, c, d_above) < 0.0);
        // 零 = 共面
        let d_on = [0.5, 0.5, 0.0];
        assert_eq!(orient3d(a, b, c, d_on), 0.0);
    }

    #[test]
    fn tet_signed_volume_unit() {
        // 标准正交四面体体积 = 1/6
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        let v = tet_signed_volume(a, b, c, d);
        assert!((v - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn orient3d_coplanar() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [1.0, 1.0, 0.0]; // 共面
        assert!(is_coplanar(a, b, c, d));
    }

    #[test]
    fn incircle_inside() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let d = [0.25, 0.25]; // 在外接圆内
        assert!(incircle(a, b, c, d) > 0.0);
    }

    #[test]
    fn incircle_outside() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let d = [2.0, 2.0]; // 在外接圆外
        assert!(incircle(a, b, c, d) < 0.0);
    }

    #[test]
    fn incircle_on_circle() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let c = [-1.0, 0.0];
        let d = [0.0, -1.0]; // 在单位圆上
        assert_eq!(incircle(a, b, c, d), 0.0);
    }

    #[test]
    fn insphere_inside() {
        // 选择 a, b, c, d 使 orient3d(a, b, c, d) > 0（insphere 要求正方向）
        // abc 平面 z=0，从 z<0 方向看呈 CCW（即 (b-a)×(c-a) 指向 -z）
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        let d = [0.0, 0.0, 1.0]; // 在 abc 下方 → orient3d > 0
        assert!(orient3d(a, b, c, d) > 0.0);
        let e = [0.1, 0.1, 0.1]; // 在外接球内
        assert!(insphere(a, b, c, d, e) > 0.0);
    }

    #[test]
    fn insphere_outside() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        assert!(orient3d(a, b, c, d) > 0.0);
        let e = [2.0, 2.0, 2.0]; // 在外接球外
        assert!(insphere(a, b, c, d, e) < 0.0);
    }

    #[test]
    fn convex_vertex_ccw() {
        // CCW 三角形的中间顶点 b 是凸的
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert!(is_convex_vertex2d(a, b, c));
        // 凹顶点（CW 顺序）
        let a2 = [0.0, 0.0];
        let b2 = [0.0, 1.0];
        let c2 = [1.0, 0.0];
        assert!(!is_convex_vertex2d(a2, b2, c2));
    }

    #[test]
    fn triangle_degenerate_3d_normal() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        assert!(!is_triangle_degenerate_3d(a, b, c));
    }

    #[test]
    fn triangle_degenerate_3d_collinear() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        let c = [2.0, 2.0, 2.0];
        assert!(is_triangle_degenerate_3d(a, b, c));
    }

    #[test]
    fn triangle_degenerate_3d_coincident() {
        let a = [1.0, 2.0, 3.0];
        assert!(is_triangle_degenerate_3d(a, a, a));
    }

    #[test]
    fn triangle_degenerate_3d_axis_aligned() {
        // 法向沿 z 轴最大，丢弃 z 投影到 xy
        let a = [0.0, 0.0, 5.0];
        let b = [1.0, 0.0, 5.0];
        let c = [0.0, 1.0, 5.0];
        assert!(!is_triangle_degenerate_3d(a, b, c));
        // 共线（沿 x 轴）
        let a2 = [0.0, 0.0, 5.0];
        let b2 = [1.0, 0.0, 5.0];
        let c2 = [2.0, 0.0, 5.0];
        assert!(is_triangle_degenerate_3d(a2, b2, c2));
    }
}
