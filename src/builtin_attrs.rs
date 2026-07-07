//! 内置属性模块：normal / uv / color / selection / size 作为一等公民属性。
//!
//! 与 [`crate::property`] 的泛型属性系统互补，本模块定义强类型 Newtype
//! 包装，让顶点法向 / 顶点 UV / 顶点颜色 / 面法向 / 选择态 / 边面色 /
//! 可视化尺寸成为内置属性：
//!
//! - [`VertexNormal`]：`[f64; 3]`，与 `geometry::vertex_normal` 一致；
//! - [`VertexColor`]：`[f64; 3]`，RGB 通道 ∈ [0, 1]；
//! - [`VertexUv`]：`[f64; 2]`，纹理坐标；
//! - [`FaceNormal`]：`[f64; 3]`，与 `geometry::face_normal` 一致；
//! - [`VertexSelected`] / [`HalfEdgeSelected`] / [`FaceSelected`]：`bool` 选择态；
//! - [`HalfEdgeColor`] / [`FaceColor`]：`[f64; 3]`，边/面颜色；
//! - [`VertexSize`]：`f64`，顶点显示半径；
//! - [`HalfEdgeThickness`]：`f64`，边粗细（twin 同步）；
//! - [`FaceOpacity`]：`f64`，面透明度 ∈ [0, 1]。
//!
//! 通过 [`MeshProperties`] 注册后，可通过句柄 `PropertyHandle<T>` 读写。
//! [`io`] 模块的 OBJ/PLY 路径会感知这些属性并双向同步。

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::property::{MeshProperties, PropertyHandle};
use crate::storage::MeshStorage;

// ============================================================
// Newtype 包装
// ============================================================

/// 顶点法向（单位向量）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VertexNormal(pub [f64; 3]);

/// 顶点颜色（RGB，每个通道 ∈ [0, 1]）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VertexColor(pub [f64; 3]);

/// 顶点 UV 纹理坐标。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VertexUv(pub [f64; 2]);

/// 面法向（单位向量）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FaceNormal(pub [f64; 3]);

/// 顶点选择态（`true` = 选中）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VertexSelected(pub bool);

/// 半边选择态（`true` = 选中）。无向边的选择应同时设置 twin 对，
/// 详见 [`select_edge`] / [`is_edge_selected`]。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HalfEdgeSelected(pub bool);

/// 面选择态（`true` = 选中）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FaceSelected(pub bool);

/// 半边颜色（RGB，每通道 ∈ [0, 1]）。无向边颜色应同时设置 twin 对，
/// 详见 [`set_edge_color`] / [`edge_color`]。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HalfEdgeColor(pub [f64; 3]);

/// 面颜色（RGB，每通道 ∈ [0, 1]）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FaceColor(pub [f64; 3]);

/// 顶点显示大小（半径，单位与 position 一致）。默认 0 表示不渲染标记。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VertexSize(pub f64);

/// 半边粗细（线宽，单位与 position 一致）。无向边粗细应同时设置 twin 对，
/// 详见 [`set_edge_thickness`] / [`edge_thickness`]。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HalfEdgeThickness(pub f64);

/// 面透明度 ∈ [0, 1]：0 = 完全透明，1 = 完全不透明。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FaceOpacity(pub f64);

// ============================================================
// 类型化句柄便捷函数
// ============================================================

/// 注册顶点法向属性，返回句柄。
pub fn add_vertex_normals(props: &mut MeshProperties) -> PropertyHandle<VertexNormal> {
    props.add_vertex_prop::<VertexNormal>()
}

/// 注册顶点颜色属性，返回句柄。
pub fn add_vertex_colors(props: &mut MeshProperties) -> PropertyHandle<VertexColor> {
    props.add_vertex_prop::<VertexColor>()
}

/// 注册顶点 UV 属性，返回句柄。
pub fn add_vertex_uvs(props: &mut MeshProperties) -> PropertyHandle<VertexUv> {
    props.add_vertex_prop::<VertexUv>()
}

/// 注册面法向属性，返回句柄。
pub fn add_face_normals(props: &mut MeshProperties) -> PropertyHandle<FaceNormal> {
    props.add_face_prop::<FaceNormal>()
}

/// 注册顶点选择态属性，返回句柄。
pub fn add_vertex_selection(props: &mut MeshProperties) -> PropertyHandle<VertexSelected> {
    props.add_vertex_prop::<VertexSelected>()
}

/// 注册半边选择态属性，返回句柄。
pub fn add_halfedge_selection(props: &mut MeshProperties) -> PropertyHandle<HalfEdgeSelected> {
    props.add_halfedge_prop::<HalfEdgeSelected>()
}

/// 注册面选择态属性，返回句柄。
pub fn add_face_selection(props: &mut MeshProperties) -> PropertyHandle<FaceSelected> {
    props.add_face_prop::<FaceSelected>()
}

/// 注册半边颜色属性，返回句柄。
pub fn add_halfedge_colors(props: &mut MeshProperties) -> PropertyHandle<HalfEdgeColor> {
    props.add_halfedge_prop::<HalfEdgeColor>()
}

/// 注册面颜色属性，返回句柄。
pub fn add_face_colors(props: &mut MeshProperties) -> PropertyHandle<FaceColor> {
    props.add_face_prop::<FaceColor>()
}

/// 注册顶点显示大小属性，返回句柄。
pub fn add_vertex_sizes(props: &mut MeshProperties) -> PropertyHandle<VertexSize> {
    props.add_vertex_prop::<VertexSize>()
}

/// 注册半边粗细属性，返回句柄。
pub fn add_halfedge_thickness(props: &mut MeshProperties) -> PropertyHandle<HalfEdgeThickness> {
    props.add_halfedge_prop::<HalfEdgeThickness>()
}

/// 注册面透明度属性，返回句柄。
pub fn add_face_opacity(props: &mut MeshProperties) -> PropertyHandle<FaceOpacity> {
    props.add_face_prop::<FaceOpacity>()
}

// ============================================================
// 批量计算与填充
// ============================================================

/// 为所有顶点计算法向并写入属性。
///
/// 法向通过 [`crate::geometry::vertex_normal`] 计算（面法向面积加权）。
/// 若某顶点法向不可计算（孤立 / 边界退化），跳过该顶点。
pub fn populate_vertex_normals(
    mesh: &crate::storage::MeshStorage,
    props: &mut MeshProperties,
    handle: PropertyHandle<VertexNormal>,
) {
    for v in mesh.vertex_ids() {
        if let Some(n) = crate::geometry::vertex_normal(mesh, v) {
            props.set_vertex_prop(handle, v, VertexNormal(n));
        }
    }
}

/// 为所有面计算法向并写入属性。
pub fn populate_face_normals(
    mesh: &crate::storage::MeshStorage,
    props: &mut MeshProperties,
    handle: PropertyHandle<FaceNormal>,
) {
    for f in mesh.face_ids() {
        if let Some(n) = crate::geometry::face_normal(mesh, f) {
            props.set_face_prop(handle, f, FaceNormal(n));
        }
    }
}

// ============================================================
// IO 辅助：导出/导入属性
// ============================================================

/// 收集所有顶点的法向（按 `vertex_ids()` 顺序）。
///
/// 若属性未注册或某顶点未设置法向，对应位置填 `[0,0,0]`。
pub fn collect_vertex_normals(
    mesh: &crate::storage::MeshStorage,
    props: &MeshProperties,
    handle: PropertyHandle<VertexNormal>,
) -> Vec<[f64; 3]> {
    mesh.vertex_ids()
        .map(|v| {
            props
                .get_vertex_prop(handle, v)
                .map(|n| n.0)
                .unwrap_or([0.0, 0.0, 0.0])
        })
        .collect()
}

/// 收集所有顶点的 UV（按 `vertex_ids()` 顺序）。
pub fn collect_vertex_uvs(
    mesh: &crate::storage::MeshStorage,
    props: &MeshProperties,
    handle: PropertyHandle<VertexUv>,
) -> Vec<[f64; 2]> {
    mesh.vertex_ids()
        .map(|v| {
            props
                .get_vertex_prop(handle, v)
                .map(|uv| uv.0)
                .unwrap_or([0.0, 0.0])
        })
        .collect()
}

/// 收集所有顶点的颜色（按 `vertex_ids()` 顺序）。
pub fn collect_vertex_colors(
    mesh: &crate::storage::MeshStorage,
    props: &MeshProperties,
    handle: PropertyHandle<VertexColor>,
) -> Vec<[f64; 3]> {
    mesh.vertex_ids()
        .map(|v| {
            props
                .get_vertex_prop(handle, v)
                .map(|c| c.0)
                .unwrap_or([1.0, 1.0, 1.0])
        })
        .collect()
}

/// 从顶点位置数组与可选属性数组重建属性系统。
///
/// 适用于 OBJ/PLY 解析后的属性同步：传入按顶点顺序排列的法向/UV/颜色
/// 切片，自动注册属性并填充。任一切片为 `None` 则跳过对应属性。
pub fn install_vertex_attrs(
    props: &mut MeshProperties,
    normals: Option<&[[f64; 3]]>,
    uvs: Option<&[[f64; 2]]>,
    colors: Option<&[[f64; 3]]>,
    vertex_ids: &[VertexId],
) {
    if let Some(ns) = normals {
        let h = add_vertex_normals(props);
        for (i, &v) in vertex_ids.iter().enumerate() {
            if let Some(n) = ns.get(i) {
                props.set_vertex_prop(h, v, VertexNormal(*n));
            }
        }
    }
    if let Some(uvs_arr) = uvs {
        let h = add_vertex_uvs(props);
        for (i, &v) in vertex_ids.iter().enumerate() {
            if let Some(uv) = uvs_arr.get(i) {
                props.set_vertex_prop(h, v, VertexUv(*uv));
            }
        }
    }
    if let Some(cs) = colors {
        let h = add_vertex_colors(props);
        for (i, &v) in vertex_ids.iter().enumerate() {
            if let Some(c) = cs.get(i) {
                props.set_vertex_prop(h, v, VertexColor(*c));
            }
        }
    }
}

// ============================================================
// OBJ 属性感知 IO
// ============================================================

use crate::io::{ObjError, build_mesh_from_polygons};

/// 解析 OBJ 文本，同时提取 `vn`（顶点法向）与 `vt`（顶点 UV）属性。
///
/// 返回 `(mesh, props)`，其中 `props` 已注册 `VertexNormal` 与 `VertexUv`。
/// 颜色信息不在标准 OBJ 中，需通过 PLY 通道。
///
/// 面 `f` 行支持 `v/vt/vn` 形式：本函数按面顶点顺序收集 `vt`/`vn` 索引，
/// 并按顶点位置去重后写入属性。若某顶点未被任何面引用，不写入属性。
pub fn parse_obj_with_attrs(
    text: &str,
) -> Result<(crate::storage::MeshStorage, MeshProperties), ObjError> {
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut texcoords: Vec<[f64; 2]> = Vec::new();
    let mut normals: Vec<[f64; 3]> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();
    // 每个面顶点 (vt_idx, vn_idx)（1-based，0 表示未指定）
    let mut face_attrs: Vec<Vec<(i64, i64)>> = Vec::new();

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let kind = match tokens.next() {
            Some(k) => k,
            None => continue,
        };
        match kind {
            "v" => {
                let coords: Vec<f64> = tokens
                    .take(3)
                    .map(|t| {
                        t.parse::<f64>().map_err(|_| ObjError::Parse {
                            line: line_no + 1,
                            msg: format!("无法解析顶点坐标: {}", t),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if coords.len() != 3 {
                    return Err(ObjError::Parse {
                        line: line_no + 1,
                        msg: "顶点行缺少坐标分量".into(),
                    });
                }
                vertices.push([coords[0], coords[1], coords[2]]);
            }
            "vt" => {
                let coords: Vec<f64> = tokens
                    .take(2)
                    .map(|t| {
                        t.parse::<f64>().map_err(|_| ObjError::Parse {
                            line: line_no + 1,
                            msg: format!("无法解析纹理坐标: {}", t),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if coords.len() == 2 {
                    texcoords.push([coords[0], coords[1]]);
                }
            }
            "vn" => {
                let coords: Vec<f64> = tokens
                    .take(3)
                    .map(|t| {
                        t.parse::<f64>().map_err(|_| ObjError::Parse {
                            line: line_no + 1,
                            msg: format!("无法解析法向: {}", t),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if coords.len() == 3 {
                    normals.push([coords[0], coords[1], coords[2]]);
                }
            }
            "f" => {
                let mut verts: Vec<i64> = Vec::new();
                let mut attrs: Vec<(i64, i64)> = Vec::new();
                for t in tokens {
                    // 三种形式：v / v/vt / v/vt/vn / v//vn
                    let parts: Vec<&str> = t.split('/').collect();
                    let v = parts[0].parse::<i64>().map_err(|_| ObjError::Parse {
                        line: line_no + 1,
                        msg: format!("无法解析面索引: {}", t),
                    })?;
                    let vt = parts
                        .get(1)
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or(0);
                    let vn = parts
                        .get(2)
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or(0);
                    verts.push(v);
                    attrs.push((vt, vn));
                }
                if verts.len() < 3 {
                    return Err(ObjError::NotTriangular {
                        line: line_no + 1,
                        face_verts: verts.len(),
                    });
                }
                let to_zero = |i: i64| -> Result<u32, ObjError> {
                    let zero_based = if i > 0 {
                        (i - 1) as usize
                    } else if i < 0 {
                        let n = vertices.len() as i64;
                        if n + i < 0 {
                            return Err(ObjError::IndexOutOfRange {
                                line: line_no + 1,
                                idx: i,
                                vertex_count: vertices.len(),
                            });
                        }
                        (n + i) as usize
                    } else {
                        return Err(ObjError::Parse {
                            line: line_no + 1,
                            msg: "面索引不能为 0".into(),
                        });
                    };
                    if zero_based >= vertices.len() {
                        return Err(ObjError::IndexOutOfRange {
                            line: line_no + 1,
                            idx: i,
                            vertex_count: vertices.len(),
                        });
                    }
                    Ok(zero_based as u32)
                };
                let indices: Vec<u32> = verts
                    .iter()
                    .map(|&i| to_zero(i))
                    .collect::<Result<_, _>>()?;
                faces.push(indices);
                face_attrs.push(attrs);
            }
            _ => {}
        }
    }

    let mesh = build_mesh_from_polygons(&vertices, &faces)
        .expect("OBJ indices already validated during parsing");
    let mut props = MeshProperties::new();

    // 把 vn / vt 属性按面顶点关联到对应顶点
    // 注意：同一顶点可能被多个面引用，后面的覆盖前面
    let mut vert_normal: Vec<Option<[f64; 3]>> = vec![None; vertices.len()];
    let mut vert_uv: Vec<Option<[f64; 2]>> = vec![None; vertices.len()];
    for (face_idx, attrs) in face_attrs.iter().enumerate() {
        let face = &faces[face_idx];
        for (k, &(vt, vn)) in attrs.iter().enumerate() {
            let v = face[k] as usize;
            if vn > 0 {
                let ni = (vn - 1) as usize;
                if ni < normals.len() {
                    vert_normal[v] = Some(normals[ni]);
                }
            }
            if vt > 0 {
                let ti = (vt - 1) as usize;
                if ti < texcoords.len() {
                    vert_uv[v] = Some(texcoords[ti]);
                }
            }
        }
    }

    let ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let ns: Vec<[f64; 3]> = vert_normal.iter().filter_map(|x| *x).collect();
    let uvs_v: Vec<[f64; 2]> = vert_uv.iter().filter_map(|x| *x).collect();
    let has_normals = !ns.is_empty();
    let has_uvs = !uvs_v.is_empty();

    if has_normals {
        let h = add_vertex_normals(&mut props);
        for (i, &v) in ids.iter().enumerate() {
            if let Some(n) = vert_normal.get(i).and_then(|x| *x) {
                props.set_vertex_prop(h, v, VertexNormal(n));
            }
        }
    }
    if has_uvs {
        let h = add_vertex_uvs(&mut props);
        for (i, &v) in ids.iter().enumerate() {
            if let Some(uv) = vert_uv.get(i).and_then(|x| *x) {
                props.set_vertex_prop(h, v, VertexUv(uv));
            }
        }
    }

    Ok((mesh, props))
}

/// 把网格与属性序列化为带 `vn`/`vt` 的 OBJ 文本。
///
/// 仅写出已设置属性的顶点；未设置属性的顶点写 0 占位。
pub fn format_obj_with_attrs(
    mesh: &crate::storage::MeshStorage,
    props: &MeshProperties,
    normal_handle: Option<PropertyHandle<VertexNormal>>,
    uv_handle: Option<PropertyHandle<VertexUv>>,
) -> String {
    let mut out = String::new();
    // 顶点行
    for v in mesh.vertex_ids() {
        let p = mesh.get_vertex(v).map(|v| v.position).unwrap_or([0.0; 3]);
        out.push_str(&format!("v {:.6} {:.6} {:.6}\n", p[0], p[1], p[2]));
    }
    // vt 行
    let mut vt_lines: Vec<String> = Vec::new();
    if let Some(h) = uv_handle {
        for v in mesh.vertex_ids() {
            if let Some(uv) = props.get_vertex_prop(h, v) {
                vt_lines.push(format!("vt {:.6} {:.6}\n", uv.0[0], uv.0[1]));
            } else {
                vt_lines.push("vt 0.000000 0.000000\n".into());
            }
        }
    }
    // vn 行
    let mut vn_lines: Vec<String> = Vec::new();
    if let Some(h) = normal_handle {
        for v in mesh.vertex_ids() {
            if let Some(n) = props.get_vertex_prop(h, v) {
                vn_lines.push(format!("vn {:.6} {:.6} {:.6}\n", n.0[0], n.0[1], n.0[2]));
            } else {
                vn_lines.push("vn 0.000000 0.000000 0.000000\n".into());
            }
        }
    }
    for l in &vt_lines {
        out.push_str(l);
    }
    for l in &vn_lines {
        out.push_str(l);
    }
    // 面行：v/vt/vn 形式
    let has_vt = !vt_lines.is_empty();
    let has_vn = !vn_lines.is_empty();
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.is_empty() {
            continue;
        }
        out.push('f');
        for (i, v) in verts.iter().enumerate() {
            let v_idx = vertex_index(mesh, *v) + 1;
            if has_vt && has_vn {
                out.push_str(&format!(" {}/{}/{}", v_idx, v_idx, v_idx));
            } else if has_vt {
                out.push_str(&format!(" {}/{}", v_idx, v_idx));
            } else if has_vn {
                out.push_str(&format!(" {}//{}", v_idx, v_idx));
            } else {
                out.push_str(&format!(" {}", v_idx));
            }
            let _ = i;
        }
        out.push('\n');
    }
    out
}

/// 内部辅助：按 `vertex_ids()` 顺序返回指定顶点的 1-based 索引。
fn vertex_index(mesh: &crate::storage::MeshStorage, target: VertexId) -> usize {
    mesh.vertex_ids().position(|v| v == target).unwrap_or(0)
}

// ============================================================
// 选择态便捷 API
// ============================================================
//
// 对 vertex / halfedge / face 三类元素提供统一的「选择 / 取消 / 切换 / 查询 /
// 清空 / 遍历选中」接口。语义统一：未设置属性视为 `false`，避免强制初始化。

// ---------- Vertex 选择 ----------

/// 选中顶点 `id`。
#[inline]
pub fn select_vertex(props: &mut MeshProperties, h: PropertyHandle<VertexSelected>, id: VertexId) {
    props.set_vertex_prop(h, id, VertexSelected(true));
}

/// 取消选中顶点 `id`。
#[inline]
pub fn deselect_vertex(
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSelected>,
    id: VertexId,
) {
    props.set_vertex_prop(h, id, VertexSelected(false));
}

/// 切换顶点 `id` 的选择态，返回切换后的新状态。
#[inline]
pub fn toggle_vertex_selection(
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSelected>,
    id: VertexId,
) -> bool {
    let cur = is_vertex_selected(props, h, id);
    props.set_vertex_prop(h, id, VertexSelected(!cur));
    !cur
}

/// 查询顶点 `id` 是否被选中。未注册或未设置返回 `false`。
#[inline]
pub fn is_vertex_selected(
    props: &MeshProperties,
    h: PropertyHandle<VertexSelected>,
    id: VertexId,
) -> bool {
    props.get_vertex_prop(h, id).map(|s| s.0).unwrap_or(false)
}

/// 清空所有顶点选择（保留其他属性与其他类型的注册）。
pub fn clear_vertex_selection(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSelected>,
) {
    for v in mesh.vertex_ids() {
        props.remove_vertex_prop(h, v);
    }
}

/// 遍历所有选中的顶点 ID（懒迭代器）。
pub fn selected_vertex_ids<'a>(
    mesh: &'a MeshStorage,
    props: &'a MeshProperties,
    h: PropertyHandle<VertexSelected>,
) -> impl Iterator<Item = VertexId> + 'a {
    mesh.vertex_ids()
        .filter(move |&v| is_vertex_selected(props, h, v))
}

/// 选中所有顶点。
pub fn select_all_vertices(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSelected>,
) {
    for v in mesh.vertex_ids() {
        props.set_vertex_prop(h, v, VertexSelected(true));
    }
}

/// 反转顶点选择（选中→取消，未选中→选中）。
pub fn invert_vertex_selection(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSelected>,
) {
    for v in mesh.vertex_ids() {
        let next = !is_vertex_selected(props, h, v);
        props.set_vertex_prop(h, v, VertexSelected(next));
    }
}

/// 统计被选中的顶点数。
pub fn count_selected_vertices(
    mesh: &MeshStorage,
    props: &MeshProperties,
    h: PropertyHandle<VertexSelected>,
) -> usize {
    selected_vertex_ids(mesh, props, h).count()
}

// ---------- HalfEdge 选择（按半边） ----------

/// 选中半边 `id`。
#[inline]
pub fn select_halfedge(
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    id: HalfEdgeId,
) {
    props.set_halfedge_prop(h, id, HalfEdgeSelected(true));
}

/// 取消选中半边 `id`。
#[inline]
pub fn deselect_halfedge(
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    id: HalfEdgeId,
) {
    props.set_halfedge_prop(h, id, HalfEdgeSelected(false));
}

/// 切换半边 `id` 的选择态，返回切换后的新状态。
#[inline]
pub fn toggle_halfedge_selection(
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    id: HalfEdgeId,
) -> bool {
    let cur = is_halfedge_selected(props, h, id);
    props.set_halfedge_prop(h, id, HalfEdgeSelected(!cur));
    !cur
}

/// 查询半边 `id` 是否被选中。未注册或未设置返回 `false`。
#[inline]
pub fn is_halfedge_selected(
    props: &MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    id: HalfEdgeId,
) -> bool {
    props.get_halfedge_prop(h, id).map(|s| s.0).unwrap_or(false)
}

/// 清空所有半边选择（保留其他属性）。
pub fn clear_halfedge_selection(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
) {
    for he in mesh.halfedge_ids() {
        props.remove_halfedge_prop(h, he);
    }
}

/// 遍历所有选中的半边 ID（懒迭代器）。
pub fn selected_halfedge_ids<'a>(
    mesh: &'a MeshStorage,
    props: &'a MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
) -> impl Iterator<Item = HalfEdgeId> + 'a {
    mesh.halfedge_ids()
        .filter(move |&he| is_halfedge_selected(props, h, he))
}

// ---------- Face 选择 ----------

/// 选中面 `id`。
#[inline]
pub fn select_face(props: &mut MeshProperties, h: PropertyHandle<FaceSelected>, id: FaceId) {
    props.set_face_prop(h, id, FaceSelected(true));
}

/// 取消选中面 `id`。
#[inline]
pub fn deselect_face(props: &mut MeshProperties, h: PropertyHandle<FaceSelected>, id: FaceId) {
    props.set_face_prop(h, id, FaceSelected(false));
}

/// 切换面 `id` 的选择态，返回切换后的新状态。
#[inline]
pub fn toggle_face_selection(
    props: &mut MeshProperties,
    h: PropertyHandle<FaceSelected>,
    id: FaceId,
) -> bool {
    let cur = is_face_selected(props, h, id);
    props.set_face_prop(h, id, FaceSelected(!cur));
    !cur
}

/// 查询面 `id` 是否被选中。未注册或未设置返回 `false`。
#[inline]
pub fn is_face_selected(
    props: &MeshProperties,
    h: PropertyHandle<FaceSelected>,
    id: FaceId,
) -> bool {
    props.get_face_prop(h, id).map(|s| s.0).unwrap_or(false)
}

/// 清空所有面选择（保留其他属性）。
pub fn clear_face_selection(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<FaceSelected>,
) {
    for f in mesh.face_ids() {
        props.remove_face_prop(h, f);
    }
}

/// 遍历所有选中的面 ID（懒迭代器）。
pub fn selected_face_ids<'a>(
    mesh: &'a MeshStorage,
    props: &'a MeshProperties,
    h: PropertyHandle<FaceSelected>,
) -> impl Iterator<Item = FaceId> + 'a {
    mesh.face_ids()
        .filter(move |&f| is_face_selected(props, h, f))
}

/// 选中所有面。
pub fn select_all_faces(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<FaceSelected>,
) {
    for f in mesh.face_ids() {
        props.set_face_prop(h, f, FaceSelected(true));
    }
}

/// 统计被选中的面数。
pub fn count_selected_faces(
    mesh: &MeshStorage,
    props: &MeshProperties,
    h: PropertyHandle<FaceSelected>,
) -> usize {
    selected_face_ids(mesh, props, h).count()
}

// ============================================================
// 边级（twin 对）选择 / 颜色 API
// ============================================================
//
// 无向边由两条互为 twin 的半边组成。从语义上，边的选择/颜色应当一致。
// 下面这一组助手自动同步 twin 对，调用方只需提供任一半边 ID。
//
// 约定：若 `twin = None`（边界），仅作用于本半边。

/// 选中由半边 `he` 表示的无向边（同时设置 `he` 与其 twin）。
pub fn select_edge(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    he: HalfEdgeId,
) {
    props.set_halfedge_prop(h, he, HalfEdgeSelected(true));
    if let Some(twin) = mesh.get_halfedge(he).and_then(|h| h.twin) {
        props.set_halfedge_prop(h, twin, HalfEdgeSelected(true));
    }
}

/// 取消选中由半边 `he` 表示的无向边。
pub fn deselect_edge(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    he: HalfEdgeId,
) {
    props.set_halfedge_prop(h, he, HalfEdgeSelected(false));
    if let Some(twin) = mesh.get_halfedge(he).and_then(|h| h.twin) {
        props.set_halfedge_prop(h, twin, HalfEdgeSelected(false));
    }
}

/// 切换无向边的选择态，返回切换后的新状态（基于 `he` 本身的状态）。
pub fn toggle_edge_selection(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    he: HalfEdgeId,
) -> bool {
    let next = !is_edge_selected(mesh, props, h, he);
    if next {
        select_edge(mesh, props, h, he);
    } else {
        deselect_edge(mesh, props, h, he);
    }
    next
}

/// 查询由半边 `he` 表示的无向边是否被选中（任一 half 被选中即视为选中）。
pub fn is_edge_selected(
    mesh: &MeshStorage,
    props: &MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
    he: HalfEdgeId,
) -> bool {
    if is_halfedge_selected(props, h, he) {
        return true;
    }
    mesh.get_halfedge(he)
        .and_then(|h| h.twin)
        .map(|twin| is_halfedge_selected(props, h, twin))
        .unwrap_or(false)
}

/// 遍历所有被选中的无向边，每个边返回一次（canonical half：`he <= twin`）。
pub fn selected_edge_ids<'a>(
    mesh: &'a MeshStorage,
    props: &'a MeshProperties,
    h: PropertyHandle<HalfEdgeSelected>,
) -> impl Iterator<Item = HalfEdgeId> + 'a {
    mesh.halfedge_ids().filter(move |&he| {
        if !is_halfedge_selected(props, h, he) {
            return false;
        }
        // canonical：仅当 twin 不存在或 twin >= he 时输出
        match mesh.get_halfedge(he).and_then(|h| h.twin) {
            None => true,
            Some(twin) => he <= twin,
        }
    })
}

// ---------- 边颜色（twin 同步） ----------

/// 设置由半边 `he` 表示的无向边颜色（同时设置 `he` 与其 twin）。
pub fn set_edge_color(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeColor>,
    he: HalfEdgeId,
    color: [f64; 3],
) {
    props.set_halfedge_prop(h, he, HalfEdgeColor(color));
    if let Some(twin) = mesh.get_halfedge(he).and_then(|h| h.twin) {
        props.set_halfedge_prop(h, twin, HalfEdgeColor(color));
    }
}

/// 查询由半边 `he` 表示的无向边颜色（优先 `he`，未设置则尝试 twin）。
pub fn edge_color(
    mesh: &MeshStorage,
    props: &MeshProperties,
    h: PropertyHandle<HalfEdgeColor>,
    he: HalfEdgeId,
) -> Option<[f64; 3]> {
    if let Some(c) = props.get_halfedge_prop(h, he).map(|c| c.0) {
        return Some(c);
    }
    mesh.get_halfedge(he)
        .and_then(|h| h.twin)
        .and_then(|twin| props.get_halfedge_prop(h, twin).map(|c| c.0))
}

// ============================================================
// 面颜色便捷 API
// ============================================================

/// 设置面颜色。
#[inline]
pub fn set_face_color(
    props: &mut MeshProperties,
    h: PropertyHandle<FaceColor>,
    id: FaceId,
    color: [f64; 3],
) {
    props.set_face_prop(h, id, FaceColor(color));
}

/// 查询面颜色。
#[inline]
pub fn face_color(
    props: &MeshProperties,
    h: PropertyHandle<FaceColor>,
    id: FaceId,
) -> Option<[f64; 3]> {
    props.get_face_prop(h, id).map(|c| c.0)
}

/// 清空所有面颜色（保留其他属性）。
pub fn clear_face_colors(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<FaceColor>,
) {
    for f in mesh.face_ids() {
        props.remove_face_prop(h, f);
    }
}

// ============================================================
// 可视化尺寸 / 粗细 / 透明度 API
// ============================================================
//
// 顶点大小（半径）、半边粗细（twin 同步）、面透明度。
// 与颜色 API 形态一致：未注册视为 0（不渲染标记）或 None。

// ---------- 顶点大小 ----------

/// 设置顶点 `id` 的显示大小（半径）。
#[inline]
pub fn set_vertex_size(
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSize>,
    id: VertexId,
    size: f64,
) {
    props.set_vertex_prop(h, id, VertexSize(size));
}

/// 查询顶点 `id` 的显示大小。未注册或未设置返回 `None`。
#[inline]
pub fn vertex_size(
    props: &MeshProperties,
    h: PropertyHandle<VertexSize>,
    id: VertexId,
) -> Option<f64> {
    props.get_vertex_prop(h, id).map(|s| s.0)
}

/// 清空所有顶点大小（保留其他属性）。
pub fn clear_vertex_sizes(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSize>,
) {
    for v in mesh.vertex_ids() {
        props.remove_vertex_prop(h, v);
    }
}

/// 批量设置所有顶点大小为同一值。
pub fn set_uniform_vertex_size(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<VertexSize>,
    size: f64,
) {
    for v in mesh.vertex_ids() {
        props.set_vertex_prop(h, v, VertexSize(size));
    }
}

// ---------- 边粗细（twin 同步） ----------

/// 设置由半边 `he` 表示的无向边粗细（同时设置 `he` 与其 twin）。
pub fn set_edge_thickness(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeThickness>,
    he: HalfEdgeId,
    thickness: f64,
) {
    props.set_halfedge_prop(h, he, HalfEdgeThickness(thickness));
    if let Some(twin) = mesh.get_halfedge(he).and_then(|h| h.twin) {
        props.set_halfedge_prop(h, twin, HalfEdgeThickness(thickness));
    }
}

/// 查询由半边 `he` 表示的无向边粗细（优先 `he`，未设置则尝试 twin）。
pub fn edge_thickness(
    mesh: &MeshStorage,
    props: &MeshProperties,
    h: PropertyHandle<HalfEdgeThickness>,
    he: HalfEdgeId,
) -> Option<f64> {
    if let Some(t) = props.get_halfedge_prop(h, he).map(|t| t.0) {
        return Some(t);
    }
    mesh.get_halfedge(he)
        .and_then(|h| h.twin)
        .and_then(|twin| props.get_halfedge_prop(h, twin).map(|t| t.0))
}

/// 设置单条半边的粗细（不同步 twin，用于非对称渲染）。
#[inline]
pub fn set_halfedge_thickness(
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeThickness>,
    he: HalfEdgeId,
    thickness: f64,
) {
    props.set_halfedge_prop(h, he, HalfEdgeThickness(thickness));
}

/// 查询单条半边的粗细。
#[inline]
pub fn halfedge_thickness(
    props: &MeshProperties,
    h: PropertyHandle<HalfEdgeThickness>,
    he: HalfEdgeId,
) -> Option<f64> {
    props.get_halfedge_prop(h, he).map(|t| t.0)
}

/// 清空所有半边粗细（保留其他属性）。
pub fn clear_halfedge_thickness(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeThickness>,
) {
    for he in mesh.halfedge_ids() {
        props.remove_halfedge_prop(h, he);
    }
}

/// 批量设置所有边的粗细为同一值（每条边自动同步 twin）。
pub fn set_uniform_edge_thickness(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<HalfEdgeThickness>,
    thickness: f64,
) {
    // 仅处理 canonical half，避免重复设置
    let canonical: Vec<HalfEdgeId> = mesh
        .halfedge_ids()
        .filter(|&he| match mesh.get_halfedge(he).and_then(|h| h.twin) {
            None => true,
            Some(twin) => he <= twin,
        })
        .collect();
    for he in canonical {
        set_edge_thickness(mesh, props, h, he, thickness);
    }
}

// ---------- 面透明度 ----------

/// 设置面 `id` 的透明度（0 = 完全透明，1 = 完全不透明）。
/// 输入会自动 clamp 到 [0, 1]。
#[inline]
pub fn set_face_opacity(
    props: &mut MeshProperties,
    h: PropertyHandle<FaceOpacity>,
    id: FaceId,
    opacity: f64,
) {
    let clamped = opacity.clamp(0.0, 1.0);
    props.set_face_prop(h, id, FaceOpacity(clamped));
}

/// 查询面 `id` 的透明度。未注册或未设置返回 `None`。
#[inline]
pub fn face_opacity(
    props: &MeshProperties,
    h: PropertyHandle<FaceOpacity>,
    id: FaceId,
) -> Option<f64> {
    props.get_face_prop(h, id).map(|o| o.0)
}

/// 清空所有面透明度（保留其他属性）。
pub fn clear_face_opacity(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<FaceOpacity>,
) {
    for f in mesh.face_ids() {
        props.remove_face_prop(h, f);
    }
}

/// 批量设置所有面透明度为同一值。
pub fn set_uniform_face_opacity(
    mesh: &MeshStorage,
    props: &mut MeshProperties,
    h: PropertyHandle<FaceOpacity>,
    opacity: f64,
) {
    let clamped = opacity.clamp(0.0, 1.0);
    for f in mesh.face_ids() {
        props.set_face_prop(h, f, FaceOpacity(clamped));
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MeshStorage;

    #[test]
    fn vertex_normal_roundtrip() {
        let mesh = crate::test_util::build_icosphere(1);
        let mut props = MeshProperties::new();
        let h = add_vertex_normals(&mut props);
        populate_vertex_normals(&mesh, &mut props, h);

        let normals = collect_vertex_normals(&mesh, &props, h);
        assert_eq!(normals.len(), mesh.vertex_count());
        // 每个法向应是单位向量
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-10, "法向应单位化, len={len}");
        }
    }

    #[test]
    fn face_normal_roundtrip() {
        let mesh = crate::test_util::build_icosphere(1);
        let mut props = MeshProperties::new();
        let h = add_face_normals(&mut props);
        populate_face_normals(&mesh, &mut props, h);

        let count = mesh.face_count();
        let mut found = 0;
        for f in mesh.face_ids() {
            if props.get_face_prop(h, f).is_some() {
                found += 1;
            }
        }
        assert_eq!(found, count);
    }

    #[test]
    fn vertex_color_default_is_white() {
        let mesh = crate::test_util::build_icosphere(0);
        let mut props = MeshProperties::new();
        let h = add_vertex_colors(&mut props);
        // 没有设置任何颜色，应返回默认白色
        let colors = collect_vertex_colors(&mesh, &props, h);
        assert_eq!(colors.len(), mesh.vertex_count());
        for c in &colors {
            assert_eq!(*c, [1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn vertex_uv_set_and_get() {
        let mesh = crate::test_util::build_icosphere(0);
        let mut props = MeshProperties::new();
        let h = add_vertex_uvs(&mut props);
        let v0 = mesh.vertex_ids().next().unwrap();
        props.set_vertex_prop(h, v0, VertexUv([0.5, 0.7]));
        let uv = props.get_vertex_prop(h, v0).map(|u| u.0);
        assert_eq!(uv, Some([0.5, 0.7]));
    }

    #[test]
    fn install_vertex_attrs_handles_partial_input() {
        let mesh = crate::test_util::build_icosphere(0);
        let ids: Vec<VertexId> = mesh.vertex_ids().collect();
        let mut props = MeshProperties::new();

        // 只传法向，不传 UV 与颜色
        let normals: Vec<[f64; 3]> = ids.iter().map(|_| [0.0, 1.0, 0.0]).collect();
        install_vertex_attrs(&mut props, Some(&normals), None, None, &ids);

        assert!(props.has_vertex_prop::<VertexNormal>());
        assert!(!props.has_vertex_prop::<VertexUv>());
        assert!(!props.has_vertex_prop::<VertexColor>());
    }

    #[test]
    fn install_vertex_attrs_all_three() {
        let mesh = crate::test_util::build_icosphere(0);
        let ids: Vec<VertexId> = mesh.vertex_ids().collect();
        let mut props = MeshProperties::new();

        let normals: Vec<[f64; 3]> = ids.iter().map(|_| [0.0, 1.0, 0.0]).collect();
        let uvs: Vec<[f64; 2]> = ids
            .iter()
            .enumerate()
            .map(|(i, _)| [i as f64 * 0.1, 1.0 - i as f64 * 0.1])
            .collect();
        let colors: Vec<[f64; 3]> = ids.iter().map(|_| [0.5, 0.5, 0.5]).collect();

        install_vertex_attrs(&mut props, Some(&normals), Some(&uvs), Some(&colors), &ids);

        assert!(props.has_vertex_prop::<VertexNormal>());
        assert!(props.has_vertex_prop::<VertexUv>());
        assert!(props.has_vertex_prop::<VertexColor>());
    }

    #[test]
    fn newtype_default_is_zero() {
        let n = VertexNormal::default();
        assert_eq!(n.0, [0.0, 0.0, 0.0]);
        let uv = VertexUv::default();
        assert_eq!(uv.0, [0.0, 0.0]);
        let c = VertexColor::default();
        assert_eq!(c.0, [0.0, 0.0, 0.0]);
        let f = FaceNormal::default();
        assert_eq!(f.0, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn missing_handle_returns_none() {
        let mesh = crate::test_util::build_icosphere(0);
        let props = MeshProperties::new();
        let h = PropertyHandle::<VertexNormal>::default();
        let v = mesh.vertex_ids().next().unwrap();
        assert!(props.get_vertex_prop(h, v).is_none());
    }

    #[test]
    fn populate_face_normals_skips_invalid() {
        // 空网格不应 panic
        let mesh = MeshStorage::new();
        let mut props = MeshProperties::new();
        let h = add_face_normals(&mut props);
        populate_face_normals(&mesh, &mut props, h);
        assert_eq!(mesh.face_count(), 0);
    }

    // ---------- 选择态测试 ----------

    fn build_two_tri_mesh() -> (MeshStorage, [VertexId; 4], [HalfEdgeId; 6], [FaceId; 2]) {
        // 两三角形共享一条边：v0-v1-v2 与 v1-v3-v2
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let vids: Vec<VertexId> = mesh.vertex_ids().collect();
        let heids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
        let fids: Vec<FaceId> = mesh.face_ids().collect();
        assert!(vids.len() >= 4, "需要 4 个顶点，实际 {}", vids.len());
        assert!(heids.len() >= 6, "需要至少 6 条半边");
        assert_eq!(fids.len(), 2, "需要 2 个面");
        (
            mesh,
            [vids[0], vids[1], vids[2], vids[3]],
            [heids[0], heids[1], heids[2], heids[3], heids[4], heids[5]],
            [fids[0], fids[1]],
        )
    }

    #[test]
    fn vertex_selection_basic() {
        let mesh = crate::test_util::build_icosphere(0);
        let mut props = MeshProperties::new();
        let h = add_vertex_selection(&mut props);
        let v0 = mesh.vertex_ids().next().unwrap();

        // 初始未选中
        assert!(!is_vertex_selected(&props, h, v0));
        assert_eq!(count_selected_vertices(&mesh, &props, h), 0);

        select_vertex(&mut props, h, v0);
        assert!(is_vertex_selected(&props, h, v0));
        assert_eq!(count_selected_vertices(&mesh, &props, h), 1);

        // toggle 关闭
        let new_state = toggle_vertex_selection(&mut props, h, v0);
        assert!(!new_state);
        assert!(!is_vertex_selected(&props, h, v0));

        // toggle 重新打开
        let new_state = toggle_vertex_selection(&mut props, h, v0);
        assert!(new_state);
        assert!(is_vertex_selected(&props, h, v0));
    }

    #[test]
    fn vertex_select_all_and_invert() {
        let mesh = crate::test_util::build_icosphere(0);
        let n = mesh.vertex_count();
        let mut props = MeshProperties::new();
        let h = add_vertex_selection(&mut props);

        select_all_vertices(&mesh, &mut props, h);
        assert_eq!(count_selected_vertices(&mesh, &props, h), n);

        invert_vertex_selection(&mesh, &mut props, h);
        assert_eq!(count_selected_vertices(&mesh, &props, h), 0);

        // 再反转回来
        invert_vertex_selection(&mesh, &mut props, h);
        assert_eq!(count_selected_vertices(&mesh, &props, h), n);
    }

    #[test]
    fn vertex_clear_selection_keeps_registration() {
        let mesh = crate::test_util::build_icosphere(0);
        let mut props = MeshProperties::new();
        let h = add_vertex_selection(&mut props);
        select_all_vertices(&mesh, &mut props, h);

        clear_vertex_selection(&mesh, &mut props, h);
        assert_eq!(count_selected_vertices(&mesh, &props, h), 0);
        // 类型注册仍保留
        assert!(props.has_vertex_prop::<VertexSelected>());
    }

    #[test]
    fn selected_vertex_ids_iterator_correct() {
        let mesh = crate::test_util::build_icosphere(0);
        let mut props = MeshProperties::new();
        let h = add_vertex_selection(&mut props);
        let vids: Vec<VertexId> = mesh.vertex_ids().collect();
        // 选中第 0、2 个
        select_vertex(&mut props, h, vids[0]);
        select_vertex(&mut props, h, vids[2]);

        let collected: Vec<VertexId> = selected_vertex_ids(&mesh, &props, h).collect();
        assert_eq!(collected.len(), 2);
        assert!(collected.contains(&vids[0]));
        assert!(collected.contains(&vids[2]));
    }

    #[test]
    fn face_selection_basic() {
        let (mesh, _, _, fids) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_face_selection(&mut props);

        assert!(!is_face_selected(&props, h, fids[0]));
        select_face(&mut props, h, fids[0]);
        assert!(is_face_selected(&props, h, fids[0]));
        assert!(!is_face_selected(&props, h, fids[1]));

        assert_eq!(count_selected_faces(&mesh, &props, h), 1);

        select_all_faces(&mesh, &mut props, h);
        assert_eq!(count_selected_faces(&mesh, &props, h), 2);

        clear_face_selection(&mesh, &mut props, h);
        assert_eq!(count_selected_faces(&mesh, &props, h), 0);
    }

    #[test]
    fn halfedge_selection_basic() {
        let (mesh, _, heids, _) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_halfedge_selection(&mut props);

        select_halfedge(&mut props, h, heids[0]);
        assert!(is_halfedge_selected(&props, h, heids[0]));
        assert!(!is_halfedge_selected(&props, h, heids[1]));

        let count = selected_halfedge_ids(&mesh, &props, h).count();
        assert_eq!(count, 1);

        toggle_halfedge_selection(&mut props, h, heids[0]);
        assert!(!is_halfedge_selected(&props, h, heids[0]));

        clear_halfedge_selection(&mesh, &mut props, h);
        assert_eq!(selected_halfedge_ids(&mesh, &props, h).count(), 0);
    }

    #[test]
    fn edge_selection_syncs_twin_pair() {
        let (mesh, _, heids, _) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_halfedge_selection(&mut props);

        // 找一对 twin 半边
        let he0 = heids[0];
        let twin = mesh
            .get_halfedge(he0)
            .and_then(|h| h.twin)
            .expect("应至少有一条带 twin 的内部边");

        // 选中 he0，twin 应也被同步选中
        select_edge(&mesh, &mut props, h, he0);
        assert!(is_halfedge_selected(&props, h, he0));
        assert!(is_halfedge_selected(&props, h, twin));
        assert!(is_edge_selected(&mesh, &props, h, he0));
        assert!(is_edge_selected(&mesh, &props, h, twin));

        // 取消选中
        deselect_edge(&mesh, &mut props, h, he0);
        assert!(!is_halfedge_selected(&props, h, he0));
        assert!(!is_halfedge_selected(&props, h, twin));
        assert!(!is_edge_selected(&mesh, &props, h, he0));

        // 仅选中 twin 时，is_edge_selected 也应返回 true
        select_halfedge(&mut props, h, twin);
        assert!(!is_halfedge_selected(&props, h, he0));
        assert!(is_halfedge_selected(&props, h, twin));
        assert!(is_edge_selected(&mesh, &props, h, he0));
        assert!(is_edge_selected(&mesh, &props, h, twin));
    }

    #[test]
    fn selected_edge_ids_returns_canonical_only() {
        let (mesh, _, heids, _) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_halfedge_selection(&mut props);

        // 选中两条带 twin 的边（4 条半边）
        for &he in &heids[..4] {
            select_edge(&mesh, &mut props, h, he);
        }
        // selected_edge_ids 应去重，每个无向边只返回一次
        let edges: Vec<HalfEdgeId> = selected_edge_ids(&mesh, &props, h).collect();
        // 2 个三角形的内部边数 = 1（共享边）；外部边数 = 4；共 5 条无向边
        // 但我们只选了前 4 条半边所在的边，可能覆盖 2~4 条无向边
        // 验证：每个返回的 he，其 twin 不在结果集中
        for &he in &edges {
            if let Some(twin) = mesh.get_halfedge(he).and_then(|h| h.twin) {
                assert!(
                    !edges.contains(&twin),
                    "twin {twin:?} 与 {he:?} 不应同时出现"
                );
            }
        }
        assert!(!edges.is_empty());
    }

    #[test]
    fn edge_color_syncs_twin_pair() {
        let (mesh, _, heids, _) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_halfedge_colors(&mut props);

        let he0 = heids[0];
        let twin = mesh
            .get_halfedge(he0)
            .and_then(|h| h.twin)
            .expect("应至少有一条带 twin 的内部边");

        // 设置 he0 的颜色，twin 也应同步
        set_edge_color(&mesh, &mut props, h, he0, [0.2, 0.4, 0.6]);
        assert_eq!(
            props.get_halfedge_prop(h, he0).map(|c| c.0),
            Some([0.2, 0.4, 0.6])
        );
        assert_eq!(
            props.get_halfedge_prop(h, twin).map(|c| c.0),
            Some([0.2, 0.4, 0.6])
        );

        // edge_color 任一半边都能读到
        assert_eq!(edge_color(&mesh, &props, h, he0), Some([0.2, 0.4, 0.6]));
        assert_eq!(edge_color(&mesh, &props, h, twin), Some([0.2, 0.4, 0.6]));

        // 未设置颜色的边返回 None
        let uncolored = heids.iter().find(|&&he| {
            !is_halfedge_colored(&props, h, he)
                && mesh
                    .get_halfedge(he)
                    .and_then(|h| h.twin)
                    .is_some_and(|t| !is_halfedge_colored(&props, h, t))
        });
        if let Some(&uncolored_he) = uncolored {
            assert_eq!(edge_color(&mesh, &props, h, uncolored_he), None);
        }
    }

    fn is_halfedge_colored(
        props: &MeshProperties,
        h: PropertyHandle<HalfEdgeColor>,
        he: HalfEdgeId,
    ) -> bool {
        props.get_halfedge_prop(h, he).is_some()
    }

    #[test]
    fn face_color_basic() {
        let (mesh, _, _, fids) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_face_colors(&mut props);

        assert_eq!(face_color(&props, h, fids[0]), None);

        set_face_color(&mut props, h, fids[0], [0.8, 0.1, 0.1]);
        assert_eq!(face_color(&props, h, fids[0]), Some([0.8, 0.1, 0.1]));
        assert_eq!(face_color(&props, h, fids[1]), None);

        clear_face_colors(&mesh, &mut props, h);
        assert_eq!(face_color(&props, h, fids[0]), None);
        assert_eq!(face_color(&props, h, fids[1]), None);
    }

    #[test]
    fn selection_newtype_default_is_false() {
        assert!(!VertexSelected::default().0);
        assert!(!HalfEdgeSelected::default().0);
        assert!(!FaceSelected::default().0);
    }

    #[test]
    fn color_newtype_default_is_zero() {
        assert_eq!(HalfEdgeColor::default().0, [0.0, 0.0, 0.0]);
        assert_eq!(FaceColor::default().0, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn unregistered_selection_safe_no_panic() {
        // 未注册属性时调用查询接口不应 panic
        let mesh = crate::test_util::build_icosphere(0);
        let props = MeshProperties::new();
        let h: PropertyHandle<VertexSelected> = PropertyHandle::new();
        let v = mesh.vertex_ids().next().unwrap();
        assert!(!is_vertex_selected(&props, h, v));
        assert_eq!(count_selected_vertices(&mesh, &props, h), 0);
        assert_eq!(selected_vertex_ids(&mesh, &props, h).count(), 0);
    }

    #[test]
    fn unregistered_edge_safe_no_panic() {
        let (mesh, _, heids, _) = build_two_tri_mesh();
        let props = MeshProperties::new();
        let h: PropertyHandle<HalfEdgeSelected> = PropertyHandle::new();
        let he = heids[0];
        assert!(!is_edge_selected(&mesh, &props, h, he));
        assert_eq!(selected_edge_ids(&mesh, &props, h).count(), 0);
    }

    #[test]
    fn empty_mesh_selection_no_panic() {
        // 空网格上调用选择接口不应 panic
        let mesh = MeshStorage::new();
        let mut props = MeshProperties::new();
        let h = add_vertex_selection(&mut props);

        select_all_vertices(&mesh, &mut props, h);
        invert_vertex_selection(&mesh, &mut props, h);
        assert_eq!(count_selected_vertices(&mesh, &props, h), 0);
        assert_eq!(selected_vertex_ids(&mesh, &props, h).count(), 0);
    }

    // ---------- 可视化尺寸 / 粗细 / 透明度测试 ----------

    #[test]
    fn vertex_size_basic() {
        let mesh = crate::test_util::build_icosphere(0);
        let mut props = MeshProperties::new();
        let h = add_vertex_sizes(&mut props);
        let v0 = mesh.vertex_ids().next().unwrap();

        // 未设置返回 None
        assert_eq!(vertex_size(&props, h, v0), None);

        set_vertex_size(&mut props, h, v0, 0.05);
        assert_eq!(vertex_size(&props, h, v0), Some(0.05));

        // 批量统一设置
        set_uniform_vertex_size(&mesh, &mut props, h, 0.1);
        for v in mesh.vertex_ids() {
            assert_eq!(vertex_size(&props, h, v), Some(0.1));
        }

        // 清空
        clear_vertex_sizes(&mesh, &mut props, h);
        for v in mesh.vertex_ids() {
            assert_eq!(vertex_size(&props, h, v), None);
        }
        // 类型注册保留
        assert!(props.has_vertex_prop::<VertexSize>());
    }

    #[test]
    fn vertex_size_default_is_zero() {
        assert_eq!(VertexSize::default().0, 0.0);
    }

    #[test]
    fn edge_thickness_syncs_twin_pair() {
        let (mesh, _, heids, _) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_halfedge_thickness(&mut props);

        let he0 = heids[0];
        let twin = mesh
            .get_halfedge(he0)
            .and_then(|h| h.twin)
            .expect("应至少有一条带 twin 的内部边");

        // 设置 he0 粗细，twin 也应同步
        set_edge_thickness(&mesh, &mut props, h, he0, 0.3);
        assert_eq!(props.get_halfedge_prop(h, he0).map(|t| t.0), Some(0.3));
        assert_eq!(props.get_halfedge_prop(h, twin).map(|t| t.0), Some(0.3));

        // edge_thickness 任一半边都能读到
        assert_eq!(edge_thickness(&mesh, &props, h, he0), Some(0.3));
        assert_eq!(edge_thickness(&mesh, &props, h, twin), Some(0.3));

        // 仅设置 twin，he0 也能读到（fallback）
        clear_halfedge_thickness(&mesh, &mut props, h);
        set_halfedge_thickness(&mut props, h, twin, 0.7);
        assert_eq!(edge_thickness(&mesh, &props, h, he0), Some(0.7));

        // 找一个 he 与 twin 都未设置的边，edge_thickness 应返回 None
        let uncolored_he = heids
            .iter()
            .copied()
            .find(|&he| {
                halfedge_thickness(&props, h, he).is_none()
                    && mesh
                        .get_halfedge(he)
                        .and_then(|h| h.twin)
                        .is_none_or(|twin| halfedge_thickness(&props, h, twin).is_none())
            })
            .expect("应存在 he 与 twin 都未设置的半边");
        assert_eq!(edge_thickness(&mesh, &props, h, uncolored_he), None);
    }

    #[test]
    fn edge_thickness_uniform_sets_all() {
        let (mesh, _, _, _) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_halfedge_thickness(&mut props);

        set_uniform_edge_thickness(&mesh, &mut props, h, 0.25);

        // 每条边（twin 对）都应被设置，且值一致
        let mut edges_checked = 0;
        for he in mesh.halfedge_ids() {
            if let Some(t) = edge_thickness(&mesh, &props, h, he) {
                assert!((t - 0.25).abs() < 1e-12);
                if let Some(twin) = mesh.get_halfedge(he).and_then(|h| h.twin) {
                    if he <= twin {
                        edges_checked += 1;
                    }
                } else {
                    edges_checked += 1;
                }
            }
        }
        assert!(edges_checked > 0, "应至少设置一条边");
    }

    #[test]
    fn halfedge_thickness_independent_set() {
        let (mesh, _, heids, _) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_halfedge_thickness(&mut props);

        // set_halfedge_thickness 不同步 twin
        set_halfedge_thickness(&mut props, h, heids[0], 0.5);
        assert_eq!(halfedge_thickness(&props, h, heids[0]), Some(0.5));

        let twin = mesh
            .get_halfedge(heids[0])
            .and_then(|h| h.twin)
            .expect("应存在 twin");
        // twin 不应被设置（无对称写入）
        assert_eq!(halfedge_thickness(&props, h, twin), None);
    }

    #[test]
    fn face_opacity_basic() {
        let (mesh, _, _, fids) = build_two_tri_mesh();
        let mut props = MeshProperties::new();
        let h = add_face_opacity(&mut props);

        assert_eq!(face_opacity(&props, h, fids[0]), None);

        set_face_opacity(&mut props, h, fids[0], 0.5);
        assert_eq!(face_opacity(&props, h, fids[0]), Some(0.5));
        assert_eq!(face_opacity(&props, h, fids[1]), None);

        // clamp 测试：超出 [0, 1] 的输入被截断
        set_face_opacity(&mut props, h, fids[0], -0.5);
        assert_eq!(face_opacity(&props, h, fids[0]), Some(0.0));
        set_face_opacity(&mut props, h, fids[0], 1.5);
        assert_eq!(face_opacity(&props, h, fids[0]), Some(1.0));

        // 批量统一
        set_uniform_face_opacity(&mesh, &mut props, h, 0.7);
        for f in mesh.face_ids() {
            assert_eq!(face_opacity(&props, h, f), Some(0.7));
        }
        // 批量统一也 clamp
        set_uniform_face_opacity(&mesh, &mut props, h, 2.0);
        for f in mesh.face_ids() {
            assert_eq!(face_opacity(&props, h, f), Some(1.0));
        }

        // 清空
        clear_face_opacity(&mesh, &mut props, h);
        for f in mesh.face_ids() {
            assert_eq!(face_opacity(&props, h, f), None);
        }
        assert!(props.has_face_prop::<FaceOpacity>());
    }

    #[test]
    fn face_opacity_default_is_zero() {
        assert_eq!(FaceOpacity::default().0, 0.0);
    }

    #[test]
    fn unregistered_size_thickness_opacity_safe_no_panic() {
        // 未注册属性时调用查询接口不应 panic
        let mesh = crate::test_util::build_icosphere(0);
        let props = MeshProperties::new();

        let hv: PropertyHandle<VertexSize> = PropertyHandle::new();
        let v = mesh.vertex_ids().next().unwrap();
        assert_eq!(vertex_size(&props, hv, v), None);

        let he: PropertyHandle<HalfEdgeThickness> = PropertyHandle::new();
        let he_id = mesh.halfedge_ids().next().unwrap();
        assert_eq!(edge_thickness(&mesh, &props, he, he_id), None);

        let hf: PropertyHandle<FaceOpacity> = PropertyHandle::new();
        let f = mesh.face_ids().next().unwrap();
        assert_eq!(face_opacity(&props, hf, f), None);
    }

    #[test]
    fn empty_mesh_size_thickness_opacity_no_panic() {
        // 空网格上调用尺寸/粗细/透明度接口不应 panic
        let mesh = MeshStorage::new();
        let mut props = MeshProperties::new();

        let hv = add_vertex_sizes(&mut props);
        set_uniform_vertex_size(&mesh, &mut props, hv, 0.1);
        clear_vertex_sizes(&mesh, &mut props, hv);

        let he = add_halfedge_thickness(&mut props);
        set_uniform_edge_thickness(&mesh, &mut props, he, 0.2);
        clear_halfedge_thickness(&mesh, &mut props, he);

        let hf = add_face_opacity(&mut props);
        set_uniform_face_opacity(&mesh, &mut props, hf, 0.5);
        clear_face_opacity(&mesh, &mut props, hf);
    }
}
