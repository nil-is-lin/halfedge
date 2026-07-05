//! 内置属性模块：normal / uv / color 作为一等公民属性。
//!
//! 与 [`crate::property`] 的泛型属性系统互补，本模块定义强类型 Newtype
//! 包装，让顶点法向 / 顶点 UV / 顶点颜色 / 面法向成为内置属性：
//!
//! - [`VertexNormal`]：`[f64; 3]`，与 `geometry::vertex_normal` 一致；
//! - [`VertexColor`]：`[f64; 3]`，RGB 通道 ∈ [0, 1]；
//! - [`VertexUv`]：`[f64; 2]`，纹理坐标；
//! - [`FaceNormal`]：`[f64; 3]`，与 `geometry::face_normal` 一致。
//!
//! 通过 [`MeshProperties`] 注册后，可通过句柄 `PropertyHandle<T>` 读写。
//! [`io`] 模块的 OBJ/PLY 路径会感知这些属性并双向同步。

use crate::ids::VertexId;
use crate::property::{MeshProperties, PropertyHandle};

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

    let mesh = build_mesh_from_polygons(&vertices, &faces);
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
}
