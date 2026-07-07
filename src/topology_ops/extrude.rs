//! 面挤出操作：extrude_face, extrude_faces, extrude_region。

use std::collections::{HashMap, HashSet};

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
use crate::traversal::{FaceHalfEdges, VertexAdjacentFaces};

use super::helpers::TopologyError;

// ============================================================
// extrude_face：面挤出
// ============================================================

/// 将三角面 `face` 沿 `offset` 方向挤出，形成棱柱体。
///
/// ## 拓扑修改逻辑
///
/// 设 `F = (v0, v1, v2)` CCW，半边 `h0(v0→v1), h1(v1→v2), h2(v2→v0)`，
/// 对应边界 twin `t0(v1→v0), t1(v2→v1), t2(v0→v2)`（均 `face = None`）。
///
/// ### 限制
/// F 的三条边必须均为边界边（twin 的 `face = None`），否则返回
/// [`TopologyError::Inconsistent`]。
///
/// ### 创建元素
/// - 3 个新顶点 `v0', v1', v2'` = 原位置 + offset
/// - 7 个新面：1 顶面 `F'` + 6 侧面三角形（每条侧边 2 个）
/// - 18 条新半边（顶面 3 + 侧面 15），重用 3 条原边界 twin
///
/// ### 顶面 F'
/// `F' = (v2', v1', v0')` CCW（反向），半边
/// `tp_0(v2'→v1'), tp_1(v1'→v0'), tp_2(v0'→v2')`。
///
/// ### 侧面三角形（边 i: `hi(vi→vi+1)`）
/// - `T1_i = (vi+1, vi, vi')`：重用 `ti(vi+1→vi)` + 新建 `a_i(vi→vi')` + 新建 `b_i(vi'→vi+1)`
/// - `T2_i = (vi+1, vi', vi+1')`：新建 `c_i(vi+1→vi')` + 新建 `d_i(vi'→vi+1')` + 新建 `e_i(vi+1'→vi+1)`
///
/// ### twin 对应（12 对）
/// - `ti ↔ hi`（已存在，保留）
/// - `a_i ↔ e_{i-1}`（垂直边，相邻侧面共享）
/// - `b_i ↔ c_i`（侧面内对角线）
/// - `d_i ↔ tp_{map(i)}`（顶面与侧面共享边）：
///   `d_0↔tp_1`, `d_1↔tp_0`, `d_2↔tp_2`
///
/// # 返回
/// 新创建的面 ID 列表（顺序：`[F', T1_0, T2_0, T1_1, T2_1, T1_2, T2_2]`）。
///
/// # 退化检查
/// - offset 为零向量（长度 < 1e-12）返回 [`TopologyError::DegenerateTriangle`]
/// - 任一侧面三角形面积 < 1e-12（offset 与边平行）返回 [`TopologyError::DegenerateTriangle`]
///
/// # 校验
/// 操作完成后调用 [`crate::validate::validate_topology`] 完整校验网格。
pub fn extrude_face(
    mesh: &mut MeshStorage,
    face: FaceId,
    offset: [f64; 3],
) -> Result<Vec<FaceId>, TopologyError> {
    let data = validate_extrude_face_input(mesh, face, offset)?;
    let elements = create_extrude_face_elements(mesh, &data, offset);
    setup_extrude_face_top(mesh, &elements);
    setup_extrude_face_sides(mesh, &data, &elements);
    update_extrude_face_vertex_outgoing(mesh, &elements);

    let errors = crate::validate::validate_topology(mesh);
    if !errors.is_empty() {
        return Err(TopologyError::Inconsistent(format!(
            "extrude_face 后校验失败：{:?}",
            errors
        )));
    }

    Ok(vec![
        elements.f_top,
        elements.f_t1[0],
        elements.f_t2[0],
        elements.f_t1[1],
        elements.f_t2[1],
        elements.f_t1[2],
        elements.f_t2[2],
    ])
}

// ---- extrude_face 辅助 ----

/// 面挤出收集的拓扑与几何数据。
struct ExtrudeFaceData {
    h_arr: [HalfEdgeId; 3],
    /// 边界 twin 半边（`face = None`）。
    t: [HalfEdgeId; 3],
    v: [VertexId; 3],
    pos: [[f64; 3]; 3],
}

/// 面挤出创建的新元素（顶点、半边、面）。
struct ExtrudeFaceElements {
    vp: [VertexId; 3],
    tp: [HalfEdgeId; 3],
    a: [HalfEdgeId; 3],
    b: [HalfEdgeId; 3],
    c: [HalfEdgeId; 3],
    d: [HalfEdgeId; 3],
    e: [HalfEdgeId; 3],
    f_top: FaceId,
    f_t1: [FaceId; 3],
    f_t2: [FaceId; 3],
}

/// 校验面挤出输入：offset 非零、面为三角面、所有边为边界边、侧面不退化。
fn validate_extrude_face_input(
    mesh: &MeshStorage,
    face: FaceId,
    offset: [f64; 3],
) -> Result<ExtrudeFaceData, TopologyError> {
    // offset 非零
    let offset_len = (offset[0].powi(2) + offset[1].powi(2) + offset[2].powi(2)).sqrt();
    if offset_len < 1e-12 {
        return Err(TopologyError::DegenerateTriangle);
    }

    // 面存在且为三角面
    if !mesh.contains_face(face) {
        return Err(TopologyError::Inconsistent(format!("面 {:?} 不存在", face)));
    }
    let hs: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    if hs.len() != 3 {
        return Err(TopologyError::Inconsistent(format!(
            "面 {:?} 边界环长度={}, 非三角面",
            face,
            hs.len()
        )));
    }
    let h_arr: [HalfEdgeId; 3] = [hs[0], hs[1], hs[2]];

    let h0d = mesh
        .get_halfedge(h_arr[0])
        .ok_or(TopologyError::InvalidHalfEdge(h_arr[0]))?
        .clone();
    let h1d = mesh
        .get_halfedge(h_arr[1])
        .ok_or(TopologyError::InvalidHalfEdge(h_arr[1]))?
        .clone();
    let h2d = mesh
        .get_halfedge(h_arr[2])
        .ok_or(TopologyError::InvalidHalfEdge(h_arr[2]))?
        .clone();

    // h0: v0→v1 (vertex=v1), h1: v1→v2 (vertex=v2), h2: v2→v0 (vertex=v0)
    let v: [VertexId; 3] = [h2d.vertex, h0d.vertex, h1d.vertex];

    let pos: [[f64; 3]; 3] = {
        let mut arr = [[0.0f64; 3]; 3];
        for i in 0..3 {
            arr[i] = mesh
                .get_vertex(v[i])
                .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v[i])))?
                .position;
        }
        arr
    };

    // 所有边为边界边
    let t: [HalfEdgeId; 3] = [
        h0d.twin.ok_or(TopologyError::NoTwin(h_arr[0]))?,
        h1d.twin.ok_or(TopologyError::NoTwin(h_arr[1]))?,
        h2d.twin.ok_or(TopologyError::NoTwin(h_arr[2]))?,
    ];
    for &item in &t {
        let td = mesh
            .get_halfedge(item)
            .ok_or(TopologyError::InvalidHalfEdge(item))?;
        if td.face.is_some() {
            return Err(TopologyError::Inconsistent(
                "extrude_face 仅支持所有边为边界边的面".into(),
            ));
        }
    }

    // 退化侧面检查：用 Shewchuk 鲁棒谓词精确判定
    for i in 0..3 {
        let a = pos[(i + 1) % 3];
        let b = pos[i];
        let c = [
            pos[i][0] + offset[0],
            pos[i][1] + offset[1],
            pos[i][2] + offset[2],
        ];
        if crate::predicates::is_triangle_degenerate_3d(a, b, c) {
            return Err(TopologyError::DegenerateTriangle);
        }
    }

    Ok(ExtrudeFaceData { h_arr, t, v, pos })
}

/// 创建面挤出的新顶点（3 个）、新半边（18 条）和新面（7 个）。
fn create_extrude_face_elements(
    mesh: &mut MeshStorage,
    data: &ExtrudeFaceData,
    offset: [f64; 3],
) -> ExtrudeFaceElements {
    let v = &data.v;
    let pos = &data.pos;

    // 新顶点
    let vp: [VertexId; 3] = [
        mesh.add_vertex(Vertex::new([
            pos[0][0] + offset[0],
            pos[0][1] + offset[1],
            pos[0][2] + offset[2],
        ])),
        mesh.add_vertex(Vertex::new([
            pos[1][0] + offset[0],
            pos[1][1] + offset[1],
            pos[1][2] + offset[2],
        ])),
        mesh.add_vertex(Vertex::new([
            pos[2][0] + offset[0],
            pos[2][1] + offset[1],
            pos[2][2] + offset[2],
        ])),
    ];

    // 顶面 3 条: tp[0]=v2'→v1', tp[1]=v1'→v0', tp[2]=v0'→v2'
    let tp: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[1])),
        mesh.add_halfedge(HalfEdge::new(vp[0])),
        mesh.add_halfedge(HalfEdge::new(vp[2])),
    ];

    // 侧面 15 条
    let a: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[0])),
        mesh.add_halfedge(HalfEdge::new(vp[1])),
        mesh.add_halfedge(HalfEdge::new(vp[2])),
    ];
    let b: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(v[1])),
        mesh.add_halfedge(HalfEdge::new(v[2])),
        mesh.add_halfedge(HalfEdge::new(v[0])),
    ];
    let c: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[0])),
        mesh.add_halfedge(HalfEdge::new(vp[1])),
        mesh.add_halfedge(HalfEdge::new(vp[2])),
    ];
    let d: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[1])),
        mesh.add_halfedge(HalfEdge::new(vp[2])),
        mesh.add_halfedge(HalfEdge::new(vp[0])),
    ];
    let e: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(v[1])),
        mesh.add_halfedge(HalfEdge::new(v[2])),
        mesh.add_halfedge(HalfEdge::new(v[0])),
    ];

    // 新面
    let f_top = mesh.add_face(Face::new());
    let f_t1: [FaceId; 3] = [
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
    ];
    let f_t2: [FaceId; 3] = [
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
    ];

    ExtrudeFaceElements {
        vp,
        tp,
        a,
        b,
        c,
        d,
        e,
        f_top,
        f_t1,
        f_t2,
    }
}

/// 设置顶面 F' 半边的 twin/next/prev/face 环。
fn setup_extrude_face_top(mesh: &mut MeshStorage, el: &ExtrudeFaceElements) {
    // twin 映射: tp[0]↔d[1], tp[1]↔d[0], tp[2]↔d[2]
    let tp_twin: [HalfEdgeId; 3] = [el.d[1], el.d[0], el.d[2]];
    // tp[0] → tp[1] → tp[2] → tp[0]
    for (i, &twin) in tp_twin.iter().enumerate() {
        let he = mesh.get_halfedge_mut(el.tp[i]).expect("tp[i] just created");
        he.twin = Some(twin);
        he.next = Some(el.tp[(i + 1) % 3]);
        he.prev = Some(el.tp[(i + 2) % 3]);
        he.face = Some(el.f_top);
    }
    mesh.get_face_mut(el.f_top)
        .expect("f_top just created")
        .halfedge = Some(el.tp[0]);
}

/// 设置侧面 T1_i 和 T2_i 的 twin/next/prev/face 环。
fn setup_extrude_face_sides(
    mesh: &mut MeshStorage,
    data: &ExtrudeFaceData,
    el: &ExtrudeFaceElements,
) {
    // d[i] ↔ d_twin[i]: d[0]↔tp[1], d[1]↔tp[0], d[2]↔tp[2]
    let d_twin: [HalfEdgeId; 3] = [el.tp[1], el.tp[0], el.tp[2]];

    for (i, &dt) in d_twin.iter().enumerate() {
        // T1_i = (vi+1, vi, vi'): t[i] → a[i] → b[i] → t[i]
        {
            let he = mesh
                .get_halfedge_mut(data.t[i])
                .expect("t[i] validated earlier");
            he.twin = Some(data.h_arr[i]); // 保持原 twin
            he.next = Some(el.a[i]);
            he.prev = Some(el.b[i]);
            he.face = Some(el.f_t1[i]);

            let he = mesh.get_halfedge_mut(el.a[i]).expect("a[i] just created");
            he.twin = Some(el.e[(i + 2) % 3]); // a_i ↔ e_{i-1}
            he.next = Some(el.b[i]);
            he.prev = Some(data.t[i]);
            he.face = Some(el.f_t1[i]);

            let he = mesh.get_halfedge_mut(el.b[i]).expect("b[i] just created");
            he.twin = Some(el.c[i]); // b_i ↔ c_i
            he.next = Some(data.t[i]);
            he.prev = Some(el.a[i]);
            he.face = Some(el.f_t1[i]);
        }
        mesh.get_face_mut(el.f_t1[i])
            .expect("f_t1[i] just created")
            .halfedge = Some(data.t[i]);

        // T2_i = (vi+1, vi', vi+1'): c[i] → d[i] → e[i] → c[i]
        {
            let he = mesh.get_halfedge_mut(el.c[i]).expect("c[i] just created");
            he.twin = Some(el.b[i]); // c_i ↔ b_i
            he.next = Some(el.d[i]);
            he.prev = Some(el.e[i]);
            he.face = Some(el.f_t2[i]);

            let he = mesh.get_halfedge_mut(el.d[i]).expect("d[i] just created");
            he.twin = Some(dt); // d_i ↔ tp_?
            he.next = Some(el.e[i]);
            he.prev = Some(el.c[i]);
            he.face = Some(el.f_t2[i]);

            let he = mesh.get_halfedge_mut(el.e[i]).expect("e[i] just created");
            he.twin = Some(el.a[(i + 1) % 3]); // e_i ↔ a_{i+1}
            he.next = Some(el.c[i]);
            he.prev = Some(el.d[i]);
            he.face = Some(el.f_t2[i]);
        }
        mesh.get_face_mut(el.f_t2[i])
            .expect("f_t2[i] just created")
            .halfedge = Some(el.c[i]);
    }
}

/// 更新新顶点 `v_i'` 的 outgoing 半边入口为 `d[i]`。
fn update_extrude_face_vertex_outgoing(mesh: &mut MeshStorage, el: &ExtrudeFaceElements) {
    for i in 0..3 {
        mesh.get_vertex_mut(el.vp[i])
            .expect("vp[i] just created")
            .halfedge = Some(el.d[i]);
    }
}

/// 批量挤出多个面。
///
/// 逐个调用 [`extrude_face`]。如果两个面共享边，先挤出的面会使共享边变为内部边，
/// 后续挤出会失败并返回 [`TopologyError::Inconsistent`]（此时网格已部分修改）。
///
/// # 返回
/// 所有新创建的面 ID 列表（每个面 7 个，顺序同 [`extrude_face`]）。
pub fn extrude_faces(
    mesh: &mut MeshStorage,
    faces: &[FaceId],
    offset: [f64; 3],
) -> Result<Vec<FaceId>, TopologyError> {
    let mut all_new = Vec::new();
    for &f in faces {
        let new = extrude_face(mesh, f, offset)?;
        all_new.extend(new);
    }
    Ok(all_new)
}

// ============================================================
// extrude_region：区域挤出
// ============================================================

/// 区域挤出：一次性挤出多个相邻面，自动处理内部边。
///
/// 模仿 Blender/Maya 中 "Extrude Region" 的行为：选中区域内所有面共同移动
/// `offset`，区域内部边不产生侧面，仅在区域边界生成侧面四边形（三角化）。
///
/// ## 算法概览
///
/// 1. **顶点分类**：对每个区域顶点 `v`，若所有邻接面均在区域内 → 内部顶点
///    （原地平移 `pos += offset`，`vert_map[v] = v`）；否则 → 边界顶点
///    （创建新顶点 `v' = pos + offset`，`vert_map[v] = v'`）。
/// 2. **半边分类**：对每条区域半边 `h`，若 `h.twin.face` 也在区域内 → 内部半边
///    （直接更新 `h.vertex = vert_map[v_t]`，twin 同理）；否则 → 边界半边
///    （创建 `h_top` 替换 `h` 在面环中的位置，`h` 释放供侧面使用）。
/// 3. **侧面创建**：每条边界半边 `h(v_o→v_t)` 生成四边形 `(v_o, v_t, v_t', v_o')`，
///    三角化为 `T1=(v_o,v_t,v_t')` 与 `T2=(v_o,v_t',v_o')`。
/// 4. **垂直半边**：每个边界顶点 `v` 创建一对 `(up: v→v', down: v'→v)`，
///    相邻侧面共享。
///
/// ## 边界情况
/// - `offset` 长度 < 1e-12 → [`TopologyError::DegenerateTriangle`]
/// - 任一侧面面积 < 1e-12 → [`TopologyError::DegenerateTriangle`]
/// - 区域包含整个网格 → 所有顶点为内部顶点，仅平移，无侧面
///
/// ## 返回
/// 新创建的面 ID 列表（仅侧面三角形，顺序为每条边界边的 `[T1, T2]`）。
/// 原区域面被复用为顶面（朝向不变），不在返回列表中。
pub fn extrude_region(
    mesh: &mut MeshStorage,
    faces: &[FaceId],
    offset: [f64; 3],
) -> Result<Vec<FaceId>, TopologyError> {
    let region_set = validate_extrude_region_input(mesh, faces, offset)?;
    let topo = collect_region_topology(mesh, faces, &region_set)?;
    let vert_map = classify_region_vertices(mesh, &topo.verts, &region_set, offset)?;
    let classification = classify_region_halfedges(mesh, &topo.face_hes, &region_set, offset)?;
    let maps = setup_region_halfedges(
        mesh,
        &classification.boundary_hes,
        &classification.interior_hes,
        &vert_map,
    );
    rebuild_region_face_rings(mesh, &topo.face_hes, &maps.top_he_map);
    let new_faces = create_region_side_faces(mesh, &classification.boundary_hes, &vert_map, &maps);
    fix_region_vertex_outgoing(mesh, &classification.boundary_hes, &maps.vert_he, &vert_map);

    let errors = crate::validate::validate_topology(mesh);
    if !errors.is_empty() {
        return Err(TopologyError::Inconsistent(format!(
            "extrude_region 后校验失败：{:?}",
            errors
        )));
    }

    Ok(new_faces)
}

// ---- extrude_region 辅助 ----

/// 区域半边分类结果。
struct RegionHalfedgeClassification {
    /// 边界半边列表 `(h, v_o, v_t, F)`。
    boundary_hes: Vec<(HalfEdgeId, VertexId, VertexId, FaceId)>,
    /// 内部半边集合。
    interior_hes: HashSet<HalfEdgeId>,
}

/// 区域挤出创建的半边映射。
struct RegionHalfedgeMaps {
    /// 边界顶点 → `(up: v→v', down: v'→v)`。
    vert_he: HashMap<VertexId, (HalfEdgeId, HalfEdgeId)>,
    /// 原边界半边 h → 新顶半边 h_top。
    top_he_map: HashMap<HalfEdgeId, HalfEdgeId>,
    /// 原边界半边 h → s_top（h_top 的 twin）。
    s_top_map: HashMap<HalfEdgeId, HalfEdgeId>,
}

/// 校验区域挤出输入：offset 非零、面列表非空、无重复、面均存在。
fn validate_extrude_region_input(
    mesh: &MeshStorage,
    faces: &[FaceId],
    offset: [f64; 3],
) -> Result<HashSet<FaceId>, TopologyError> {
    let offset_len = (offset[0].powi(2) + offset[1].powi(2) + offset[2].powi(2)).sqrt();
    if offset_len < 1e-12 {
        return Err(TopologyError::DegenerateTriangle);
    }

    if faces.is_empty() {
        return Err(TopologyError::Inconsistent("挤出区域为空".into()));
    }
    let region_set: HashSet<FaceId> = faces.iter().copied().collect();
    if region_set.len() != faces.len() {
        return Err(TopologyError::Inconsistent("挤出区域包含重复面".into()));
    }
    for &f in faces {
        if !mesh.contains_face(f) {
            return Err(TopologyError::Inconsistent(format!("面 {:?} 不存在", f)));
        }
    }
    Ok(region_set)
}

/// 区域拓扑信息：每个面的三条半边及所有顶点集合。
struct RegionTopology {
    face_hes: Vec<(FaceId, [HalfEdgeId; 3])>,
    verts: HashSet<VertexId>,
}

/// 收集区域中每个面的三条半边及所有顶点。
fn collect_region_topology(
    mesh: &MeshStorage,
    faces: &[FaceId],
    _region_set: &HashSet<FaceId>,
) -> Result<RegionTopology, TopologyError> {
    let mut face_hes: Vec<(FaceId, [HalfEdgeId; 3])> = Vec::new();
    let mut verts: HashSet<VertexId> = HashSet::new();
    for &f in faces {
        let hes: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, f).collect();
        if hes.len() != 3 {
            return Err(TopologyError::Inconsistent(format!(
                "面 {:?} 边界环长度={}, 非三角面",
                f,
                hes.len()
            )));
        }
        for &h in &hes {
            let v_t = mesh
                .get_halfedge(h)
                .ok_or(TopologyError::InvalidHalfEdge(h))?
                .vertex;
            verts.insert(v_t);
        }
        face_hes.push((f, [hes[0], hes[1], hes[2]]));
    }
    Ok(RegionTopology { face_hes, verts })
}

/// 顶点分类：内部顶点原地平移（`vert_map[v] = v`），边界顶点创建新顶点（`vert_map[v] = v'`）。
fn classify_region_vertices(
    mesh: &mut MeshStorage,
    region_verts_set: &HashSet<VertexId>,
    region_set: &HashSet<FaceId>,
    offset: [f64; 3],
) -> Result<HashMap<VertexId, VertexId>, TopologyError> {
    let mut vert_map: HashMap<VertexId, VertexId> = HashMap::new();
    for &v in region_verts_set {
        let adj_faces: Vec<FaceId> = VertexAdjacentFaces::new(mesh, v).collect();
        let all_in_region =
            !adj_faces.is_empty() && adj_faces.iter().all(|f| region_set.contains(f));
        if all_in_region {
            // 内部顶点：原地平移
            let pos = mesh
                .get_vertex(v)
                .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v)))?
                .position;
            mesh.get_vertex_mut(v)
                .expect("vertex v validated above")
                .position = [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]];
            vert_map.insert(v, v);
        } else {
            // 边界顶点：创建新顶点
            let pos = mesh
                .get_vertex(v)
                .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v)))?
                .position;
            let v_new = mesh.add_vertex(Vertex::new([
                pos[0] + offset[0],
                pos[1] + offset[1],
                pos[2] + offset[2],
            ]));
            vert_map.insert(v, v_new);
        }
    }
    Ok(vert_map)
}

/// 半边分类：将区域半边分为边界半边和内部半边，并检查侧面退化。
fn classify_region_halfedges(
    mesh: &MeshStorage,
    region_face_hes: &[(FaceId, [HalfEdgeId; 3])],
    region_set: &HashSet<FaceId>,
    offset: [f64; 3],
) -> Result<RegionHalfedgeClassification, TopologyError> {
    let mut boundary_hes: Vec<(HalfEdgeId, VertexId, VertexId, FaceId)> = Vec::new();
    let mut interior_hes: HashSet<HalfEdgeId> = HashSet::new();

    for &(f, hes) in region_face_hes {
        for &h in &hes {
            let h_data = mesh
                .get_halfedge(h)
                .ok_or(TopologyError::InvalidHalfEdge(h))?;
            let twin_id = h_data
                .twin
                .ok_or_else(|| TopologyError::Inconsistent(format!("半边 {:?} 无 twin", h)))?;
            let twin_data = mesh
                .get_halfedge(twin_id)
                .ok_or(TopologyError::InvalidHalfEdge(twin_id))?;
            let v_t = h_data.vertex;
            let v_o = twin_data.vertex;
            match twin_data.face {
                Some(twin_face) if region_set.contains(&twin_face) => {
                    interior_hes.insert(h);
                }
                _ => {
                    boundary_hes.push((h, v_o, v_t, f));
                }
            }
        }
    }

    // 退化侧面检查：用 Shewchuk 鲁棒谓词精确判定
    for &(_h, v_o, v_t, _f) in &boundary_hes {
        let pos_o = mesh
            .get_vertex(v_o)
            .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v_o)))?
            .position;
        let pos_t = mesh
            .get_vertex(v_t)
            .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v_t)))?
            .position;
        let pos_o_up = [
            pos_o[0] + offset[0],
            pos_o[1] + offset[1],
            pos_o[2] + offset[2],
        ];
        if crate::predicates::is_triangle_degenerate_3d(pos_o, pos_t, pos_o_up) {
            return Err(TopologyError::DegenerateTriangle);
        }
    }

    Ok(RegionHalfedgeClassification {
        boundary_hes,
        interior_hes,
    })
}

/// 创建垂直半边对、更新内部半边 vertex、创建边界半边的 h_top/s_top。
fn setup_region_halfedges(
    mesh: &mut MeshStorage,
    boundary_hes: &[(HalfEdgeId, VertexId, VertexId, FaceId)],
    interior_hes: &HashSet<HalfEdgeId>,
    vert_map: &HashMap<VertexId, VertexId>,
) -> RegionHalfedgeMaps {
    // 为每个边界顶点创建垂直半边对 (up: v→v', down: v'→v)
    let mut vert_he: HashMap<VertexId, (HalfEdgeId, HalfEdgeId)> = HashMap::new();
    for &(_h, v_o, v_t, _f) in boundary_hes {
        for &v in &[v_o, v_t] {
            vert_he.entry(v).or_insert_with(|| {
                let v_new = vert_map[&v];
                let up = mesh.add_halfedge(HalfEdge::new(v_new)); // v → v'
                let down = mesh.add_halfedge(HalfEdge::new(v)); // v' → v
                mesh.get_halfedge_mut(up).expect("up just created").twin = Some(down);
                mesh.get_halfedge_mut(down).expect("down just created").twin = Some(up);
                (up, down)
            });
        }
    }

    // 处理内部半边：更新 vertex
    for &h in interior_hes {
        let v_t = mesh
            .get_halfedge(h)
            .expect("h from interior_hes, validated")
            .vertex;
        let new_v = vert_map[&v_t];
        mesh.get_halfedge_mut(h)
            .expect("h from interior_hes, validated")
            .vertex = new_v;
    }

    // 处理边界半边：创建 h_top 和 s_top
    let mut top_he_map: HashMap<HalfEdgeId, HalfEdgeId> = HashMap::new();
    let mut s_top_map: HashMap<HalfEdgeId, HalfEdgeId> = HashMap::new();
    for &(h, v_o, v_t, _f) in boundary_hes {
        let v_t_new = vert_map[&v_t];
        let v_o_new = vert_map[&v_o];
        let h_top = mesh.add_halfedge(HalfEdge::new(v_t_new)); // v_o' → v_t'
        let s_top = mesh.add_halfedge(HalfEdge::new(v_o_new)); // v_t' → v_o'
        mesh.get_halfedge_mut(h_top)
            .expect("h_top just created")
            .twin = Some(s_top);
        mesh.get_halfedge_mut(s_top)
            .expect("s_top just created")
            .twin = Some(h_top);
        top_he_map.insert(h, h_top);
        s_top_map.insert(h, s_top);
    }

    RegionHalfedgeMaps {
        vert_he,
        top_he_map,
        s_top_map,
    }
}

/// 重建区域面的半边环：边界半边替换为 h_top，内部半边保留。
fn rebuild_region_face_rings(
    mesh: &mut MeshStorage,
    region_face_hes: &[(FaceId, [HalfEdgeId; 3])],
    top_he_map: &HashMap<HalfEdgeId, HalfEdgeId>,
) {
    for &(f, hes) in region_face_hes {
        let [h0, h1, h2] = hes;
        let r0 = *top_he_map.get(&h0).unwrap_or(&h0);
        let r1 = *top_he_map.get(&h1).unwrap_or(&h1);
        let r2 = *top_he_map.get(&h2).unwrap_or(&h2);

        let r0_he = mesh
            .get_halfedge_mut(r0)
            .expect("r0 from top_he_map or original, validated");
        r0_he.next = Some(r1);
        r0_he.prev = Some(r2);
        r0_he.face = Some(f);

        let r1_he = mesh
            .get_halfedge_mut(r1)
            .expect("r1 from top_he_map or original, validated");
        r1_he.next = Some(r2);
        r1_he.prev = Some(r0);
        r1_he.face = Some(f);

        let r2_he = mesh
            .get_halfedge_mut(r2)
            .expect("r2 from top_he_map or original, validated");
        r2_he.next = Some(r0);
        r2_he.prev = Some(r1);
        r2_he.face = Some(f);

        mesh.get_face_mut(f).expect("f validated earlier").halfedge = Some(r0);

        // 清除原边界半边的 next/prev/face（供侧面复用）
        for &h in &hes {
            if top_he_map.contains_key(&h) {
                let h_he = mesh
                    .get_halfedge_mut(h)
                    .expect("h from region_face_hes, validated");
                h_he.next = None;
                h_he.prev = None;
                h_he.face = None;
            }
        }
    }
}

/// 为每条边界半边创建侧面三角形 T1 和 T2。
fn create_region_side_faces(
    mesh: &mut MeshStorage,
    boundary_hes: &[(HalfEdgeId, VertexId, VertexId, FaceId)],
    vert_map: &HashMap<VertexId, VertexId>,
    maps: &RegionHalfedgeMaps,
) -> Vec<FaceId> {
    let mut new_faces = Vec::new();
    for &(h, v_o, v_t, _f) in boundary_hes {
        let v_t_new = vert_map[&v_t];
        let s_top = maps.s_top_map[&h];
        let (up_t, _down_t) = maps.vert_he[&v_t]; // v_t → v_t_new
        let (_up_o, down_o) = maps.vert_he[&v_o]; // v_o_new → v_o

        // 对角线
        let diag = mesh.add_halfedge(HalfEdge::new(v_t_new)); // v_o → v_t_new
        let diag_rev = mesh.add_halfedge(HalfEdge::new(v_o)); // v_t_new → v_o
        mesh.get_halfedge_mut(diag).expect("diag just created").twin = Some(diag_rev);
        mesh.get_halfedge_mut(diag_rev)
            .expect("diag_rev just created")
            .twin = Some(diag);

        // T1 = (v_o, v_t, v_t_new): h → up_t → diag_rev
        let f_t1 = mesh.add_face(Face::new());
        {
            let he = mesh
                .get_halfedge_mut(h)
                .expect("h from boundary_hes, validated");
            he.next = Some(up_t);
            he.prev = Some(diag_rev);
            he.face = Some(f_t1);

            let he = mesh
                .get_halfedge_mut(up_t)
                .expect("up_t from vert_he, just created");
            he.next = Some(diag_rev);
            he.prev = Some(h);
            he.face = Some(f_t1);

            let he = mesh
                .get_halfedge_mut(diag_rev)
                .expect("diag_rev just created");
            he.next = Some(h);
            he.prev = Some(up_t);
            he.face = Some(f_t1);
        }
        mesh.get_face_mut(f_t1).expect("f_t1 just created").halfedge = Some(h);

        // T2 = (v_o, v_t_new, v_o_new): diag → s_top → down_o
        let f_t2 = mesh.add_face(Face::new());
        {
            let he = mesh.get_halfedge_mut(diag).expect("diag just created");
            he.next = Some(s_top);
            he.prev = Some(down_o);
            he.face = Some(f_t2);

            let he = mesh.get_halfedge_mut(s_top).expect("s_top just created");
            he.next = Some(down_o);
            he.prev = Some(diag);
            he.face = Some(f_t2);

            let he = mesh
                .get_halfedge_mut(down_o)
                .expect("down_o from vert_he, just created");
            he.next = Some(diag);
            he.prev = Some(s_top);
            he.face = Some(f_t2);
        }
        mesh.get_face_mut(f_t2).expect("f_t2 just created").halfedge = Some(diag);

        new_faces.push(f_t1);
        new_faces.push(f_t2);
    }
    new_faces
}

/// 修复边界顶点和新顶点的 outgoing 半边入口。
fn fix_region_vertex_outgoing(
    mesh: &mut MeshStorage,
    boundary_hes: &[(HalfEdgeId, VertexId, VertexId, FaceId)],
    vert_he: &HashMap<VertexId, (HalfEdgeId, HalfEdgeId)>,
    vert_map: &HashMap<VertexId, VertexId>,
) {
    for &(_h, v_o, v_t, _f) in boundary_hes {
        for &v in &[v_o, v_t] {
            let (up, down) = vert_he[&v];
            // 原顶点 v 的 outgoing 指向 up（v→v'）
            mesh.get_vertex_mut(v)
                .expect("v from boundary_hes, validated")
                .halfedge = Some(up);
            // 新顶点 v' 的 outgoing 指向 down（v'→v）
            let v_new = vert_map[&v];
            mesh.get_vertex_mut(v_new)
                .expect("v_new from vert_map, validated")
                .halfedge = Some(down);
        }
    }
}
