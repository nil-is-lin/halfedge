//! 边/面编辑操作：flip_edge, split_edge, collapse_edge, split_face。

use std::collections::HashSet;

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
use crate::traversal::{FaceHalfEdges, VertexAdjacentVerts, VertexRing};

use super::helpers::TopologyError;
use super::validate::validate_mesh;

// ============================================================
// split_edge：边分裂
// ============================================================

/// 分裂半边 `he` 所在的边，在两端顶点中点处插入新顶点 `M`，重构相邻面的拓扑。
///
/// ## 拓扑修改逻辑
///
/// 设 `h = he`（若 `h.face = None` 但 `twin.face = Some`，自动改为操作 `twin`），
/// `A = h.twin.vertex`（origin），`B = h.vertex`（tip），`F1 = h.face`。
///
/// `F1 = (A→B→C→A)`，其中 `n1 = h.next (B→C)`，`p1 = h.prev (C→A)`。
///
/// 若 `twin.face = Some(F2)`（内部边），`F2 = (B→A→D→B)`，
/// `n2 = twin.next (A→D)`，`p2 = twin.prev (D→B)`。
///
/// ### 内部边（两侧均有面）
/// 删除 `h, twin`。新建 8 条半边与 2 个面：
/// - `F1` 重用为 `(A→M→C→A)`，新建 `F_new1 = (M→B→C→M)`；
/// - `F2` 重用为 `(B→M→D→B)`，新建 `F_new2 = (M→A→D→M)`；
/// - 复用 `n1, p1, n2, p2`，仅修改其 `next/prev/face`。
///
/// ### 边界边（仅一侧有面）
/// 删除 `h, twin`。新建 6 条半边与 1 个面：
/// - `F1` 重用为 `(A→M→C→A)`，新建 `F_new1 = (M→B→C→M)`；
/// - `twin (B→A, 边界)` 分裂为 `b_to_m (B→M, 边界)` + `m_to_a (M→A, 边界)`。
///
/// # 返回
/// 新顶点 `M` 的 ID。
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, split_edge};
///
/// let verts = vec![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
/// ];
/// let faces = vec![[0u32, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
/// let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
///
/// let he = mesh.halfedge_ids().next().unwrap();
/// assert!(split_edge(&mut mesh, he).is_ok());
/// ```
pub fn split_edge(mesh: &mut MeshStorage, he: HalfEdgeId) -> Result<VertexId, TopologyError> {
    let data = collect_split_edge_data(mesh, he)?;
    let new_he = create_split_elements(mesh, &data)?;
    reconnect_split_f1_side(mesh, &data, &new_he);
    reconnect_split_twin_side(mesh, &data, &new_he);
    finalize_split_edge(mesh, &data, &new_he);
    Ok(new_he.m)
}

// ---- split_edge 辅助 ----

/// 边分裂收集的拓扑数据。
struct SplitEdgeData {
    h_id: HalfEdgeId,
    twin_id: HalfEdgeId,
    a: VertexId,
    b: VertexId,
    c: VertexId,
    f1: FaceId,
    n1: HalfEdgeId,
    p1: HalfEdgeId,
    has_f2: bool,
    f2: Option<FaceId>,
    n2: Option<HalfEdgeId>,
    p2: Option<HalfEdgeId>,
    d: Option<VertexId>,
}

/// 边分裂创建的新半边。
struct SplitHalfedges {
    m: VertexId,
    a_to_m: HalfEdgeId,
    m_to_a: HalfEdgeId,
    m_to_b: HalfEdgeId,
    b_to_m: HalfEdgeId,
    m_to_c: HalfEdgeId,
    c_to_m: HalfEdgeId,
    m_to_d: Option<HalfEdgeId>,
    d_to_m: Option<HalfEdgeId>,
}

/// 校验半边并收集边分裂所需的拓扑数据。
///
/// 若传入的半边为边界半边（`face = None`），自动切换到其 twin 进行操作。
fn collect_split_edge_data(
    mesh: &MeshStorage,
    he: HalfEdgeId,
) -> Result<SplitEdgeData, TopologyError> {
    let h_data = mesh
        .get_halfedge(he)
        .ok_or(TopologyError::InvalidHalfEdge(he))?
        .clone();

    // 若 h 是边界半边，改操作 twin（确保 h.face = Some）
    let (h_id, h) = if h_data.face.is_none() {
        let twin_id = h_data.twin.ok_or(TopologyError::NoTwin(he))?;
        let twin_data = mesh
            .get_halfedge(twin_id)
            .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
            .clone();
        if twin_data.face.is_none() {
            return Err(TopologyError::NoFace(he));
        }
        (twin_id, twin_data)
    } else {
        (he, h_data)
    };

    let twin_id = h.twin.ok_or(TopologyError::NoTwin(h_id))?;
    let twin = mesh
        .get_halfedge(twin_id)
        .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
        .clone();

    let a = twin.vertex; // origin of h
    let b = h.vertex; // tip of h
    let f1 = h.face.ok_or(TopologyError::NoFace(h_id))?;
    let n1 = h
        .next
        .ok_or_else(|| TopologyError::Inconsistent("h.next 为 None".into()))?;
    let p1 = h
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("h.prev 为 None".into()))?;
    let c = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .vertex;

    let has_f2 = twin.face.is_some();
    let (f2, n2, p2, d) = if has_f2 {
        let f2 = twin.face.expect("has_f2 guarantees twin.face is Some");
        let n2 = twin
            .next
            .ok_or_else(|| TopologyError::Inconsistent("twin.next 为 None".into()))?;
        let p2 = twin
            .prev
            .ok_or_else(|| TopologyError::Inconsistent("twin.prev 为 None".into()))?;
        let d = mesh
            .get_halfedge(n2)
            .ok_or(TopologyError::InvalidHalfEdge(n2))?
            .vertex;
        (Some(f2), Some(n2), Some(p2), Some(d))
    } else {
        (None, None, None, None)
    };

    Ok(SplitEdgeData {
        h_id,
        twin_id,
        a,
        b,
        c,
        f1,
        n1,
        p1,
        has_f2,
        f2,
        n2,
        p2,
        d,
    })
}

/// 创建中点顶点 M 和边分裂所需的全部新半边。
fn create_split_elements(
    mesh: &mut MeshStorage,
    data: &SplitEdgeData,
) -> Result<SplitHalfedges, TopologyError> {
    let pos_a = mesh
        .get_vertex(data.a)
        .ok_or_else(|| TopologyError::Inconsistent("顶点 A 不存在".into()))?
        .position;
    let pos_b = mesh
        .get_vertex(data.b)
        .ok_or_else(|| TopologyError::Inconsistent("顶点 B 不存在".into()))?
        .position;
    let mid = [
        (pos_a[0] + pos_b[0]) * 0.5,
        (pos_a[1] + pos_b[1]) * 0.5,
        (pos_a[2] + pos_b[2]) * 0.5,
    ];
    let m = mesh.add_vertex(Vertex::new(mid));

    let a_to_m = mesh.add_halfedge(HalfEdge::new(m)); // A→M
    let m_to_a = mesh.add_halfedge(HalfEdge::new(data.a)); // M→A
    let m_to_b = mesh.add_halfedge(HalfEdge::new(data.b)); // M→B
    let b_to_m = mesh.add_halfedge(HalfEdge::new(m)); // B→M
    let m_to_c = mesh.add_halfedge(HalfEdge::new(data.c)); // M→C
    let c_to_m = mesh.add_halfedge(HalfEdge::new(m)); // C→M

    let (m_to_d, d_to_m) = if data.has_f2 {
        let d = data.d.expect("d is Some when has_f2");
        (
            Some(mesh.add_halfedge(HalfEdge::new(d))), // M→D
            Some(mesh.add_halfedge(HalfEdge::new(m))), // D→M
        )
    } else {
        (None, None)
    };

    Ok(SplitHalfedges {
        m,
        a_to_m,
        m_to_a,
        m_to_b,
        b_to_m,
        m_to_c,
        c_to_m,
        m_to_d,
        d_to_m,
    })
}

/// 重连 F1 面 = (A→M→C→A) 并新建 F_new1 = (M→B→C→M)。
fn reconnect_split_f1_side(mesh: &mut MeshStorage, data: &SplitEdgeData, he: &SplitHalfedges) {
    // F1 = (A→M→C→A)
    {
        let am = mesh
            .get_halfedge_mut(he.a_to_m)
            .expect("a_to_m just created");
        am.twin = Some(he.m_to_a);
        am.next = Some(he.m_to_c);
        am.prev = Some(data.p1);
        am.face = Some(data.f1);

        let mc = mesh
            .get_halfedge_mut(he.m_to_c)
            .expect("m_to_c just created");
        mc.twin = Some(he.c_to_m);
        mc.next = Some(data.p1);
        mc.prev = Some(he.a_to_m);
        mc.face = Some(data.f1);

        // p1 (C→A): 原 next=h, prev=n1 → 现 next=a_to_m, prev=m_to_c
        let p1h = mesh
            .get_halfedge_mut(data.p1)
            .expect("p1 validated earlier");
        p1h.next = Some(he.a_to_m);
        p1h.prev = Some(he.m_to_c);
    }

    // F_new1 = (M→B→C→M)
    let f_new1 = mesh.add_face(Face::new());
    {
        let mb = mesh
            .get_halfedge_mut(he.m_to_b)
            .expect("m_to_b just created");
        mb.twin = Some(he.b_to_m);
        mb.next = Some(data.n1);
        mb.prev = Some(he.c_to_m);
        mb.face = Some(f_new1);

        // n1 (B→C): 原 next=p1, prev=h → 现 next=c_to_m, prev=m_to_b, face=F_new1
        let n1h = mesh
            .get_halfedge_mut(data.n1)
            .expect("n1 validated earlier");
        n1h.next = Some(he.c_to_m);
        n1h.prev = Some(he.m_to_b);
        n1h.face = Some(f_new1);

        let cm = mesh
            .get_halfedge_mut(he.c_to_m)
            .expect("c_to_m just created");
        cm.twin = Some(he.m_to_c);
        cm.next = Some(he.m_to_b);
        cm.prev = Some(data.n1);
        cm.face = Some(f_new1);
    }
    mesh.get_face_mut(f_new1)
        .expect("f_new1 just created")
        .halfedge = Some(he.m_to_b);
}

/// 处理 twin 侧：内部边重连 F2 + 新建 F_new2，或边界边分裂 twin。
fn reconnect_split_twin_side(mesh: &mut MeshStorage, data: &SplitEdgeData, he: &SplitHalfedges) {
    if data.has_f2 {
        let f2 = data.f2.expect("f2 is Some when has_f2");
        let n2 = data.n2.expect("n2 is Some when has_f2");
        let p2 = data.p2.expect("p2 is Some when has_f2");
        let m_to_d = he.m_to_d.expect("m_to_d is Some when has_f2");
        let d_to_m = he.d_to_m.expect("d_to_m is Some when has_f2");

        // F2 重用为 (B→M→D→B)
        {
            let bm = mesh
                .get_halfedge_mut(he.b_to_m)
                .expect("b_to_m just created");
            bm.twin = Some(he.m_to_b);
            bm.next = Some(m_to_d);
            bm.prev = Some(p2);
            bm.face = Some(f2);

            let md = mesh.get_halfedge_mut(m_to_d).expect("m_to_d just created");
            md.twin = Some(d_to_m);
            md.next = Some(p2);
            md.prev = Some(he.b_to_m);
            md.face = Some(f2);

            // p2 (D→B): 原 next=twin, prev=n2 → 现 next=b_to_m, prev=m_to_d
            let p2h = mesh.get_halfedge_mut(p2).expect("p2 validated earlier");
            p2h.next = Some(he.b_to_m);
            p2h.prev = Some(m_to_d);
        }

        // 新建 F_new2 = (M→A→D→M)
        let f_new2 = mesh.add_face(Face::new());
        {
            let ma = mesh
                .get_halfedge_mut(he.m_to_a)
                .expect("m_to_a just created");
            ma.twin = Some(he.a_to_m);
            ma.next = Some(n2);
            ma.prev = Some(d_to_m);
            ma.face = Some(f_new2);

            // n2 (A→D): 原 next=p2, prev=twin → 现 next=d_to_m, prev=m_to_a, face=F_new2
            let n2h = mesh.get_halfedge_mut(n2).expect("n2 validated earlier");
            n2h.next = Some(d_to_m);
            n2h.prev = Some(he.m_to_a);
            n2h.face = Some(f_new2);

            let dm = mesh.get_halfedge_mut(d_to_m).expect("d_to_m just created");
            dm.twin = Some(m_to_d);
            dm.next = Some(he.m_to_a);
            dm.prev = Some(n2);
            dm.face = Some(f_new2);
        }
        mesh.get_face_mut(f_new2)
            .expect("f_new2 just created")
            .halfedge = Some(he.m_to_a);

        mesh.get_face_mut(f2)
            .expect("f2 validated earlier")
            .halfedge = Some(he.b_to_m);
    } else {
        // 边界情形：twin (B→A) 分裂为 b_to_m (B→M) + m_to_a (M→A)，均 face=None
        let bm = mesh
            .get_halfedge_mut(he.b_to_m)
            .expect("b_to_m just created");
        bm.twin = Some(he.m_to_b);
        bm.face = None;

        let ma = mesh
            .get_halfedge_mut(he.m_to_a)
            .expect("m_to_a just created");
        ma.twin = Some(he.a_to_m);
        ma.face = None;
    }
}

/// 更新面入口、顶点 outgoing，删除原 h 和 twin。
fn finalize_split_edge(mesh: &mut MeshStorage, data: &SplitEdgeData, he: &SplitHalfedges) {
    // 面入口更新
    mesh.get_face_mut(data.f1)
        .expect("f1 validated earlier")
        .halfedge = Some(he.a_to_m);

    // 顶点 outgoing 更新
    if mesh
        .get_vertex(data.a)
        .expect("vertex a validated earlier")
        .halfedge
        == Some(data.h_id)
    {
        mesh.get_vertex_mut(data.a)
            .expect("vertex a validated earlier")
            .halfedge = Some(he.a_to_m);
    }
    if mesh
        .get_vertex(data.b)
        .expect("vertex b validated earlier")
        .halfedge
        == Some(data.twin_id)
    {
        mesh.get_vertex_mut(data.b)
            .expect("vertex b validated earlier")
            .halfedge = Some(he.b_to_m);
    }
    // M 的 outgoing 入口必须是 origin=M 的半边（即 twin.vertex=M）。
    // a_to_m 是 A→M（vertex=M, origin=A），是 A 的 outgoing；
    // m_to_a 是 M→A（vertex=A, origin=M），是 M 的 outgoing。
    mesh.get_vertex_mut(he.m).expect("m just created").halfedge = Some(he.m_to_a);

    // 删除原 h, twin
    mesh.remove_halfedge(data.h_id);
    mesh.remove_halfedge(data.twin_id);
}

// ============================================================
// flip_edge：内部边翻转
// ============================================================

/// 翻转内部边 `he` 所在的对角线，将四边形 `A-B-C-D` 的对角线从 `A-B` 替换为 `C-D`。
///
/// ## 拓扑修改逻辑
///
/// 设 `h = he`，`A = h.twin.vertex`（origin），`B = h.vertex`（tip）。
/// `F1 = h.face = (A→B→C→A)`，`F2 = twin.face = (B→A→D→B)`。
/// `n1 = h.next (B→C)`，`p1 = h.prev (C→A)`，
/// `n2 = twin.next (A→D)`，`p2 = twin.prev (D→B)`。
///
/// 翻转后：
/// - `h` 从 `A→B` 变为 `D→C`（`vertex = C`）；
/// - `twin` 从 `B→A` 变为 `C→D`（`vertex = D`）；
/// - `F1` 重用为 `(D→C→A→D) = h → p1 → n2`；
/// - `F2` 重用为 `(C→D→B→C) = twin → p2 → n1`。
///
/// # 校验
/// - `h.face` 与 `twin.face` 均须为 `Some`（内部边，禁止翻转边界边）；
/// - `C != D`（防退化）。
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, flip_edge};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 3], [1, 2, 3]];
/// let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
///
/// let interior_he = mesh.halfedge_ids()
///     .find(|&he| {
///         let h = mesh.get_halfedge(he).unwrap();
///         h.face.is_some() && h.twin.and_then(|t| mesh.get_halfedge(t)).map_or(false, |t| t.face.is_some())
///     })
///     .unwrap();
///
/// flip_edge(&mut mesh, interior_he).unwrap();
/// assert_eq!(mesh.face_count(), 2);
/// ```
pub fn flip_edge(mesh: &mut MeshStorage, he: HalfEdgeId) -> Result<(), TopologyError> {
    // ---------- 1. 校验与数据收集 ----------

    let h = mesh
        .get_halfedge(he)
        .ok_or(TopologyError::InvalidHalfEdge(he))?
        .clone();
    let twin_id = h.twin.ok_or(TopologyError::NoTwin(he))?;
    let twin = mesh
        .get_halfedge(twin_id)
        .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
        .clone();

    let f1 = h.face.ok_or(TopologyError::FlipOnBoundaryEdge(he))?;
    let f2 = twin.face.ok_or(TopologyError::FlipOnBoundaryEdge(he))?;

    let a = twin.vertex; // origin of h（翻转后 A 不再是 h 的端点）
    let b = h.vertex; // tip of h（翻转后 B 不再是 h 的端点）

    let n1 = h
        .next
        .ok_or_else(|| TopologyError::Inconsistent("h.next 为 None".into()))?;
    let p1 = h
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("h.prev 为 None".into()))?;
    let n2 = twin
        .next
        .ok_or_else(|| TopologyError::Inconsistent("twin.next 为 None".into()))?;
    let p2 = twin
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("twin.prev 为 None".into()))?;

    let c = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .vertex;
    let d = mesh
        .get_halfedge(n2)
        .ok_or(TopologyError::InvalidHalfEdge(n2))?
        .vertex;

    if c == d {
        return Err(TopologyError::DegenerateTriangle);
    }

    // ---------- 2. 翻转 h / twin ----------

    // h: A→B → D→C (vertex=C)
    // twin: B→A → C→D (vertex=D)
    {
        let h_mut = mesh
            .get_halfedge_mut(he)
            .expect("he validated at function start");
        h_mut.vertex = c;
        h_mut.next = Some(p1);
        h_mut.prev = Some(n2);
        h_mut.face = Some(f1);

        let twin_mut = mesh
            .get_halfedge_mut(twin_id)
            .expect("twin_id validated earlier");
        twin_mut.vertex = d;
        twin_mut.next = Some(p2);
        twin_mut.prev = Some(n1);
        twin_mut.face = Some(f2);
    }

    // ---------- 3. 重连四条邻接半边 ----------

    // F1 = (D→C→A→D) = h → p1 → n2 → h
    // p1 (C→A): 原 next=h, prev=n1 → 现 next=n2, prev=h
    // n2 (A→D): 原 next=p2, prev=twin → 现 next=h, prev=p1, face=F1
    {
        let p1_mut = mesh.get_halfedge_mut(p1).expect("p1 validated earlier");
        p1_mut.next = Some(n2);
        p1_mut.prev = Some(he);

        let n2_mut = mesh.get_halfedge_mut(n2).expect("n2 validated earlier");
        n2_mut.next = Some(he);
        n2_mut.prev = Some(p1);
        n2_mut.face = Some(f1);
    }

    // F2 = (C→D→B→C) = twin → p2 → n1 → twin
    // p2 (D→B): 原 next=twin, prev=n2 → 现 next=n1, prev=twin
    // n1 (B→C): 原 next=p1, prev=h → 现 next=twin, prev=p2, face=F2
    {
        let p2_mut = mesh.get_halfedge_mut(p2).expect("p2 validated earlier");
        p2_mut.next = Some(n1);
        p2_mut.prev = Some(twin_id);

        let n1_mut = mesh.get_halfedge_mut(n1).expect("n1 validated earlier");
        n1_mut.next = Some(twin_id);
        n1_mut.prev = Some(p2);
        n1_mut.face = Some(f2);
    }

    // ---------- 4. 面入口 ----------

    mesh.get_face_mut(f1)
        .expect("f1 validated earlier")
        .halfedge = Some(he);
    mesh.get_face_mut(f2)
        .expect("f2 validated earlier")
        .halfedge = Some(twin_id);

    // ---------- 5. 顶点 outgoing 更新 ----------

    // A 原本 outgoing 可能是 he（A→B），翻转后 he 起点变为 D，需更新为 n2（A→D）
    if mesh
        .get_vertex(a)
        .expect("vertex a is twin.vertex, validated")
        .halfedge
        == Some(he)
    {
        mesh.get_vertex_mut(a)
            .expect("vertex a is twin.vertex, validated")
            .halfedge = Some(n2);
    }
    // B 原本 outgoing 可能是 twin（B→A），翻转后 twin 起点变为 C，需更新为 n1（B→C）
    if mesh
        .get_vertex(b)
        .expect("vertex b is h.vertex, validated")
        .halfedge
        == Some(twin_id)
    {
        mesh.get_vertex_mut(b)
            .expect("vertex b is h.vertex, validated")
            .halfedge = Some(n1);
    }
    // C、D 的 outgoing 不需要改（它们仍指向 n1/p1/n2/p2，这些半边存活）

    Ok(())
}

// ============================================================
// collapse_edge：边折叠
// ============================================================

/// 折叠半边 `he` 两端顶点为一个新顶点 `K`，删除两个相邻三角形。
///
/// 新顶点 `K` 的位置为 `A`、`B` 中点。若需指定最优位置（如 QEM 减面），
/// 请使用 [`collapse_edge_at`]。
///
/// ## 拓扑修改逻辑
///
/// 设 `h = he`，`A = h.twin.vertex`（origin），`B = h.vertex`（tip）。
/// `F1 = h.face = (A→B→C→A)`，`F2 = twin.face = (B→A→D→B)`。
/// `n1 = h.next (B→C)`，`p1 = h.prev (C→A)`，
/// `n2 = twin.next (A→D)`，`p2 = twin.prev (D→B)`。
///
/// # 校验
/// - `h.face` 与 `twin.face` 均须为 `Some`（禁止折叠边界边）；
/// - `C != D`（防退化三角形）；
/// - **链接条件**：`A` 与 `B` 的公共邻居集合恰好为 `{C, D}`，否则折叠会产生非流形。
///
/// # 操作步骤
/// 1. 创建新顶点 `K`（位置为 `A`、`B` 中点，或由 `target_pos` 指定）；
/// 2. 缝合孔洞：`p1.twin ↔ n1.twin`，`n2.twin ↔ p2.twin`；
/// 3. 将所有 `vertex = A` 或 `vertex = B` 的存活半边的 `vertex` 更新为 `K`；
/// 4. 更新 `K`、`C`、`D` 的 `outgoing` 入口；
/// 5. 删除 `h, twin, n1, p1, n2, p2, F1, F2, A, B`。
///
/// # 返回
/// 新顶点 `K` 的 ID。
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, collapse_edge};
///
/// let verts = vec![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
/// ];
/// let faces = vec![[0u32, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
/// let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let faces_before = mesh.face_count();
///
/// let he = mesh.halfedge_ids().next().unwrap();
/// let _new_v = collapse_edge(&mut mesh, he).unwrap();
/// assert_eq!(mesh.face_count(), faces_before - 2);
/// ```
pub fn collapse_edge(mesh: &mut MeshStorage, he: HalfEdgeId) -> Result<VertexId, TopologyError> {
    collapse_edge_impl(mesh, he, None)
}

/// 折叠半边 `he`，新顶点位置为 `target_pos`（而非中点）。
///
/// 用于 QEM 减面等需要最优折叠位置的场景。拓扑修改逻辑与
/// [`collapse_edge`] 完全相同，仅新顶点 `K` 的位置不同。
///
/// # 返回
/// 新顶点 `K` 的 ID。
pub fn collapse_edge_at(
    mesh: &mut MeshStorage,
    he: HalfEdgeId,
    target_pos: [f64; 3],
) -> Result<VertexId, TopologyError> {
    collapse_edge_impl(mesh, he, Some(target_pos))
}

/// 边折叠核心实现。
///
/// `target_pos = None` 时新顶点取 `A`、`B` 中点；
/// `target_pos = Some(p)` 时新顶点取 `p`（用于 QEM 减面等）。
fn collapse_edge_impl(
    mesh: &mut MeshStorage,
    he: HalfEdgeId,
    target_pos: Option<[f64; 3]>,
) -> Result<VertexId, TopologyError> {
    let data = collect_collapse_data(mesh, he)?;
    let to_update = collect_halfedges_to_update(mesh, &data);
    let new_pos = compute_collapse_position(mesh, &data, target_pos)?;
    let k = mesh.add_vertex(Vertex::new(new_pos));
    sew_collapse_twins(mesh, &data);
    update_collapse_vertex_refs(mesh, &data, &to_update, k);
    remove_collapse_elements(mesh, &data);
    Ok(k)
}

// ---- collapse_edge_impl 辅助 ----

/// 边折叠收集的拓扑数据。
struct CollapseData {
    he: HalfEdgeId,
    twin_id: HalfEdgeId,
    a: VertexId,
    b: VertexId,
    c: VertexId,
    d: VertexId,
    f1: FaceId,
    f2: FaceId,
    n1: HalfEdgeId,
    p1: HalfEdgeId,
    n2: HalfEdgeId,
    p2: HalfEdgeId,
    /// 缝合用的 twin 半边 ID（均不在被删除集合中）。
    p1_twin: HalfEdgeId,
    n1_twin: HalfEdgeId,
    n2_twin: HalfEdgeId,
    p2_twin: HalfEdgeId,
}

/// 校验半边并收集边折叠所需的拓扑数据（含链接条件检查与缝合 twin ID）。
fn collect_collapse_data(
    mesh: &MeshStorage,
    he: HalfEdgeId,
) -> Result<CollapseData, TopologyError> {
    let h = mesh
        .get_halfedge(he)
        .ok_or(TopologyError::InvalidHalfEdge(he))?
        .clone();
    let twin_id = h.twin.ok_or(TopologyError::NoTwin(he))?;
    let twin = mesh
        .get_halfedge(twin_id)
        .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
        .clone();

    let f1 = h.face.ok_or(TopologyError::CollapseOnBoundaryEdge(he))?;
    let f2 = twin.face.ok_or(TopologyError::CollapseOnBoundaryEdge(he))?;

    let a = twin.vertex;
    let b = h.vertex;

    let n1 = h
        .next
        .ok_or_else(|| TopologyError::Inconsistent("h.next 为 None".into()))?;
    let p1 = h
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("h.prev 为 None".into()))?;
    let n2 = twin
        .next
        .ok_or_else(|| TopologyError::Inconsistent("twin.next 为 None".into()))?;
    let p2 = twin
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("twin.prev 为 None".into()))?;

    let c = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .vertex;
    let d = mesh
        .get_halfedge(n2)
        .ok_or(TopologyError::InvalidHalfEdge(n2))?
        .vertex;

    if c == d {
        return Err(TopologyError::DegenerateTriangle);
    }

    // 链接条件：A、B 的公共邻居恰好为 {C, D}
    if !check_link_condition(mesh, a, b, c, d) {
        return Err(TopologyError::LinkConditionViolated { a, b });
    }

    // 获取 twin 的 twin ID（缝合用），注意这些 twin 不在 deleted_set 中
    let p1_twin = mesh
        .get_halfedge(p1)
        .ok_or(TopologyError::InvalidHalfEdge(p1))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("p1.twin 为 None".into()))?;
    let n1_twin = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("n1.twin 为 None".into()))?;
    let n2_twin = mesh
        .get_halfedge(n2)
        .ok_or(TopologyError::InvalidHalfEdge(n2))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("n2.twin 为 None".into()))?;
    let p2_twin = mesh
        .get_halfedge(p2)
        .ok_or(TopologyError::InvalidHalfEdge(p2))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("p2.twin 为 None".into()))?;

    Ok(CollapseData {
        he,
        twin_id,
        a,
        b,
        c,
        d,
        f1,
        f2,
        n1,
        p1,
        n2,
        p2,
        p1_twin,
        n1_twin,
        n2_twin,
        p2_twin,
    })
}

/// 收集 A、B 的所有 incoming 半边（排除将被删除的），供后续批量更新 vertex。
fn collect_halfedges_to_update(mesh: &MeshStorage, data: &CollapseData) -> Vec<HalfEdgeId> {
    let deleted = [data.he, data.twin_id, data.n1, data.p1, data.n2, data.p2];
    let deleted_set: HashSet<HalfEdgeId> = deleted.iter().copied().collect();

    let mut to_update: Vec<HalfEdgeId> = Vec::new();
    for out_he in VertexRing::new(mesh, data.a).collect::<Vec<_>>() {
        if let Some(t_id) = mesh.get_halfedge(out_he).and_then(|h| h.twin)
            && !deleted_set.contains(&t_id)
        {
            to_update.push(t_id);
        }
    }
    for out_he in VertexRing::new(mesh, data.b).collect::<Vec<_>>() {
        if let Some(t_id) = mesh.get_halfedge(out_he).and_then(|h| h.twin)
            && !deleted_set.contains(&t_id)
        {
            to_update.push(t_id);
        }
    }
    to_update
}

/// 计算折叠后新顶点 K 的位置：`target_pos = Some(p)` 用 `p`，否则取 A、B 中点。
fn compute_collapse_position(
    mesh: &MeshStorage,
    data: &CollapseData,
    target_pos: Option<[f64; 3]>,
) -> Result<[f64; 3], TopologyError> {
    match target_pos {
        Some(p) => Ok(p),
        None => {
            let pos_a = mesh
                .get_vertex(data.a)
                .ok_or_else(|| TopologyError::Inconsistent("顶点 A 不存在".into()))?
                .position;
            let pos_b = mesh
                .get_vertex(data.b)
                .ok_or_else(|| TopologyError::Inconsistent("顶点 B 不存在".into()))?
                .position;
            Ok([
                (pos_a[0] + pos_b[0]) * 0.5,
                (pos_a[1] + pos_b[1]) * 0.5,
                (pos_a[2] + pos_b[2]) * 0.5,
            ])
        }
    }
}

/// 缝合 twin：`p1.twin ↔ n1.twin`，`n2.twin ↔ p2.twin`。
fn sew_collapse_twins(mesh: &mut MeshStorage, data: &CollapseData) {
    // p1.twin (A→C) 与 n1.twin (C→B) 互为 twin（原 twin p1/n1 被删除）
    {
        let p1t = mesh
            .get_halfedge_mut(data.p1_twin)
            .expect("p1_twin validated earlier");
        p1t.twin = Some(data.n1_twin);
        let n1t = mesh
            .get_halfedge_mut(data.n1_twin)
            .expect("n1_twin validated earlier");
        n1t.twin = Some(data.p1_twin);
    }
    // n2.twin (D→A) 与 p2.twin (B→D) 互为 twin（原 twin n2/p2 被删除）
    {
        let n2t = mesh
            .get_halfedge_mut(data.n2_twin)
            .expect("n2_twin validated earlier");
        n2t.twin = Some(data.p2_twin);
        let p2t = mesh
            .get_halfedge_mut(data.p2_twin)
            .expect("p2_twin validated earlier");
        p2t.twin = Some(data.n2_twin);
    }
}

/// 批量更新 `vertex = A/B → K`，并修复 K、C、D 的 outgoing 入口。
fn update_collapse_vertex_refs(
    mesh: &mut MeshStorage,
    data: &CollapseData,
    to_update: &[HalfEdgeId],
    k: VertexId,
) {
    // 批量更新 vertex = A/B → K
    for he_id in to_update {
        let h_mut = mesh
            .get_halfedge_mut(*he_id)
            .expect("he_id collected from valid halfedges");
        if h_mut.vertex == data.a || h_mut.vertex == data.b {
            h_mut.vertex = k;
        }
    }

    // K 的 outgoing：优先选 p1_twin（原 A→C，现 K→C），它一定存活
    mesh.get_vertex_mut(k).expect("k just created").halfedge = Some(data.p1_twin);

    // C 的 outgoing 若指向被删除的半边，更新为 n1_twin（C→K）
    let c_out = mesh
        .get_vertex(data.c)
        .expect("vertex c validated earlier")
        .halfedge;
    if c_out == Some(data.n1) || c_out == Some(data.p1) || c_out.is_none() {
        mesh.get_vertex_mut(data.c)
            .expect("vertex c validated earlier")
            .halfedge = Some(data.n1_twin);
    }
    // D 的 outgoing 若指向被删除的半边，更新为 n2_twin（D→K）
    let d_out = mesh
        .get_vertex(data.d)
        .expect("vertex d validated earlier")
        .halfedge;
    if d_out == Some(data.n2) || d_out == Some(data.p2) || d_out.is_none() {
        mesh.get_vertex_mut(data.d)
            .expect("vertex d validated earlier")
            .halfedge = Some(data.n2_twin);
    }
}

/// 删除折叠后废弃的半边、面和顶点。
fn remove_collapse_elements(mesh: &mut MeshStorage, data: &CollapseData) {
    mesh.remove_halfedge(data.he);
    mesh.remove_halfedge(data.twin_id);
    mesh.remove_halfedge(data.n1);
    mesh.remove_halfedge(data.p1);
    mesh.remove_halfedge(data.n2);
    mesh.remove_halfedge(data.p2);
    mesh.remove_face(data.f1);
    mesh.remove_face(data.f2);
    mesh.remove_vertex(data.a);
    mesh.remove_vertex(data.b);
}

/// 检查链接条件：`A` 与 `B` 的公共邻居恰好为 `{C, D}`。
pub(super) fn check_link_condition(
    mesh: &MeshStorage,
    a: VertexId,
    b: VertexId,
    c: VertexId,
    d: VertexId,
) -> bool {
    let neighbors_a: HashSet<VertexId> = VertexAdjacentVerts::new(mesh, a).collect();
    let neighbors_b: HashSet<VertexId> = VertexAdjacentVerts::new(mesh, b).collect();
    let common: HashSet<&VertexId> = neighbors_a.intersection(&neighbors_b).collect();
    common.len() == 2 && common.contains(&c) && common.contains(&d)
}

// ============================================================
// split_face：面分裂 / poke
// ============================================================

/// 面分裂：在面中心插入新顶点，将 1 个三角面分裂为 3 个。
///
/// 新顶点位于原面三个顶点的几何中心。返回新顶点 ID。
pub fn split_face(mesh: &mut MeshStorage, face: FaceId) -> Result<VertexId, TopologyError> {
    // 1. 获取面信息
    let face_data = mesh
        .get_face(face)
        .ok_or_else(|| TopologyError::Inconsistent("面不存在".into()))?;
    let _start_he = face_data
        .halfedge
        .ok_or_else(|| TopologyError::Inconsistent("面无半边".into()))?;

    // 2. 收集面顶点
    let he_ids: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    if he_ids.len() != 3 {
        return Err(TopologyError::Inconsistent("仅支持三角面".into()));
    }
    let he_data: Vec<_> = he_ids
        .iter()
        .map(|&h| {
            mesh.get_halfedge(h)
                .expect("h from FaceHalfEdges, validated")
                .clone()
        })
        .collect();
    let verts: Vec<VertexId> = he_data.iter().map(|h| h.vertex).collect();
    let [v0, v1, v2] = [verts[0], verts[1], verts[2]];
    let [_he0, _he1, _he2] = [he_ids[0], he_ids[1], he_ids[2]];
    let old_twins = [he_data[0].twin, he_data[1].twin, he_data[2].twin];

    // 3. 计算中心点
    let pos: Vec<[f64; 3]> = verts
        .iter()
        .map(|&v| {
            mesh.get_vertex(v)
                .expect("v from he_data, validated")
                .position
        })
        .collect();
    let center = [
        (pos[0][0] + pos[1][0] + pos[2][0]) / 3.0,
        (pos[0][1] + pos[1][1] + pos[2][1]) / 3.0,
        (pos[0][2] + pos[1][2] + pos[2][2]) / 3.0,
    ];

    // 4. 插入新顶点
    let m = mesh.add_vertex(Vertex::new(center));

    // 5. 删除旧面及其半边
    mesh.remove_face(face);
    for &he in &he_ids {
        mesh.remove_halfedge(he);
    }

    // 6. 创建 3 个新面，每个面两条"辐条"边 + 一条外边界边
    // F0: m→v1→v2, F1: m→v2→v0, F2: m→v0→v1
    let mut new_faces = Vec::new();
    for (i, &(_, src, dst)) in [(0, v0, v1), (1, v1, v2), (2, v2, v0)].iter().enumerate() {
        // 辐条边: m→src, dst→m
        let spoke_out = mesh.add_halfedge(HalfEdge::new(src)); // m→src
        let spoke_in = mesh.add_halfedge(HalfEdge::new(m)); // dst→m

        // 辐条互为 twin
        mesh.get_halfedge_mut(spoke_out)
            .expect("spoke_out just created")
            .twin = Some(spoke_in);
        mesh.get_halfedge_mut(spoke_in)
            .expect("spoke_in just created")
            .twin = Some(spoke_out);

        // 外边界边：从 src→dst
        let outer = mesh.add_halfedge(HalfEdge::new(dst));
        // 恢复旧 twin 关系
        if let Some(old_twin) = old_twins[(i + 1) % 3] {
            mesh.get_halfedge_mut(outer)
                .expect("outer just created")
                .twin = Some(old_twin);
            if let Some(t) = mesh.get_halfedge_mut(old_twin) {
                t.twin = Some(outer);
            }
        }

        // 设置 next/prev 环: m→src→dst→m (spoke_out → outer → spoke_in)
        mesh.get_halfedge_mut(spoke_out)
            .expect("spoke_out just created")
            .next = Some(outer);
        mesh.get_halfedge_mut(spoke_out)
            .expect("spoke_out just created")
            .prev = Some(spoke_in);
        mesh.get_halfedge_mut(outer)
            .expect("outer just created")
            .next = Some(spoke_in);
        mesh.get_halfedge_mut(outer)
            .expect("outer just created")
            .prev = Some(spoke_out);
        mesh.get_halfedge_mut(spoke_in)
            .expect("spoke_in just created")
            .next = Some(spoke_out);
        mesh.get_halfedge_mut(spoke_in)
            .expect("spoke_in just created")
            .prev = Some(outer);

        // 创建面
        let new_f = mesh.add_face(Face::new());
        mesh.get_face_mut(new_f)
            .expect("new_f just created")
            .halfedge = Some(spoke_out);
        for &he in &[spoke_out, outer, spoke_in] {
            mesh.get_halfedge_mut(he).expect("he just created").face = Some(new_f);
        }
        new_faces.push(new_f);
    }

    // 7. 设置中心顶点的 outgoing 半边
    mesh.get_vertex_mut(m).expect("m just created").halfedge = Some(
        mesh.get_face(new_faces[0])
            .and_then(|f| f.halfedge)
            .expect("face halfedge just set"),
    );

    // 8. 校验
    validate_mesh(mesh)?;

    Ok(m)
}
