//! 集成测试：端到端网格处理工作流
//!
//! 测试跨模块协作：构建 → 校验 → 几何计算 → 拓扑操作 → 导出

use halfedge::{
    build_cube, build_icosphere,
    geometry::{edge_length_stats, mesh_volume, surface_area},
    io::{format_obj, format_stl_binary, parse_obj, parse_stl_binary},
    isotropic_remesh,
    topology_ops::{flip_edge, split_edge},
    traversal::is_closed,
    validate_mesh, validate_topology,
};

/// 完整工作流：构建 icosphere → 校验 → 几何统计 → 导出 → 重新导入
#[test]
fn workflow_icosphere_validate_geometry_export_reimport() {
    // 1. 构建
    let mesh = build_icosphere(1);
    assert!(mesh.vertex_count() > 0);
    assert!(mesh.face_count() > 0);

    // 2. 校验
    assert!(validate_mesh(&mesh).is_ok());
    assert!(validate_topology(&mesh).is_empty());
    assert!(is_closed(&mesh));

    // 3. 几何统计
    let area = surface_area(&mesh);
    assert!(area > 0.0);
    let volume = mesh_volume(&mesh);
    assert!(volume.abs() > 0.0);
    let stats = edge_length_stats(&mesh);
    assert!(stats.count > 0);

    // 4. 导出为 OBJ
    let obj_text = format_obj(&mesh);
    assert!(obj_text.contains("v "));
    assert!(obj_text.contains("f "));

    // 5. 重新导入
    let mesh2 = parse_obj(&obj_text).expect("OBJ roundtrip 应成功");
    assert_eq!(mesh2.vertex_count(), mesh.vertex_count());
    assert_eq!(mesh2.face_count(), mesh.face_count());

    // 6. 导出为 STL binary 并重新导入
    let stl_bytes = format_stl_binary(&mesh);
    let mesh3 = parse_stl_binary(&stl_bytes, mesh.face_count()).expect("STL roundtrip 应成功");
    assert_eq!(mesh3.vertex_count(), mesh.vertex_count());
    assert_eq!(mesh3.face_count(), mesh.face_count());
}

/// 拓扑操作链：split → flip → collapse 不破坏流形
#[test]
fn workflow_topology_operations_chain() {
    let mut mesh = build_icosphere(0);

    // 选一条半边进行 split
    let he = mesh.halfedge_ids().next().expect("网格非空");
    if split_edge(&mut mesh, he).is_ok() {
        assert!(mesh.vertex_count() > 0);
        // split 后校验
        assert!(validate_mesh(&mesh).is_ok());
    }

    // 尝试 flip 另一条边
    let he2 = mesh.halfedge_ids().find(|&h| {
        if let Some(h_data) = mesh.get_halfedge(h) {
            h_data.face.is_some() && h_data.twin.is_some()
        } else {
            false
        }
    });
    if let Some(he2) = he2 {
        let _ = flip_edge(&mut mesh, he2);
        assert!(validate_mesh(&mesh).is_ok());
    }

    // remesh 后校验
    let _stats = isotropic_remesh(&mut mesh, None, 2, false);
    assert!(validate_mesh(&mesh).is_ok());
    assert!(is_closed(&mesh));
}

/// Cube 几何：体积近似 1.0，表面积近似 6.0
#[test]
fn workflow_cube_geometry_correctness() {
    let cube = build_cube(1.0);
    assert!(validate_mesh(&cube).is_ok());

    let volume = mesh_volume(&cube);
    assert!(
        (volume.abs() - 1.0).abs() < 0.01,
        "立方体体积应 ≈ 1.0，实际 {}",
        volume
    );

    let area = surface_area(&cube);
    assert!(
        (area - 6.0).abs() < 0.01,
        "立方体表面积应 ≈ 6.0，实际 {}",
        area
    );
}

/// 多格式 IO roundtrip：OBJ → PLY → STL → OFF 顶点/面数一致
#[test]
fn workflow_multi_format_roundtrip() {
    use halfedge::io::{format_off, format_ply, parse_off, parse_ply};

    let mesh = build_icosphere(0);
    let v_count = mesh.vertex_count();
    let f_count = mesh.face_count();

    // OBJ roundtrip
    let obj = format_obj(&mesh);
    let from_obj = parse_obj(&obj).expect("OBJ 解析");
    assert_eq!(from_obj.vertex_count(), v_count);
    assert_eq!(from_obj.face_count(), f_count);

    // PLY roundtrip
    let ply = format_ply(&mesh);
    let from_ply = parse_ply(&ply).expect("PLY 解析");
    assert_eq!(from_ply.vertex_count(), v_count);
    assert_eq!(from_ply.face_count(), f_count);

    // OFF roundtrip
    let off = format_off(&mesh);
    let from_off = parse_off(&off).expect("OFF 解析");
    assert_eq!(from_off.vertex_count(), v_count);
    assert_eq!(from_off.face_count(), f_count);
}
