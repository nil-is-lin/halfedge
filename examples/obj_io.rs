//! 示例：OBJ 文件读写
//!
//! 演示 parse_obj / format_obj / save_obj / load_obj 的用法。
//! 运行：`cargo run --example obj_io`

use halfedge::{format_obj, load_obj, parse_obj, save_obj, validate_topology};
use std::env::temp_dir;

fn main() {
    // ---------- 1. 从 OBJ 文本解析 ----------
    // 注意：面绕向必须一致（每条边在两个面中方向相反），否则 build_mesh 会生成无 twin 的悬空半边
    let obj_text = r#"
# 四面体（面绕向一致，每条边在两面中方向相反）
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 2 3
f 1 3 4
f 1 4 2
f 2 4 3
"#;

    let mesh = parse_obj(obj_text).expect("OBJ 解析应成功");
    println!(
        "解析结果：{} 顶点 / {} 半边 / {} 面",
        mesh.vertex_count(),
        mesh.halfedge_count(),
        mesh.face_count()
    );

    let errors = validate_topology(&mesh);
    println!("拓扑校验：{} 个错误", errors.len());

    // ---------- 2. 序列化回 OBJ 文本 ----------
    let serialized = format_obj(&mesh);
    println!("\n序列化后 OBJ：");
    println!("{}", serialized.trim_end());

    // ---------- 3. 文件 IO 往返 ----------
    let path = temp_dir().join("halfedge_obj_io_example.obj");
    save_obj(&mesh, &path).expect("保存应成功");
    println!("已保存到 {:?}", path);

    let loaded = load_obj(&path).expect("加载应成功");
    let _ = std::fs::remove_file(&path);

    println!(
        "加载结果：{} 顶点 / {} 半边 / {} 面",
        loaded.vertex_count(),
        loaded.halfedge_count(),
        loaded.face_count()
    );
    println!(
        "往返一致：{}",
        loaded.vertex_count() == mesh.vertex_count()
            && loaded.face_count() == mesh.face_count()
            && loaded.halfedge_count() == mesh.halfedge_count()
    );

    // ---------- 4. 支持 v/vt/vn 与负索引 ----------
    let fancy = r#"
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/1 -3/-1/-1
"#;
    let m2 = parse_obj(fancy).expect("支持 v/vt/vn 与负索引");
    println!(
        "\n高级解析：{} 顶点 / {} 面（支持 v/vt/vn 与负索引）",
        m2.vertex_count(),
        m2.face_count()
    );
}
