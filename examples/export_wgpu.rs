//! 示例：导出 wgpu 兼容的顶点/索引缓冲
//!
//! 演示 export::mesh_to_vertex_index_buffers 输出 Vec<[f32;3]> + Vec<u32>。
//! 运行：`cargo run --example export_wgpu`

use halfedge::{build_icosphere, mesh_to_vertex_index_buffers};

fn main() {
    let mesh = build_icosphere(1); // 42 顶点 / 80 面
    println!(
        "icosphere(1)：{} 顶点 / {} 面",
        mesh.vertex_count(),
        mesh.face_count()
    );

    // 导出为扁平缓冲
    let (vertices, indices) = mesh_to_vertex_index_buffers(&mesh);

    println!("\n导出结果：");
    println!(
        "  顶点缓冲：{} 个 [f32;3]（{} 字节）",
        vertices.len(),
        vertices.len() * 12
    );
    println!(
        "  索引缓冲：{} 个 u32（{} 字节，{} 个三角形）",
        indices.len(),
        indices.len() * 4,
        indices.len() / 3
    );

    // 打印前 3 个顶点
    println!("\n前 3 个顶点：");
    for (i, v) in vertices.iter().take(3).enumerate() {
        println!("  [{}] [{:.4}, {:.4}, {:.4}]", i, v[0], v[1], v[2]);
    }

    // 打印第 1 个三角形的索引
    println!(
        "\n第 1 个三角形索引：[{}, {}, {}]",
        indices[0], indices[1], indices[2]
    );
    println!("  对应顶点：");
    for i in &indices[0..3] {
        let v = vertices[*i as usize];
        println!("    [{}] [{:.4}, {:.4}, {:.4}]", i, v[0], v[1], v[2]);
    }

    // ---------- wgpu 集成伪代码 ----------
    println!("\n[wgpu 集成伪代码]");
    println!("  let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {{");
    println!("      contents: bytemuck::cast_slice(&vertices),");
    println!("      usage: BufferUsages::VERTEX,");
    println!("  }});");
    println!("  let index_buffer = device.create_buffer_init(&BufferInitDescriptor {{");
    println!("      contents: bytemuck::cast_slice(&indices),");
    println!("      usage: BufferUsages::INDEX,");
    println!("  }});");
    println!("  render_pass.draw_indexed(0..{}, 0, 0..1);", indices.len());

    // ---------- 验证索引范围合法性 ----------
    let max_idx = *indices.iter().max().unwrap() as usize;
    println!(
        "\n索引范围检查：max index = {}, vertex count = {}, 合法 = {}",
        max_idx,
        vertices.len(),
        max_idx < vertices.len()
    );

    // ---------- 验证 CCW 朝向（第 1 个三角形法向应朝外） ----------
    let a = vertices[indices[0] as usize];
    let b = vertices[indices[1] as usize];
    let c = vertices[indices[2] as usize];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal_z = ab[0] * ac[1] - ab[1] * ac[0];
    let centroid = [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ];
    let outward = normal_z * centroid[2]
        + (ab[1] * ac[2] - ab[2] * ac[1]) * centroid[0]
        + (ab[2] * ac[0] - ab[0] * ac[2]) * centroid[1];
    println!("CCW 朝向检查：法向·重心 = {:.4}（正值表示朝外）", outward);
}
