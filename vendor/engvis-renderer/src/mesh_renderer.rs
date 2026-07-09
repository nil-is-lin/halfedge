use wgpu::util::DeviceExt;
use engvis_core::Mesh;
use glam::{Affine3A, Mat4};

pub struct MeshBuffer {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub edge_endpoint_buffer: Option<wgpu::Buffer>,
    pub index_count: u32,
    pub edge_instance_count: u32,
    pub vertex_count: u32,
}

pub struct MeshRenderer {
    pub mesh_buffers: Vec<MeshBuffer>,
    pub material_bind_groups: Vec<wgpu::BindGroup>,
    pub material_uniform_buffers: Vec<wgpu::Buffer>,
    pub object_bind_group_layout: wgpu::BindGroupLayout,
}

/// GPU object uniform
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectUniforms {
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
}

impl MeshRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let object_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Object Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        Self {
            mesh_buffers: Vec::new(),
            material_bind_groups: Vec::new(),
            material_uniform_buffers: Vec::new(),
            object_bind_group_layout,
        }
    }

    pub fn upload_mesh(&mut self, device: &wgpu::Device, mesh: &Mesh) -> usize {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Mesh '{}' Vertex Buffer", mesh.name)),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Mesh '{}' Index Buffer", mesh.name)),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let edge_indices = mesh.extract_edge_indices();

        // Build edge endpoint buffer: each instance has two endpoints (24 bytes)
        #[repr(C)]
        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct EdgeEndpoints {
            pos_a: [f32; 3],
            pos_b: [f32; 3],
        }
        let edge_count = edge_indices.len() / 2;
        let mut edge_endpoints: Vec<EdgeEndpoints> = Vec::with_capacity(edge_count);
        for i in 0..edge_count {
            let i0 = edge_indices[i * 2] as usize;
            let i1 = edge_indices[i * 2 + 1] as usize;
            edge_endpoints.push(EdgeEndpoints {
                pos_a: mesh.vertices[i0].position,
                pos_b: mesh.vertices[i1].position,
            });
        }
        let edge_endpoint_buffer = {
            let byte_size = edge_count * std::mem::size_of::<EdgeEndpoints>();
            let max_buffer_size = device.limits().max_buffer_size as usize;
            if byte_size > max_buffer_size {
                log::warn!(
                    "Mesh '{}' edge buffer {} MB exceeds GPU limit {} MB, skipping edges",
                    mesh.name,
                    byte_size / (1024 * 1024),
                    max_buffer_size / (1024 * 1024),
                );
                None
            } else {
                Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Mesh '{}' Edge Endpoint Buffer", mesh.name)),
                    contents: bytemuck::cast_slice(&edge_endpoints),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
            }
        };

        let has_edges = edge_endpoint_buffer.is_some();
        let index = self.mesh_buffers.len();
        self.mesh_buffers.push(MeshBuffer {
            vertex_buffer,
            index_buffer,
            edge_endpoint_buffer,
            index_count: mesh.indices.len() as u32,
            edge_instance_count: if has_edges { edge_count as u32 } else { 0 },
            vertex_count: mesh.vertices.len() as u32,
        });
        index
    }

    pub fn create_object_bind_group(
        &self,
        device: &wgpu::Device,
        transform: Affine3A,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let model_mat = Mat4::from(transform);
        let normal_mat = model_mat.inverse().transpose();

        let uniforms = ObjectUniforms {
            model: model_mat.to_cols_array_2d(),
            normal_matrix: normal_mat.to_cols_array_2d(),
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Object Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Object Bind Group"),
            layout: &self.object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        (buffer, bind_group)
    }
}
