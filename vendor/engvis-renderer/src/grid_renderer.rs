use wgpu::util::DeviceExt;

/// Vertex for grid/axis lines (position + color)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridVertex {
    position: [f32; 3],
    color: [f32; 4],
}

pub struct GridRenderer {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
}

impl GridRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        scene_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        let (vertices, vertex_count) = Self::generate_grid_vertices();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Simple bind group for grid: just scene uniforms (group 0)
        // and an identity model matrix uniform (group 1)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid Object Bind Group Layout"),
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

        // Identity model matrix
        let identity_matrix: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // Pack as { model: mat4, normal_matrix: mat4, background_color: vec4 }
        let uniform_data: [[f32; 4]; 9] = [
            identity_matrix[0],
            identity_matrix[1],
            identity_matrix[2],
            identity_matrix[3],
            identity_matrix[0],
            identity_matrix[1],
            identity_matrix[2],
            identity_matrix[3],
            [1.0, 1.0, 1.0, 0.0], // default white background
        ];

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Object Uniform Buffer"),
            contents: bytemuck::cast_slice(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Object Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader_source = Self::build_shader_source();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grid Pipeline Layout"),
            bind_group_layouts: &[scene_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GridVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::depth::DepthTexture::FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            vertex_buffer,
            vertex_count,
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
        }
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }

    /// Update the background colour reference held in the object uniform
    /// buffer so the fragment shader can adapt grid-line visibility.
    pub fn update_background_color(&self, queue: &wgpu::Queue, color: [f32; 3]) {
        let offset = 8 * std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress;
        queue.write_buffer(&self.uniform_buffer, offset, bytemuck::cast_slice(&[[color[0], color[1], color[2], 0.0_f32]]));
    }

    fn generate_grid_vertices() -> (Vec<GridVertex>, u32) {
        let mut vertices = Vec::new();

        let grid_half = 25;
        let major_every = 5;

        // Grid lines on XY plane (Z = 0)
        for i in -grid_half..=grid_half {
            let is_major = i % major_every == 0;
            let alpha = if is_major { 0.4 } else { 0.2 };
            let color = [0.4, 0.4, 0.4, alpha];

            let fi = i as f32;
            // Line along Y at x = fi
            vertices.push(GridVertex {
                position: [fi, -grid_half as f32, 0.0],
                color,
            });
            vertices.push(GridVertex {
                position: [fi, grid_half as f32, 0.0],
                color,
            });
            // Line along X at y = fi
            vertices.push(GridVertex {
                position: [-grid_half as f32, fi, 0.0],
                color,
            });
            vertices.push(GridVertex {
                position: [grid_half as f32, fi, 0.0],
                color,
            });
        }

        // X axis (red)
        vertices.push(GridVertex {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 0.2, 0.2, 0.9],
        });
        vertices.push(GridVertex {
            position: [grid_half as f32, 0.0, 0.0],
            color: [1.0, 0.2, 0.2, 0.9],
        });

        // Y axis (green)
        vertices.push(GridVertex {
            position: [0.0, 0.0, 0.0],
            color: [0.1, 0.65, 0.1, 0.9],
        });
        vertices.push(GridVertex {
            position: [0.0, grid_half as f32, 0.0],
            color: [0.1, 0.65, 0.1, 0.9],
        });

        // Z axis (blue)
        vertices.push(GridVertex {
            position: [0.0, 0.0, 0.0],
            color: [0.3, 0.4, 1.0, 0.9],
        });
        vertices.push(GridVertex {
            position: [0.0, 0.0, grid_half as f32],
            color: [0.3, 0.4, 1.0, 0.9],
        });

        let count = vertices.len() as u32;
        (vertices, count)
    }

    fn build_shader_source() -> String {
        r#"
struct SceneUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    viewport: vec4<f32>,
    global_opacity: vec4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    background_color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (object.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_pos = scene.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let bg_lum = dot(object.background_color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Brighter background -> increase grid alpha so lines stay visible.
    let alpha = in.color.a * (1.0 + bg_lum * 2.5);
    return vec4<f32>(in.color.rgb, alpha);
}
"#
        .to_string()
    }
}
