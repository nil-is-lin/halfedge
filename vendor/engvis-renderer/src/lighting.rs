use wgpu::util::DeviceExt;
use engvis_core::LightingEnvironment;

pub const MAX_DIR_LIGHTS: usize = 4;
pub const MAX_POINT_LIGHTS: usize = 16;

/// GPU layout for lighting uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingUniforms {
    pub ambient_color: [f32; 4],
    pub dir_light_count: u32,
    pub point_light_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// GPU layout for a single directional light
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionalLightData {
    pub direction: [f32; 4],
    pub color: [f32; 4],
}

/// GPU layout for a single point light
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLightData {
    pub position: [f32; 4],
    pub color: [f32; 4],
}

pub struct LightingBuffer {
    pub uniform_buffer: wgpu::Buffer,
    pub dir_light_buffer: wgpu::Buffer,
    pub point_light_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl LightingBuffer {
    pub fn new(device: &wgpu::Device, lighting: &LightingEnvironment) -> Self {
        let uniform_data = Self::build_uniforms(lighting);
        let dir_lights = Self::build_dir_lights(lighting);
        let point_lights = Self::build_point_lights(lighting);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lighting Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dir_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dir Light Storage Buffer"),
            contents: bytemuck::cast_slice(&dir_lights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let point_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Light Storage Buffer"),
            contents: bytemuck::cast_slice(&point_lights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dir_light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_light_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            uniform_buffer,
            dir_light_buffer,
            point_light_buffer,
            bind_group,
            bind_group_layout,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, lighting: &LightingEnvironment) {
        let uniforms = Self::build_uniforms(lighting);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let dir_lights = Self::build_dir_lights(lighting);
        queue.write_buffer(&self.dir_light_buffer, 0, bytemuck::cast_slice(&dir_lights));

        let point_lights = Self::build_point_lights(lighting);
        queue.write_buffer(
            &self.point_light_buffer,
            0,
            bytemuck::cast_slice(&point_lights),
        );
    }

    fn build_uniforms(lighting: &LightingEnvironment) -> LightingUniforms {
        LightingUniforms {
            ambient_color: [
                lighting.ambient.color[0],
                lighting.ambient.color[1],
                lighting.ambient.color[2],
                lighting.ambient.intensity,
            ],
            dir_light_count: lighting.directional_lights.len().min(MAX_DIR_LIGHTS) as u32,
            point_light_count: lighting.point_lights.len().min(MAX_POINT_LIGHTS) as u32,
            _pad0: 0,
            _pad1: 0,
        }
    }

    fn build_dir_lights(lighting: &LightingEnvironment) -> Vec<DirectionalLightData> {
        let mut lights = vec![
            DirectionalLightData {
                direction: [0.0; 4],
                color: [0.0; 4],
            };
            MAX_DIR_LIGHTS
        ];
        for (i, l) in lighting
            .directional_lights
            .iter()
            .take(MAX_DIR_LIGHTS)
            .enumerate()
        {
            lights[i] = DirectionalLightData {
                direction: [l.direction.x, l.direction.y, l.direction.z, l.intensity],
                color: [l.color[0], l.color[1], l.color[2], 0.0],
            };
        }
        lights
    }

    fn build_point_lights(lighting: &LightingEnvironment) -> Vec<PointLightData> {
        let mut lights = vec![
            PointLightData {
                position: [0.0; 4],
                color: [0.0; 4],
            };
            MAX_POINT_LIGHTS
        ];
        for (i, l) in lighting
            .point_lights
            .iter()
            .take(MAX_POINT_LIGHTS)
            .enumerate()
        {
            lights[i] = PointLightData {
                position: [l.position.x, l.position.y, l.position.z, l.range],
                color: [l.color[0], l.color[1], l.color[2], l.intensity],
            };
        }
        lights
    }
}
