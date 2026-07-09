use wgpu::util::DeviceExt;

/// FXAA post-processing pipeline.
///
/// Renders a fullscreen triangle that samples the MSAA-resolved scene texture
/// and applies Fast Approximate Anti-Aliasing (FXAA 3.11 quality 12) before
/// writing to the final output.
pub struct PostProcessPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl PostProcessPipeline {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FXAA Post-Process Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::shader_source().into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Post-Process Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Post-Process Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Fullscreen triangle (covers clip space [-1,1] without a vertex buffer)
        let vertices: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fullscreen Triangle VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FXAA Post-Process Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FXAA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            sampler,
        }
    }

    /// Record an FXAA pass: read from `source_view`, write to `dest_view`.
    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_view: &wgpu::TextureView,
        dest_view: &wgpu::TextureView,
        _width: u32,
        _height: u32,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FXAA Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Post-Process Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
    }

    fn shader_source() -> &'static str {
        r#"
// ── FXAA 3.11 (simplified, quality preset 12) ─────────────────

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var out: VSOut;
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    return out;
}

fn rgb_to_luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(input_tex));

    // Sample center and 4 neighbors
    let rgb_m  = textureSample(input_tex, input_sampler, in.uv).rgb;
    let luma_m = rgb_to_luma(rgb_m);

    let rgb_n  = textureSample(input_tex, input_sampler, in.uv + vec2<f32>( 0.0, -1.0) * texel).rgb;
    let rgb_s  = textureSample(input_tex, input_sampler, in.uv + vec2<f32>( 0.0,  1.0) * texel).rgb;
    let rgb_e  = textureSample(input_tex, input_sampler, in.uv + vec2<f32>( 1.0,  0.0) * texel).rgb;
    let rgb_w  = textureSample(input_tex, input_sampler, in.uv + vec2<f32>(-1.0,  0.0) * texel).rgb;

    let luma_n = rgb_to_luma(rgb_n);
    let luma_s = rgb_to_luma(rgb_s);
    let luma_e = rgb_to_luma(rgb_e);
    let luma_w = rgb_to_luma(rgb_w);

    let range_max = max(max(luma_n, luma_s), max(luma_e, luma_w));
    let range_min = min(min(luma_n, luma_s), min(luma_e, luma_w));
    let range = range_max - range_min;

    // Skip pixels with low contrast
    if (range < max(0.0312, range_max * 0.125)) {
        return vec4<f32>(rgb_m, 1.0);
    }

    // Diagonal samples
    let rgb_nw = textureSample(input_tex, input_sampler, in.uv + vec2<f32>(-1.0, -1.0) * texel).rgb;
    let rgb_ne = textureSample(input_tex, input_sampler, in.uv + vec2<f32>( 1.0, -1.0) * texel).rgb;
    let rgb_sw = textureSample(input_tex, input_sampler, in.uv + vec2<f32>(-1.0,  1.0) * texel).rgb;
    let rgb_se = textureSample(input_tex, input_sampler, in.uv + vec2<f32>( 1.0,  1.0) * texel).rgb;

    let luma_nw = rgb_to_luma(rgb_nw);
    let luma_ne = rgb_to_luma(rgb_ne);
    let luma_sw = rgb_to_luma(rgb_sw);
    let luma_se = rgb_to_luma(rgb_se);

    // Edge direction: horizontal vs vertical
    let edge_h = abs(luma_nw + luma_ne - 2.0 * luma_n)
               + abs(luma_w  + luma_e  - 2.0 * luma_m) * 2.0
               + abs(luma_sw + luma_se - 2.0 * luma_s);
    let edge_v = abs(luma_nw + luma_sw - 2.0 * luma_w)
               + abs(luma_n  + luma_s  - 2.0 * luma_m) * 2.0
               + abs(luma_ne + luma_se - 2.0 * luma_e);
    let is_horizontal = edge_h >= edge_v;

    // Choose the steeper side
    let luma_1 = select(luma_w, luma_n, is_horizontal);
    let luma_2 = select(luma_e, luma_s, is_horizontal);
    let gradient_1 = abs(luma_1 - luma_m);
    let gradient_2 = abs(luma_2 - luma_m);

    var dir: vec2<f32>;
    if (is_horizontal) {
        dir = select(vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0), gradient_1 >= gradient_2);
    } else {
        dir = select(vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0), gradient_1 >= gradient_2);
    }

    let pixel_step = dir * texel;

    // Edge search in both directions along the edge
    let grad_scaled = max(gradient_1, gradient_2) * 0.25;
    var uv_a = in.uv + pixel_step * 0.5;
    var uv_b = in.uv - pixel_step * 0.5;
    var luma_end_a = rgb_to_luma(textureSample(input_tex, input_sampler, uv_a).rgb) - luma_m;
    var luma_end_b = rgb_to_luma(textureSample(input_tex, input_sampler, uv_b).rgb) - luma_m;
    var done_a = abs(luma_end_a) >= grad_scaled;
    var done_b = abs(luma_end_b) >= grad_scaled;

    for (var i = 0; i < 12; i = i + 1) {
        if (!done_a) {
            uv_a = uv_a + pixel_step;
            luma_end_a = rgb_to_luma(textureSample(input_tex, input_sampler, uv_a).rgb) - luma_m;
            done_a = abs(luma_end_a) >= grad_scaled;
        }
        if (!done_b) {
            uv_b = uv_b - pixel_step;
            luma_end_b = rgb_to_luma(textureSample(input_tex, input_sampler, uv_b).rgb) - luma_m;
            done_b = abs(luma_end_b) >= grad_scaled;
        }
        if (done_a && done_b) { break; }
    }

    let dist_a = length(uv_a - in.uv);
    let dist_b = length(uv_b - in.uv);
    let dist_shortest = select(dist_b, dist_a, dist_a < dist_b);
    let edge_len = dist_a + dist_b;
    var pixel_offset = 0.5 - dist_shortest / edge_len;

    // Sub-pixel anti-aliasing
    let luma_avg = (luma_n + luma_s + luma_e + luma_w) * 0.25;
    let sub_pixel = clamp(abs(luma_avg - luma_m) / range, 0.0, 1.0);
    let sub_offset = smoothstep(0.0, 1.0, sub_pixel);
    pixel_offset = max(pixel_offset, sub_offset * sub_offset * (1.0 / 3.0));

    // Final blend
    let final_uv = in.uv + pixel_step * pixel_offset;
    let result = textureSample(input_tex, input_sampler, final_uv).rgb;
    return vec4<f32>(result, 1.0);
}
"#
    }
}
