use std::borrow::Cow;

/// A custom material shader + pipeline that can replace the default PBR material
/// for selected mesh / sub-mesh combinations.
///
/// Implementations provide the WGSL shader source and a callback to configure
/// bind group layouts and draw calls.
pub trait CustomMaterial: std::fmt::Debug {
    /// Return a label for debug/profiling purposes.
    fn label(&self) -> &str {
        "custom_material"
    }

    /// Vertex shader WGSL source.
    fn vertex_shader(&self) -> Cow<'static, str>;

    /// Fragment shader WGSL source.
    fn fragment_shader(&self) -> Cow<'static, str>;

    /// Descriptors for additional bind group entries (beyond group 0 = scene uniforms).
    /// The returned entries will be placed in group 1.
    fn extra_bind_group_layout_entries(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        Vec::new()
    }

    /// Called during pipeline creation. Return the bind group for group 1.
    fn create_extra_bind_group(
        &self,
        _device: &wgpu::Device,
        _layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let layout_ref = _device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("empty_custom_bind_group_layout"),
                entries: &[],
            },
        );
        _device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty_custom_bind_group"),
            layout: &layout_ref,
            entries: &[],
        })
    }

    /// Configure the `RenderPipelineDescriptor` before creation.
    /// The default sets vertex/fragment shaders and uses the provided layouts.
    fn configure_pipeline(
        &self,
        _desc: &mut wgpu::RenderPipelineDescriptor,
        _vertex_shader: &wgpu::ShaderModule,
        _fragment_shader: &wgpu::ShaderModule,
    ) {
    }

    /// Should this custom material be used for the given material index?
    fn matches(&self, material_index: usize) -> bool;
}

/// A simple custom material that renders meshes in a flat solid color,
/// bypassing the PBR lighting pipeline.
///
/// Use it to verify the `CustomMaterial` trait works, or as a starting
/// point for your own custom shaders.
///
/// # Example
///
/// ```ignore
/// renderer.set_custom_material(Some(Box::new(BasicMaterial {
///     color: [1.0, 0.2, 0.2],
///     target_material_index: 0,
/// })));
/// ```
#[derive(Debug)]
pub struct BasicMaterial {
    /// Flat RGB color for the fragment shader output.
    pub color: [f32; 3],
    /// Only replace rendering for sub-meshes with this material index.
    /// Sub-meshes with other indices will use the default PBR pipeline.
    pub target_material_index: usize,
}

impl BasicMaterial {
    /// WGSL vertex shader: transform by view-proj * model (bind group 3 = object).
    /// Matches the standard vertex format: position @ location 0, normal @ 1, uv @ 2, tangent @ 3.
    pub const VERTEX_SHADER: &'static str = r#"
struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(3) @binding(0) var<uniform> object: ObjectUniforms;

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    viewport: vec4<f32>,
    global_opacity: vec4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip = scene.view_proj * object.model * vec4(input.position, 1.0);
    return out;
}
"#;

    /// WGSL fragment shader: output a flat color.
    pub const FRAGMENT_SHADER: &'static str = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4(1.0, 0.2, 0.2, 1.0);
}
"#;

    pub fn new(color: [f32; 3], target_material_index: usize) -> Self {
        Self { color, target_material_index }
    }
}

impl CustomMaterial for BasicMaterial {
    fn label(&self) -> &str {
        "BasicMaterial"
    }

    fn vertex_shader(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::VERTEX_SHADER)
    }

    fn fragment_shader(&self) -> Cow<'static, str> {
        // Return a shader string with the color baked in
        let src = format!(
            "@fragment\nfn fs_main() -> @location(0) vec4<f32> {{\n    return vec4({:.3}, {:.3}, {:.3}, 1.0);\n}}\n",
            self.color[0], self.color[1], self.color[2]
        );
        Cow::Owned(src)
    }

    fn matches(&self, material_index: usize) -> bool {
        material_index == self.target_material_index
    }
}
