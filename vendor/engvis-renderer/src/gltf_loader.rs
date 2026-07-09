use engvis_core::{
    Aabb, LightingEnvironment, Mesh, MeshVertex, PbrMaterial, Scene, SceneNode,
    SubMesh,
};
use glam::{Affine3A, Quat, Vec3};
use crate::texture_cache::TextureCache;

/// Convert gltf image data to image::RgbaImage
fn gltf_image_to_rgba(img: &gltf::image::Data) -> image::RgbaImage {
    let pixels = &img.pixels;
    let width = img.width;
    let height = img.height;

    match img.format {
        gltf::image::Format::R8G8B8A8 => {
            image::RgbaImage::from_raw(width, height, pixels.clone())
                .unwrap_or_else(|| image::RgbaImage::new(width, height))
        }
        gltf::image::Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity(pixels.len() / 3 * 4);
            for chunk in pixels.chunks(3) {
                if chunk.len() == 3 {
                    rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
                }
            }
            image::RgbaImage::from_raw(width, height, rgba)
                .unwrap_or_else(|| image::RgbaImage::new(width, height))
        }
        _ => {
            image::RgbaImage::from_raw(width, height, pixels.clone())
                .unwrap_or_else(|| image::RgbaImage::new(width, height))
        }
    }
}

#[derive(Debug)]
pub enum GltfLoadError {
    Io(std::io::Error),
    Gltf(gltf::Error),
    Image(image::ImageError),
}

impl From<std::io::Error> for GltfLoadError {
    fn from(e: std::io::Error) -> Self {
        GltfLoadError::Io(e)
    }
}

impl From<gltf::Error> for GltfLoadError {
    fn from(e: gltf::Error) -> Self {
        GltfLoadError::Gltf(e)
    }
}

impl From<image::ImageError> for GltfLoadError {
    fn from(e: image::ImageError) -> Self {
        GltfLoadError::Image(e)
    }
}

impl std::fmt::Display for GltfLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GltfLoadError::Io(e) => write!(f, "IO error: {}", e),
            GltfLoadError::Gltf(e) => write!(f, "glTF error: {}", e),
            GltfLoadError::Image(e) => write!(f, "Image error: {}", e),
        }
    }
}

impl std::error::Error for GltfLoadError {}

/// Load a glTF file into a `(Scene, Aabb)` pair.
/// The returned Aabb is the scene's world-space bounding box, computed from
/// the node hierarchy (not the local AABB of each mesh).
pub fn load_gltf(
    path: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_cache: &mut TextureCache,
) -> Result<(Scene, Aabb), GltfLoadError> {
    let (gltf, buffers, images) = gltf::import(path)?;

    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let mut nodes = Vec::new();

    // 1. Load materials
    for mat in gltf.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let mut material = PbrMaterial {
            name: mat.name().unwrap_or("unnamed").to_string(),
            albedo: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive: mat.emissive_factor(),
            normal_scale: mat.normal_texture().map_or(1.0, |t| t.scale()),
            alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
            albedo_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            emissive_texture: None,
        };

        // Upload albedo texture
        if let Some(info) = pbr.base_color_texture() {
            let tex_index = info.texture().index();
            if tex_index < images.len() {
                let img = gltf_image_to_rgba(&images[tex_index]);
                let idx = texture_cache.upload_image(
                    device,
                    queue,
                    &img,
                    "gltf_albedo",
                    true,
                );
                material.albedo_texture = Some(idx);
            }
        }

        // Upload metallic-roughness texture
        if let Some(info) = pbr.metallic_roughness_texture() {
            let tex_index = info.texture().index();
            if tex_index < images.len() {
                let img = gltf_image_to_rgba(&images[tex_index]);
                let idx = texture_cache.upload_image(
                    device,
                    queue,
                    &img,
                    "gltf_mr",
                    false,
                );
                material.metallic_roughness_texture = Some(idx);
            }
        }

        // Upload normal texture
        if let Some(info) = mat.normal_texture() {
            let tex_index = info.texture().index();
            if tex_index < images.len() {
                let img = gltf_image_to_rgba(&images[tex_index]);
                let idx = texture_cache.upload_image(
                    device,
                    queue,
                    &img,
                    "gltf_normal",
                    false,
                );
                material.normal_texture = Some(idx);
            }
        }

        // Upload emissive texture
        if let Some(info) = mat.emissive_texture() {
            let tex_index = info.texture().index();
            if tex_index < images.len() {
                let img = gltf_image_to_rgba(&images[tex_index]);
                let idx = texture_cache.upload_image(
                    device,
                    queue,
                    &img,
                    "gltf_emissive",
                    true,
                );
                material.emissive_texture = Some(idx);
            }
        }

        materials.push(material);
    }

    // Ensure at least one material
    if materials.is_empty() {
        materials.push(PbrMaterial::default());
    }

    // 2. Load meshes
    for gltf_mesh in gltf.meshes() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut sub_meshes = Vec::new();
        let mut aabb = Aabb::empty();

        for primitive in gltf_mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|iter| iter.collect())
                .unwrap_or_default();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_default();
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_default();

            let prim_indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_default();

            // Compute tangents
            let tangents = if let Some(tan) = reader.read_tangents() {
                tan.collect::<Vec<[f32; 4]>>()
            } else {
                engvis_core::math::compute_tangents(&positions, &normals, &uvs, &prim_indices)
            };

           let base_vertex = vertices.len() as u32;
           for (i, &pos) in positions.iter().enumerate() {
               vertices.push(MeshVertex {
                   position: pos,
                   normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                   uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                   tangent: tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
               });
               aabb.expand(Vec3::from(pos));
            }

            let index_offset = indices.len() as u32;
            let adjusted_indices: Vec<u32> =
                prim_indices.iter().map(|i| i + base_vertex).collect();
            indices.extend_from_slice(&adjusted_indices);

            sub_meshes.push(SubMesh {
                material_index: primitive.material().index().unwrap_or(0),
                index_offset,
                index_count: adjusted_indices.len() as u32,
            });
        }

        meshes.push(Mesh {
            name: gltf_mesh.name().unwrap_or("unnamed").to_string(),
            vertices,
            indices,
            sub_meshes,
            aabb,
        });
    }

    // 3. Load node hierarchy
    fn load_node(node: &gltf::Node) -> SceneNode {
        let (translation, rotation, scale) = node.transform().decomposed();
        let local_transform = Affine3A::from_scale_rotation_translation(
            Vec3::from(scale),
            Quat::from_array(rotation),
            Vec3::from(translation),
        );

        SceneNode {
            name: node.name().unwrap_or("node").to_string(),
            local_transform,
            mesh_index: node.mesh().map(|m| m.index()),
            children: node.children().map(|child| load_node(&child)).collect(),
            visible: true,
            render_surface: true,
            render_edges: false,
            edge_color_override: None,
            edge_width_override: None,
        }
    }

    for gltf_scene in gltf.scenes() {
        for node in gltf_scene.nodes() {
            nodes.push(load_node(&node));
        }
    }

    let scene = Scene {
        meshes,
        materials,
        nodes,
        lighting: LightingEnvironment::default(),
    };

    let aabb = scene.compute_aabb();

    Ok((scene, aabb))
}
