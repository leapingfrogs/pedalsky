//! Phase 6 cloud render pipelines.
//!
//! Two render pipelines are needed:
//! - `march` — fragment shader that does the volumetric raymarch and
//!   writes premultiplied luminance to the cloud RT.
//! - `composite` — premultiplied-alpha blit of the cloud RT over the HDR
//!   target. `One, OneMinusSrcAlpha` blend.

use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, world_bind_group_layout,
    HdrFramebuffer,
};

const CLOUD_UNIFORMS_BAKED: &str =
    include_str!("../../../shaders/clouds/cloud_uniforms.wgsl");
const CLOUD_UNIFORMS_REL: &str = "clouds/cloud_uniforms.wgsl";
const CLOUD_MARCH_BAKED: &str = include_str!("../../../shaders/clouds/cloud_march.wgsl");
const CLOUD_MARCH_REL: &str = "clouds/cloud_march.wgsl";
const CLOUD_COMPOSITE_BAKED: &str =
    include_str!("../../../shaders/clouds/cloud_composite.wgsl");
const CLOUD_COMPOSITE_REL: &str = "clouds/cloud_composite.wgsl";
const CLOUD_COMPOSITE_HALFRES_BAKED: &str =
    include_str!("../../../shaders/clouds/cloud_composite_halfres.wgsl");
const CLOUD_COMPOSITE_HALFRES_REL: &str =
    "clouds/cloud_composite_halfres.wgsl";
const CLOUD_TAA_BAKED: &str =
    include_str!("../../../shaders/clouds/cloud_taa.wgsl");
const CLOUD_TAA_REL: &str = "clouds/cloud_taa.wgsl";

/// Group-0 bind layout for the full-res composite pass: cloud-
/// luminance RT view + cloud-transmittance RT view + shared
/// sampler. Phase 12.2 — the composite uses dual-source blending
/// to apply per-channel transmittance to the destination HDR
/// target.
///
/// The half-res composite pipeline uses a different layout — see
/// [`composite_halfres_bind_group_layout`] — that adds a uniform
/// buffer with the cloud RT dimensions for the Catmull-Rom kernel.
/// Keeping the two layouts separate (rather than one 4-binding
/// superset) means the full-res pipeline's compiled fragment shader
/// is byte-identical to the pre-toggle baseline.
pub fn composite_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clouds-composite-bgl"),
        entries: &[
            // Premultiplied luminance attachment.
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
            // RGB transmittance attachment.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Shared sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Group-0 bind layout for the cloud TAA pass.
///
/// Six bindings: current luminance + current transmittance (from the
/// march scratch), history luminance + history transmittance (from
/// the previous frame's resolved slot), a shared sampler, and the
/// `CloudTaaParams` uniform with the blend weight and history-validity
/// flag.
///
/// Frame uniforms (group 1) are bound separately so the shader can
/// read `prev_view_proj` and `inv_view_proj` for the reprojection.
pub fn taa_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_2d = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clouds-taa-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Group-0 bind layout for the half-res composite pipeline. Adds a
/// 4th binding for the `CloudCompositeParams` uniform that the
/// Catmull-Rom kernel reads (cloud-RT pixel dimensions).
pub fn composite_halfres_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clouds-composite-halfres-bgl"),
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Cloud render pipelines + the bind-group layout owned by the
/// composite pass (the march pipeline reuses `CloudNoise::layout`
/// passed in at construction).
pub struct CloudPipelines {
    /// Full-res fragment raymarch pipeline. Built from the verbatim
    /// `cloud_march.wgsl` source so its compiled output is byte-
    /// identical to the pre-half-res-toggle baseline.
    pub march: wgpu::RenderPipeline,
    /// Half-res fragment raymarch pipeline. Built from
    /// `cloud_march.wgsl` after a couple of surgical string
    /// substitutions that retarget the depth-buffer lookup and the
    /// AP-LUT UV from `frame.viewport_size` (always full-res) to
    /// `params.cloud_rt_size` (the half-res cloud RT). The two
    /// pipelines share their bind-group layout — the differences
    /// are purely fragment-shader source-level.
    pub march_halfres: wgpu::RenderPipeline,
    /// Full-res composite — verbatim original shader, byte-identical
    /// compiled output. Used whenever the half-res toggle is off.
    pub composite: wgpu::RenderPipeline,
    /// Half-res composite — runs the 9-tap Catmull-Rom kernel from
    /// `cloud_composite_halfres.wgsl`. Distinct pipeline (and distinct
    /// bind-group layout, see `composite_halfres_layout`) so the
    /// full-res shader is not perturbed.
    pub composite_halfres: wgpu::RenderPipeline,
    /// Full-res composite group-0 layout (cloud RT + sampler).
    pub composite_layout: wgpu::BindGroupLayout,
    /// Half-res composite group-0 layout (same as `composite_layout`
    /// plus a `CloudCompositeParams` uniform at binding 3).
    pub composite_halfres_layout: wgpu::BindGroupLayout,
    /// Cloud TAA fragment pipeline. Writes blended (current ↔
    /// history) luminance + transmittance to two MRT attachments.
    pub taa: wgpu::RenderPipeline,
    /// TAA pass group-0 layout (current + history pairs + sampler +
    /// `CloudTaaParams` uniform).
    pub taa_layout: wgpu::BindGroupLayout,
}

/// Apply the half-res patches to a copy of the cloud march source.
///
/// The patcher does four things:
///   1. Inject a `@group(2) @binding(12) var<uniform> cloud_rt_uniform: vec4<f32>;`
///      declaration after the existing binding-11 (`wind_field`) entry.
///   2. Rewrite the primary `compute_view_ray(in.pos.xy)` call so the
///      shared helper (which divides by `frame.viewport_size.xy`
///      internally) gets a frag coord pre-scaled to its full-res
///      equivalent. Without this the primary cloud ray runs from
///      NDC in `[-1, 0]` (the top-left quadrant of the screen)
///      stretched to fill the framebuffer — the symptom reads as
///      "clouds doubled in size and clipped by the ground plane".
///   3. Rewrite the unscaled `textureLoad(scene_depth, jitter_xy, 0)`
///      to a scaled version that maps cloud-RT pixel coords back to
///      the (always-full-res) depth buffer's pixel coords.
///   4. Retarget the two `let viewport = frame.viewport_size.xy;`
///      lines that feed the depth-NDC reconstruction and the AP-LUT
///      UV to `cloud_rt_uniform.xy`.
///
/// Crucially this transform leaves `cloud_march.wgsl` AND
/// `cloud_uniforms.wgsl` byte-identical to their pre-toggle form, so
/// the full-res march pipeline's compiled fragment shader is
/// guaranteed bit-exact with the baseline.
///
/// Returns `Err` if any of the expected source markers is missing
/// — that catches a silent drift between this code and the shader.
fn patch_cloud_march_for_halfres(source: &str) -> Result<String, &'static str> {
    let inject_marker =
        "@group(2) @binding(11) var wind_field: texture_3d<f32>;";
    let inject_replacement =
        "@group(2) @binding(11) var wind_field: texture_3d<f32>;\n\
         // Half-res variant binding: live cloud-RT size + inverse.\n\
         @group(2) @binding(12) var<uniform> cloud_rt_uniform: vec4<f32>;";
    if !source.contains(inject_marker) {
        return Err(
            "cloud_march.wgsl: half-res patch missing binding-11 injection marker",
        );
    }

    // Primary view-ray reconstruction. `compute_view_ray` (in
    // common/math.wgsl) divides the frag coord by
    // `frame.viewport_size.xy`. At half-res `in.pos.xy` is in the
    // smaller cloud-RT pixel space, so the helper computes NDC in
    // `[-1, 0]`. Pre-scale by `viewport / cloud_rt_uniform` (= (2, 2)
    // at half-res, identity at full-res) before the call.
    let view_ray_call = "let ray = compute_view_ray(in.pos.xy);";
    let view_ray_call_halfres =
        "let ray = compute_view_ray(in.pos.xy * \
         (frame.viewport_size.xy / cloud_rt_uniform.xy));";
    if !source.contains(view_ray_call) {
        return Err("cloud_march.wgsl: half-res patch missing compute_view_ray marker");
    }

    let depth_lookup = "let depth_ndc = textureLoad(scene_depth, jitter_xy, 0);";
    let depth_lookup_halfres =
        "// Half-res render: jitter_xy is in cloud-RT pixel space \
         (which is smaller than the framebuffer); scale up to the \
         depth buffer's pixel space, which is always full-res.\n    \
         let depth_scale = frame.viewport_size.xy / cloud_rt_uniform.xy;\n    \
         let depth_xy = vec2<i32>(vec2<f32>(jitter_xy) * depth_scale);\n    \
         let depth_ndc = textureLoad(scene_depth, depth_xy, 0);";
    if !source.contains(depth_lookup) {
        return Err("cloud_march.wgsl: half-res patch missing depth-lookup marker");
    }
    let viewport_line = "let viewport = frame.viewport_size.xy;";
    let viewport_line_halfres = "let viewport = cloud_rt_uniform.xy;";
    if source.matches(viewport_line).count() != 2 {
        return Err("cloud_march.wgsl: half-res patch expected 2 viewport_size markers");
    }
    Ok(source
        .replace(inject_marker, inject_replacement)
        .replace(view_ray_call, view_ray_call_halfres)
        .replace(depth_lookup, depth_lookup_halfres)
        .replace(viewport_line, viewport_line_halfres))
}

impl CloudPipelines {
    /// Build all pipelines. `cloud_data_layout` and
    /// `cloud_data_halfres_layout` come from `CloudNoise`: the former
    /// has the 12 bindings the full-res march pipeline expects, the
    /// latter has those 12 plus a 13th uniform binding for the
    /// half-res variant.
    pub fn new(
        device: &wgpu::Device,
        cloud_data_layout: &wgpu::BindGroupLayout,
        cloud_data_halfres_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let lut_layout = atmosphere_lut_bind_group_layout(device);
        let composite_layout = composite_bind_group_layout(device);

        // March pipeline.
        let cloud_uniforms_src =
            ps_core::shaders::load_shader(CLOUD_UNIFORMS_REL, CLOUD_UNIFORMS_BAKED);
        let cloud_march_src =
            ps_core::shaders::load_shader(CLOUD_MARCH_REL, CLOUD_MARCH_BAKED);
        let march_src = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_MATH_WGSL,
            &cloud_uniforms_src,
            &cloud_march_src,
        ]);
        let march_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds-march-shader"),
            source: wgpu::ShaderSource::Wgsl(march_src.clone().into()),
        });
        let march_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-march-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(cloud_data_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });
        let march_targets = [
            // MRT attachments (Phase 12.2):
            //   0: premultiplied luminance (no blend; written verbatim)
            //   1: RGB transmittance through the cloud column
            Some(wgpu::ColorTargetState {
                format: HdrFramebuffer::COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: HdrFramebuffer::COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let march = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds-march"),
            layout: Some(&march_pl),
            vertex: wgpu::VertexState {
                module: &march_module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &march_module,
                entry_point: Some("fs_main"),
                targets: &march_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Half-res variant — patch only the cloud march fragment
        // source, with its OWN pipeline layout (and bind group
        // layout) so the full-res pipeline's WGSL→MSL emission is
        // not perturbed by binding-12's presence.
        let march_halfres_src = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_MATH_WGSL,
            &cloud_uniforms_src,
            &patch_cloud_march_for_halfres(&cloud_march_src)
                .expect("cloud_march.wgsl half-res patch failed (shader markers drifted)"),
        ]);
        let march_halfres_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds-march-halfres-shader"),
            source: wgpu::ShaderSource::Wgsl(march_halfres_src.into()),
        });
        let march_halfres_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-march-halfres-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(cloud_data_halfres_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });
        let march_halfres = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds-march-halfres"),
            layout: Some(&march_halfres_pl),
            vertex: wgpu::VertexState {
                module: &march_halfres_module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &march_halfres_module,
                entry_point: Some("fs_main"),
                targets: &march_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Composite pipeline — full-res (verbatim original shader).
        let cloud_composite_src =
            ps_core::shaders::load_shader(CLOUD_COMPOSITE_REL, CLOUD_COMPOSITE_BAKED);
        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(cloud_composite_src.into()),
        });
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-composite-pl"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });
        // Shared blend / target state: dual-source blending for
        // per-channel transmittance (Phase 12.2).
        let composite_target = Some(wgpu::ColorTargetState {
            format: HdrFramebuffer::COLOR_FORMAT,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::Src1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::Src1,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let composite_targets = [composite_target];
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds-composite"),
            layout: Some(&composite_pl),
            vertex: wgpu::VertexState {
                module: &composite_module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_module,
                entry_point: Some("fs_main"),
                targets: &composite_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // The composite pass writes over the HDR target which has a depth
            // attachment; setting depth_stencil = None means the render pass
            // descriptor must omit the depth attachment too. The pass we
            // build in lib.rs does exactly that.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Composite pipeline — half-res (9-tap Catmull-Rom upsample).
        let composite_halfres_layout = composite_halfres_bind_group_layout(device);
        let cloud_composite_halfres_src = ps_core::shaders::load_shader(
            CLOUD_COMPOSITE_HALFRES_REL,
            CLOUD_COMPOSITE_HALFRES_BAKED,
        );
        let composite_halfres_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("clouds-composite-halfres-shader"),
                source: wgpu::ShaderSource::Wgsl(cloud_composite_halfres_src.into()),
            });
        let composite_halfres_pl =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("clouds-composite-halfres-pl"),
                bind_group_layouts: &[Some(&composite_halfres_layout)],
                immediate_size: 0,
            });
        let composite_halfres = device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("clouds-composite-halfres"),
                layout: Some(&composite_halfres_pl),
                vertex: wgpu::VertexState {
                    module: &composite_halfres_module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_halfres_module,
                    entry_point: Some("fs_main"),
                    targets: &composite_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        // Cloud TAA pipeline. The shader reads two pairs of textures
        // (current march + previous resolved) and `frame.prev_view_proj`
        // for the reprojection, and writes the blended pair to its
        // two MRT attachments. The output target format matches the
        // cloud RT (Rgba16Float).
        let taa_layout = taa_bind_group_layout(device);
        let cloud_taa_src = ps_core::shaders::load_shader(CLOUD_TAA_REL, CLOUD_TAA_BAKED);
        // Compose with shared uniforms so the shader can declare
        // `var<uniform> frame: FrameUniforms;` against group 1.
        let taa_src = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            &cloud_taa_src,
        ]);
        let taa_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds-taa-shader"),
            source: wgpu::ShaderSource::Wgsl(taa_src.into()),
        });
        let taa_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-taa-pl"),
            bind_group_layouts: &[Some(&taa_layout), Some(&frame_layout)],
            immediate_size: 0,
        });
        // Two MRT attachments — luminance + transmittance — written
        // verbatim (no blend). The TAA shader has already done the
        // EMA + clamp.
        let taa_targets = [
            Some(wgpu::ColorTargetState {
                format: HdrFramebuffer::COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: HdrFramebuffer::COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let taa = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds-taa"),
            layout: Some(&taa_pl),
            vertex: wgpu::VertexState {
                module: &taa_module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &taa_module,
                entry_point: Some("fs_main"),
                targets: &taa_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            march,
            march_halfres,
            composite,
            composite_halfres,
            composite_layout,
            composite_halfres_layout,
            taa,
            taa_layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The half-res patch is a string transform; if the cloud march
    /// shader is edited and the markers drift, the patcher must fail
    /// loudly rather than silently producing an unmodified half-res
    /// shader. This test runs the patcher against the baked source
    /// (no GPU required) and checks both substitutions landed.
    #[test]
    fn halfres_patch_applies_cleanly() {
        let source = ps_core::shaders::load_shader(CLOUD_MARCH_REL, CLOUD_MARCH_BAKED);
        let patched =
            patch_cloud_march_for_halfres(&source).expect("half-res patch failed");
        assert!(
            patched.contains("frame.viewport_size.xy / cloud_rt_uniform.xy"),
            "depth-scale line missing from patched half-res shader"
        );
        assert!(
            patched.contains("let viewport = cloud_rt_uniform.xy;"),
            "viewport retarget missing from patched half-res shader"
        );
        assert!(
            !patched.contains("let viewport = frame.viewport_size.xy;"),
            "patched shader still references full-res viewport"
        );
        assert!(
            patched.contains("@group(2) @binding(12) var<uniform> cloud_rt_uniform"),
            "half-res binding declaration missing from patched shader"
        );
        assert!(
            patched.contains(
                "compute_view_ray(in.pos.xy * \
                 (frame.viewport_size.xy / cloud_rt_uniform.xy))"
            ),
            "compute_view_ray call site not rescaled in half-res shader"
        );
        assert!(
            !patched.contains("compute_view_ray(in.pos.xy);"),
            "patched shader still has the unscaled compute_view_ray call"
        );
    }
}
