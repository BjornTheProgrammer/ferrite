use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

#[derive(Debug, Clone, Copy)]
pub struct AtlasRegion {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

#[derive(Clone)]
pub struct AtlasUVMap {
    regions: HashMap<String, AtlasRegion>,
    missing: AtlasRegion,
}

impl AtlasUVMap {
    pub fn empty() -> Self {
        Self {
            regions: HashMap::new(),
            missing: AtlasRegion {
                u_min: 0.0,
                v_min: 0.0,
                u_max: 1.0,
                v_max: 1.0,
            },
        }
    }

    pub fn get_region(&self, name: &str) -> AtlasRegion {
        self.regions.get(name).copied().unwrap_or(self.missing)
    }
}

pub struct TextureAtlas {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub uv_map: AtlasUVMap,
    allocation: Option<Allocation>,
    staging_buffer: vk::Buffer,
    staging_allocation: Option<Allocation>,
}

impl TextureAtlas {
    pub fn build(
        device: &ash::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        allocator: &Arc<Mutex<Allocator>>,
        assets_dir: &Path,
        texture_names: &HashSet<&str>,
    ) -> Result<Self, vk::Result> {
        let tile_size = 16u32;
        let grid_size = (texture_names.len() as f32 + 1.0).sqrt().ceil() as u32 + 1;
        let atlas_size = (grid_size * tile_size).next_power_of_two();

        let mut atlas_pixels = vec![0u8; (atlas_size * atlas_size * 4) as usize];
        let mut regions = HashMap::new();

        let missing_region = tile_region(tile_origin(0, grid_size, tile_size), tile_size, atlas_size);

        for py in 0..tile_size {
            for px in 0..tile_size {
                let is_check = ((px / 8) + (py / 8)) % 2 == 0;
                let color: [u8; 4] = if is_check {
                    [255, 0, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                let idx = ((py * atlas_size + px) * 4) as usize;
                atlas_pixels[idx..idx + 4].copy_from_slice(&color);
            }
        }

        let textures_path = assets_dir.join("assets/minecraft/textures/block");
        let mut slot = 1u32;

        for &name in texture_names {
            let file_path = textures_path.join(format!("{name}.png"));
            let pixels = match load_png(&file_path) {
                Some(p) => p,
                None => {
                    log::warn!("Missing texture: {name}");
                    regions.insert(name.to_string(), missing_region);
                    continue;
                }
            };

            let origin = tile_origin(slot, grid_size, tile_size);
            let region = tile_region(origin, tile_size, atlas_size);

            let img_width = pixels.width.min(tile_size);
            let img_height = pixels.height.min(tile_size);
            for py in 0..img_height {
                for px in 0..img_width {
                    let src = ((py * pixels.width + px) * 4) as usize;
                    let dst = (((origin.1 + py) * atlas_size + origin.0 + px) * 4) as usize;
                    atlas_pixels[dst..dst + 4].copy_from_slice(&pixels.data[src..src + 4]);
                }
            }

            regions.insert(name.to_string(), region);
            slot += 1;
        }

        let uv_map = AtlasUVMap {
            regions,
            missing: missing_region,
        };

        let (image, view, allocation) =
            create_atlas_image(device, allocator, atlas_size)?;
        let (staging_buffer, staging_allocation) =
            create_staging_buffer(device, allocator, &atlas_pixels)?;

        upload_atlas(device, queue, command_pool, staging_buffer, image, atlas_size)?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        log::info!("Atlas built: {atlas_size}x{atlas_size}, {slot} textures");

        Ok(Self {
            image,
            view,
            sampler,
            uv_map,
            allocation: Some(allocation),
            staging_buffer,
            staging_allocation: Some(staging_allocation),
        })
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
        }
        if let Some(alloc) = self.allocation.take() {
            allocator.lock().unwrap().free(alloc).ok();
        }
        unsafe { device.destroy_image(self.image, None); }

        if let Some(alloc) = self.staging_allocation.take() {
            allocator.lock().unwrap().free(alloc).ok();
        }
        unsafe { device.destroy_buffer(self.staging_buffer, None); }
    }
}

struct PngPixels {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

fn load_png(path: &Path) -> Option<PngPixels> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let data = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => {
            let pixels = info.width as usize * info.height as usize;
            let mut rgba = Vec::with_capacity(pixels * 4);
            for chunk in buf[..pixels * 3].chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        png::ColorType::GrayscaleAlpha => {
            let pixels = info.width as usize * info.height as usize;
            let mut rgba = Vec::with_capacity(pixels * 4);
            for chunk in buf[..pixels * 2].chunks_exact(2) {
                rgba.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
            }
            rgba
        }
        png::ColorType::Grayscale => {
            let pixels = info.width as usize * info.height as usize;
            let mut rgba = Vec::with_capacity(pixels * 4);
            for &g in &buf[..pixels] {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
            rgba
        }
        png::ColorType::Indexed => {
            log::warn!("Indexed PNG not supported: {}", path.display());
            return None;
        }
    };

    Some(PngPixels {
        data,
        width: info.width,
        height: info.height,
    })
}

fn tile_origin(slot: u32, grid_size: u32, tile_size: u32) -> (u32, u32) {
    (
        (slot % grid_size) * tile_size,
        (slot / grid_size) * tile_size,
    )
}

fn tile_region(origin: (u32, u32), tile_size: u32, atlas_size: u32) -> AtlasRegion {
    let s = atlas_size as f32;
    AtlasRegion {
        u_min: origin.0 as f32 / s,
        v_min: origin.1 as f32 / s,
        u_max: (origin.0 + tile_size) as f32 / s,
        v_max: (origin.1 + tile_size) as f32 / s,
    }
}

fn create_atlas_image(
    device: &ash::Device,
    allocator: &Arc<Mutex<Allocator>>,
    size: u32,
) -> Result<(vk::Image, vk::ImageView, Allocation), vk::Result> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_SRGB)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);

    let image = unsafe { device.create_image(&image_info, None)? };
    let mem_reqs = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator
        .lock()
        .unwrap()
        .allocate(&AllocationCreateDesc {
            name: "atlas_image",
            requirements: mem_reqs,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .expect("failed to allocate atlas image memory");

    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_SRGB)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_info, None)? };

    Ok((image, view, allocation))
}

fn create_staging_buffer(
    device: &ash::Device,
    allocator: &Arc<Mutex<Allocator>>,
    data: &[u8],
) -> Result<(vk::Buffer, Allocation), vk::Result> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(data.len() as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut allocation = allocator
        .lock()
        .unwrap()
        .allocate(&AllocationCreateDesc {
            name: "atlas_staging",
            requirements: mem_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .expect("failed to allocate staging buffer memory");

    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };

    allocation.mapped_slice_mut().unwrap()[..data.len()].copy_from_slice(data);

    Ok((buffer, allocation))
}

fn upload_atlas(
    device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    staging_buffer: vk::Buffer,
    image: vk::Image,
    size: u32,
) -> Result<(), vk::Result> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info)? };

    let barrier_to_transfer = vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_to_transfer],
        );
    }

    let copy_region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        },
    };

    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
    }

    let barrier_to_shader = vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_to_shader],
        );

        device.end_command_buffer(cmd)?;

        let cmd_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, &[cmd]);
    }

    Ok(())
}
