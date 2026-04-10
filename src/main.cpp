// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary

// vulkan-interop-physx: ovphysx physics simulation driving ovrtx rendering.
//
// Each frame:
//   1. Step ovphysx (CPU mode) by a fixed 1/60 s timestep.
//   2. Read rigid-body world poses from ovphysx via tensor binding.
//   3. Push the poses to ovrtx with ovrtx_set_xform_pos_rot_scale so the
//      rendered scene matches the simulation state.
//   4. Step ovrtx and present via the usual Vulkan/CUDA interop path from
//      the vulkan-interop sample.

#include "camera/orbit_camera.hpp"
#include "cuda/cuda_kernel.hpp"
#include "glsl/spirv_loader.hpp"
#include "vk/vulkan_context.hpp"
#include <cuda.h>
#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_attributes.h>
#include <ovrtx/ovrtx_config.h>
#include <ovphysx/ovphysx.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/gtc/type_ptr.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// Window dimensions
constexpr int WINDOW_WIDTH = 1920;
constexpr int WINDOW_HEIGHT = 1080;

// Mouse state for orbit camera
static bool g_mouse_pressed = false;
static double g_last_mouse_x = 0.0;
static double g_last_mouse_y = 0.0;
static bool g_camera_dirty = false;
static OrbitCamera* g_orbit_camera = nullptr;

// Scene units for distance scaling
enum class Units { Centimeters, Meters };

// Default scene is the local physics_boxes.usda (found next to executable at runtime)
// Can be overridden with --usd
static constexpr char const* DEFAULT_RENDER_PRODUCT_PATH = "/Render/Camera";
static constexpr UpAxis DEFAULT_UP_AXIS = UpAxis::Y;
static constexpr Units DEFAULT_SCENE_UNITS = Units::Meters;

// Rigid-body prim paths shared between ovphysx tensor binding and ovrtx writes.
// Must match the prims in physics_boxes.usda that carry PhysicsRigidBodyAPI.
static constexpr int NUM_PHYSICS_BODIES = 3;
static constexpr char const* BODY_PRIM_PATHS[NUM_PHYSICS_BODIES] = {
    "/World/Cube1",
    "/World/Cube2",
    "/World/Sphere1",
};

// Kinematic "character" rigid body path - used for both physics tensor writes and visual updates.
static constexpr char const* CHAR_PRIM_PATH = "/World/Character";

static auto get_executable_dir() -> std::filesystem::path;
static void
mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
static void
cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
static void print_usage(char const* program_name);
static bool parse_args(int argc,
                       char* argv[],
                       std::string& usd_file,
                       std::string& render_product,
                       UpAxis& up_axis,
                       Units& units,
                       int& num_frames);
template <typename ResultT>
static bool check_and_print_error(ResultT const& result,
                                  std::string_view operation);

enum class OutputType { HdrColor, LdrColor };

static auto vulkan_format_for_output(OutputType type) -> VkFormat;
static auto cuda_format_for_output(OutputType type) -> CudaImageFormat;
static auto output_type_name(OutputType type) -> char const*;
static void print_frame_time_stats(int total_frames,
                                   std::vector<double> const& cpu_times_ms,
                                   std::vector<double> const& vulkan_times_ms,
                                   std::vector<double> const& cuda_times_ms);
static void record_rendering_state(CommandBuffer& cmd,
                                   VulkanContext& vk,
                                   uint32_t swapchain_image_index,
                                   ShaderHandle vert_shader,
                                   ShaderHandle frag_shader,
                                   SampledImageHandle read_image_handle);
static auto find_color_output(ovrtx_render_product_set_outputs_t const& outputs,
                              OutputType& output_type)
    -> ovrtx_rendered_output_handle_t;

// =========================================================================
// ovphysx helpers
// =========================================================================

struct PhysicsPoseBuffer {
    DLTensor tensor;
    float*   data;
    int64_t  shape[2];
};

static PhysicsPoseBuffer make_physics_pose_buffer(size_t n_bodies) {
    PhysicsPoseBuffer b{};
    b.data = new float[n_bodies * 7]();
    b.shape[0] = static_cast<int64_t>(n_bodies);
    b.shape[1] = 7;
    b.tensor.data              = b.data;
    b.tensor.ndim              = 2;
    b.tensor.shape             = b.shape;
    b.tensor.strides           = nullptr;
    b.tensor.byte_offset       = 0;
    b.tensor.dtype.code        = kDLFloat;
    b.tensor.dtype.bits        = 32;
    b.tensor.dtype.lanes       = 1;
    b.tensor.device.device_type = kDLCPU;
    b.tensor.device.device_id  = 0;
    return b;
}

static bool phys_wait_op(ovphysx_handle_t handle,
                         ovphysx_enqueue_result_t res,
                         char const* context) {
    if (res.status != OVPHYSX_API_SUCCESS) {
        std::cerr << "Physics enqueue failed in " << context << " (status="
                  << res.status << ")\n";
        if (res.error.ptr && res.error.length > 0) {
            std::cerr << "  " << std::string_view(res.error.ptr, res.error.length) << "\n";
            ovphysx_destroy_error(res.error);
        }
        return false;
    }
    ovphysx_op_wait_result_t wait{};
    ovphysx_result_t wr = ovphysx_wait_op(handle, res.op_index,
                                           10ULL * 1000 * 1000 * 1000, &wait);
    if (wait.num_errors > 0) {
        std::cerr << "Physics op failed in " << context << "\n";
        ovphysx_destroy_errors(wait.errors, wait.num_errors);
        return false;
    }
    if (wr.status != OVPHYSX_API_SUCCESS) {
        std::cerr << "Physics wait failed in " << context << "\n";
        if (wr.error.ptr) ovphysx_destroy_error(wr.error);
        return false;
    }
    return true;
}

// Write a vector3f attribute to a set of prims via ovrtx.
// Used to drive physxCharacterController:moveTarget each frame.
static void write_vec3f_attribute(ovrtx_renderer_t*   renderer,
                                  ovx_string_t const* paths, size_t n,
                                  ovx_string_t        attr_name,
                                  float const*        xyz3_per_prim) {
    DLDataType type;
    type.code  = kDLFloat;
    type.bits  = 32;
    type.lanes = 3;   // packed float3 per element

    DLTensor tensor = ovrtx_make_write_cpu_tensor(xyz3_per_prim, &n, type);

    ovrtx_input_buffer_t buffer{};
    buffer.tensors      = &tensor;
    buffer.tensor_count = 1;

    ovrtx_binding_desc_or_handle_t bd =
        ovrtx_make_binding_desc(paths, n, attr_name, OVRTX_SEMANTIC_NONE, type);

    ovrtx_write_attribute(renderer, &bd, &buffer, OVRTX_DATA_ACCESS_SYNC);
}

// =========================================================================
// main
// =========================================================================

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    std::string usd_file_path;
    std::string render_product_path;
    UpAxis up_axis;
    Units units;
    int num_frames = 0;

    if (!parse_args(argc, argv, usd_file_path, render_product_path,
                    up_axis, units, num_frames)) {
        return 0;
    }

    // Default USD: physics_boxes.usda next to the executable
    if (usd_file_path.empty()) {
        usd_file_path = (get_executable_dir() / "physics_boxes.usda").string();
    }

    std::cerr << "USD file: " << usd_file_path << "\n";
    std::cerr << "Render product: " << render_product_path << "\n";
    std::cerr << "Up axis: " << (up_axis == UpAxis::Y ? "Y" : "Z") << "\n";

    // =========================================================================
    // Initialize ovrtx
    // =========================================================================
    std::cerr << "Initializing ovrtx...\n";

    ovrtx_renderer_t* renderer = nullptr;
    GLFWwindow* window = nullptr;

    ovrtx_config_t ovrtx_config = {};
    ovrtx_result_t result = ovrtx_create_renderer(&ovrtx_config, &renderer);
    if (check_and_print_error(result, "create_renderer")) {
        return 1;
    }

    std::cerr << "Loading USD scene in ovrtx: " << usd_file_path << "\n";

    ovrtx_usd_input_t usd_input = {};
    usd_input.usd_file_path.ptr = usd_file_path.c_str();
    usd_input.usd_file_path.length = usd_file_path.size();

    ovx_string_t prefix = {};
    prefix.ptr = "";
    prefix.length = 0;

    ovrtx_usd_handle_t usd_handle = 0;
    ovrtx_enqueue_result_t enqueue_result =
        ovrtx_add_usd(renderer, usd_input, prefix, &usd_handle);
    if (check_and_print_error(enqueue_result, "add_usd")) {
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    ovrtx_op_wait_result_t wait_result = {};
    result = ovrtx_wait_op(renderer, enqueue_result.op_index,
                           ovrtx_timeout_infinite, &wait_result);
    if (wait_result.num_error_ops > 0) {
        for (int i = 0; i < wait_result.num_error_ops; ++i) {
            ovx_string_t err = ovrtx_get_last_op_error(wait_result.error_op_ids[i]);
            std::cerr << "ERROR: " << std::string_view(err.ptr, err.length) << "\n";
        }
        ovrtx_destroy_renderer(renderer);
        return 1;
    }
    if (check_and_print_error(result, "wait_op (add_usd)")) {
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    std::cerr << "USD scene loaded in ovrtx.\n";

    // =========================================================================
    // Initialize ovphysx (CPU mode) and load the same USD for physics
    // =========================================================================
    std::cerr << "Initializing ovphysx (CPU mode)...\n";

    ovphysx_handle_t phys_handle = 0;
    ovphysx_tensor_binding_handle_t phys_pose_binding = 0;
    ovphysx_tensor_binding_handle_t phys_char_binding = 0;
    PhysicsPoseBuffer phys_poses{};
    float phys_sim_time = 0.0f;
    static constexpr float PHYS_DT = 1.0f / 60.0f;

    {
        ovphysx_create_args phys_args = OVPHYSX_CREATE_ARGS_DEFAULT;
        phys_args.device = OVPHYSX_DEVICE_CPU;
        ovphysx_result_t pr = ovphysx_create_instance(&phys_args, &phys_handle);
        if (pr.status != OVPHYSX_API_SUCCESS) {
            std::cerr << "Failed to create ovphysx instance\n";
            if (pr.error.ptr) ovphysx_destroy_error(pr.error);
            ovrtx_destroy_renderer(renderer);
            return 1;
        }

        std::cerr << "Loading USD scene in ovphysx...\n";
        ovphysx_usd_handle_t phys_usd_handle = 0;
        ovphysx_enqueue_result_t pa = ovphysx_add_usd(
            phys_handle,
            ovphysx_cstr(usd_file_path.c_str()),
            ovphysx_cstr(""),
            &phys_usd_handle);
        if (!phys_wait_op(phys_handle, pa, "add_usd")) {
            ovphysx_destroy_instance(phys_handle);
            ovrtx_destroy_renderer(renderer);
            return 1;
        }

        // Create tensor binding for the three rigid bodies (explicit paths)
        ovphysx_string_t body_phys_paths[NUM_PHYSICS_BODIES];
        for (int i = 0; i < NUM_PHYSICS_BODIES; ++i) {
            body_phys_paths[i] = ovphysx_cstr(BODY_PRIM_PATHS[i]);
        }

        ovphysx_tensor_binding_desc_t phys_desc = {};
        phys_desc.prim_paths       = body_phys_paths;
        phys_desc.prim_paths_count = NUM_PHYSICS_BODIES;
        phys_desc.tensor_type      = OVPHYSX_TENSOR_RIGID_BODY_POSE_F32;

        pr = ovphysx_create_tensor_binding(phys_handle, &phys_desc, &phys_pose_binding);
        if (pr.status != OVPHYSX_API_SUCCESS) {
            std::cerr << "Failed to create physics tensor binding\n";
            if (pr.error.ptr && pr.error.length > 0) {
                std::cerr << "  " << std::string_view(pr.error.ptr, pr.error.length) << "\n";
                ovphysx_destroy_error(pr.error);
            }
            ovphysx_destroy_instance(phys_handle);
            ovrtx_destroy_renderer(renderer);
            return 1;
        }

        ovphysx_tensor_spec_t phys_spec{};
        pr = ovphysx_get_tensor_binding_spec(phys_handle, phys_pose_binding, &phys_spec);
        if (pr.status != OVPHYSX_API_SUCCESS) {
            std::cerr << "Failed to get physics tensor spec\n";
            if (pr.error.ptr) ovphysx_destroy_error(pr.error);
            ovphysx_destroy_instance(phys_handle);
            ovrtx_destroy_renderer(renderer);
            return 1;
        }
        std::cerr << "Physics tensor binding: " << phys_spec.shape[0]
                  << " rigid bodies.\n";

        phys_poses = make_physics_pose_buffer(static_cast<size_t>(phys_spec.shape[0]));
        // Fix self-referential pointer after struct copy (shape array is inline)
        phys_poses.tensor.shape = phys_poses.shape;

        // Create a write binding for the character body velocity
        ovphysx_string_t char_phys_path = ovphysx_cstr(CHAR_PRIM_PATH);
        ovphysx_tensor_binding_desc_t char_desc = {};
        char_desc.prim_paths       = &char_phys_path;
        char_desc.prim_paths_count = 1;
        char_desc.tensor_type      = OVPHYSX_TENSOR_RIGID_BODY_VELOCITY_F32;
        pr = ovphysx_create_tensor_binding(phys_handle, &char_desc, &phys_char_binding);
        if (pr.status != OVPHYSX_API_SUCCESS) {
            std::cerr << "Failed to create character tensor binding\n";
            if (pr.error.ptr) ovphysx_destroy_error(pr.error);
            ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
            ovphysx_destroy_instance(phys_handle);
            ovrtx_destroy_renderer(renderer);
            return 1;
        }
        std::cerr << "Character kinematic binding created.\n";
    }

    // Build ovx_string_t prim paths for ovrtx transform writes (same order as tensor binding)
    ovx_string_t body_ovrtx_paths[NUM_PHYSICS_BODIES];
    for (int i = 0; i < NUM_PHYSICS_BODIES; ++i) {
        body_ovrtx_paths[i].ptr    = BODY_PRIM_PATHS[i];
        body_ovrtx_paths[i].length = strlen(BODY_PRIM_PATHS[i]);
    }

    // Persistent velocity buffer for the character body [vx vy vz wx wy wz]
    // Declared here so it stays valid between ovphysx_write_tensor_binding and step completion.
    float char_vel_data[6]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int64_t char_vel_shape[2] = {1, 6};
    DLTensor char_vel_tensor  = {};
    char_vel_tensor.data              = char_vel_data;
    char_vel_tensor.device.device_type = kDLCPU;
    char_vel_tensor.device.device_id  = 0;
    char_vel_tensor.ndim              = 2;
    char_vel_tensor.dtype.code        = kDLFloat;
    char_vel_tensor.dtype.bits        = 32;
    char_vel_tensor.dtype.lanes       = 1;
    char_vel_tensor.shape             = char_vel_shape;
    char_vel_tensor.strides           = nullptr; // row-major
    char_vel_tensor.byte_offset       = 0;

    // ovx path for visual transform writes
    ovx_string_t char_visual_path = { CHAR_PRIM_PATH, strlen(CHAR_PRIM_PATH) };

    // =========================================================================
    // CUDA init (uses the CUDA context already established by ovrtx)
    // =========================================================================
    CUuuid cuda_uuid;
    if (!cuda_init(&cuda_uuid)) {
        std::cerr << "Failed to get CUDA context\n";
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    // =========================================================================
    // Initial ovrtx step to get render dimensions
    // =========================================================================
    ovx_string_t render_product_str = {};
    render_product_str.ptr    = render_product_path.c_str();
    render_product_str.length = render_product_path.size();

    ovrtx_render_product_set_t render_products = {};
    render_products.render_products     = &render_product_str;
    render_products.num_render_products = 1;

    ovrtx_step_result_handle_t step_result_handle = 0;
    enqueue_result = ovrtx_step(renderer, render_products, 0.0, &step_result_handle);
    if (check_and_print_error(enqueue_result, "step (probe)")) {
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    ovrtx_render_product_set_outputs_t outputs = {};
    result = ovrtx_fetch_results(renderer, step_result_handle,
                                 ovrtx_timeout_infinite, &outputs);
    if (check_and_print_error(result, "fetch_results (probe)")) {
        ovrtx_destroy_results(renderer, step_result_handle);
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    if (outputs.status != OVRTX_EVENT_COMPLETED || outputs.output_count == 0) {
        std::cerr << "Could not get dimensions for RenderProduct \""
                  << render_product_path << "\"\n";
        ovrtx_destroy_results(renderer, step_result_handle);
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 3;
    }

    OutputType output_type;
    ovrtx_rendered_output_handle_t color_output_handle =
        find_color_output(outputs, output_type);
    if (color_output_handle == 0) {
        std::cerr << "No color output found\n";
        ovrtx_destroy_results(renderer, step_result_handle);
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }
    std::cerr << "Using " << output_type_name(output_type) << " output\n";

    ovrtx_map_output_description_t map_desc = {};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY;
    map_desc.sync_stream = 0;

    ovrtx_rendered_output_t rendered_output = {};
    result = ovrtx_map_rendered_output(renderer, color_output_handle, &map_desc,
                                       ovrtx_timeout_infinite, &rendered_output);
    if (check_and_print_error(result, "map_rendered_output (probe)")) {
        ovrtx_destroy_results(renderer, step_result_handle);
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    DLTensor const& dl = rendered_output.buffer.dl;
    int tex_width  = static_cast<int>(dl.shape[1]);
    int tex_height = static_cast<int>(dl.shape[0]);

    result = ovrtx_unmap_rendered_output(renderer, rendered_output.map_handle,
                                         ovrtx_cuda_sync_t{});
    if (check_and_print_error(result, "unmap_rendered_output (probe)")) {
        ovrtx_destroy_results(renderer, step_result_handle);
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }
    result = ovrtx_destroy_results(renderer, step_result_handle);
    if (check_and_print_error(result, "destroy_results (probe)")) {
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    std::cerr << "Render output dimensions: " << tex_width << " x " << tex_height << "\n";

    // =========================================================================
    // GLFW window and Vulkan setup
    // =========================================================================
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                               "ovrtx-interop-physx: Falling Boxes", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Orbit camera: Y-up, looking at (0,4,0), 17m back along +Z, 25° up
    float unit_scale = (units == Units::Centimeters) ? 100.0f : 1.0f;
    float distance   = 17.0f * unit_scale;
    float azimuth    = glm::radians(90.0f);   // camera on +Z axis
    float elevation  = glm::radians(25.0f);   // 25° above horizontal
    glm::vec3 target(0.0f, 4.0f * unit_scale, 0.0f);
    OrbitCamera orbit_camera(distance, azimuth, elevation, target, up_axis);
    g_orbit_camera = &orbit_camera;
    g_camera_dirty = true;

    try {
        VulkanContextConfig vk_config;
        vk_config.window = window;
        vk_config.initial_sampled_image_capacity = 16;

        VulkanContext vk(vk_config, cuda_uuid);

        constexpr int SHARED_IMAGE_COUNT = 2;
        VkFormat vk_format       = vulkan_format_for_output(output_type);
        CudaImageFormat cuda_fmt = cuda_format_for_output(output_type);

        SampledImageHandle shared_images[SHARED_IMAGE_COUNT];
        for (int i = 0; i < SHARED_IMAGE_COUNT; ++i) {
            shared_images[i] = vk.create_sampled_image(
                tex_width, tex_height, vk_format, VK_FILTER_LINEAR, true);
        }

        std::filesystem::path shader_dir = get_executable_dir() / "shaders";
        std::string vert_path = (shader_dir / "fullscreen.vert.spv").string();
        std::string frag_path = (shader_dir / "fullscreen.frag.spv").string();
        std::vector<uint32_t> vert_spirv = load_spirv(vert_path);
        std::vector<uint32_t> frag_spirv = load_spirv(frag_path);

        auto [vert_shader, frag_shader] =
            vk.create_linked_vertex_and_fragment_shaders(vert_spirv, frag_spirv);

        for (int i = 0; i < SHARED_IMAGE_COUNT; ++i) {
            VkImage img = vk.sampled_image(shared_images[i]).image;
            vk.immediate_submit([img](CommandBuffer cmd) {
                cmd.image_memory_barrier(img,
                                         VK_IMAGE_ASPECT_COLOR_BIT,
                                         VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                         VK_ACCESS_2_NONE,
                                         VK_IMAGE_LAYOUT_GENERAL,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_WRITE_BIT);
            });
        }

        CUsurfObject cuda_surfaces[SHARED_IMAGE_COUNT];
        for (int i = 0; i < SHARED_IMAGE_COUNT; ++i) {
            SampledImage const& img = vk.sampled_image(shared_images[i]);
            auto memory_handle      = vk.export_memory_handle(shared_images[i]);
            cuda_surfaces[i]        = cuda_import_vulkan_image(
                i, memory_handle, img.size, tex_width, tex_height, cuda_fmt);
            if (cuda_surfaces[i] == 0) {
                std::cerr << "Failed to import Vulkan image " << i << " into CUDA\n";
                cuda_cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                ovphysx_destroy_instance(phys_handle);
                ovrtx_destroy_renderer(renderer);
                return 1;
            }
        }

        auto timeline_handle = vk.export_timeline_semaphore_handle();
        cuda_import_timeline_semaphore(timeline_handle);

        CUstream cuda_stream;
        cuStreamCreate(&cuda_stream, 0);

        CUevent cuda_start_event, cuda_end_event, cuda_frame_done_event,
            cuda_copy_done_event;
        cuEventCreate(&cuda_start_event, CU_EVENT_DEFAULT);
        cuEventCreate(&cuda_end_event, CU_EVENT_DEFAULT);
        cuEventCreate(&cuda_frame_done_event, CU_EVENT_DISABLE_TIMING);
        cuEventCreate(&cuda_copy_done_event, CU_EVENT_DISABLE_TIMING);

        int      write_idx = 0;
        int      read_idx  = 0;
        uint64_t cuda_frame_counter   = 0;
        uint64_t read_timeline_value  = 0;
        bool     cuda_work_pending    = false;

        ovrtx_step_result_handle_t         current_step_result = 0;
        ovrtx_rendered_output_map_handle_t current_map_handle  = 0;
        bool has_mapped_output = false;

        // =====================================================================
        // Prime first buffer
        // =====================================================================
        std::cerr << "Priming first frame...\n";
        {
            enqueue_result = ovrtx_step(renderer, render_products, 0.0,
                                        &current_step_result);
            if (check_and_print_error(enqueue_result, "step (prime)")) {
                cuda_cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                ovphysx_destroy_instance(phys_handle);
                ovrtx_destroy_renderer(renderer);
                return 1;
            }

            result = ovrtx_fetch_results(renderer, current_step_result,
                                         ovrtx_timeout_infinite, &outputs);
            if (check_and_print_error(result, "fetch_results (prime)")) {
                cuda_cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                ovphysx_destroy_instance(phys_handle);
                ovrtx_destroy_renderer(renderer);
                return 1;
            }

            OutputType frame_output_type;
            color_output_handle = find_color_output(outputs, frame_output_type);
            if (color_output_handle == OVRTX_INVALID_HANDLE) {
                std::cerr << "ERROR: could not find output from "
                          << render_product_path << "\n";
                ovrtx_destroy_results(renderer, current_step_result);
                cuda_cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                ovphysx_destroy_instance(phys_handle);
                ovrtx_destroy_renderer(renderer);
                return 2;
            }

            result = ovrtx_map_rendered_output(renderer, color_output_handle,
                                               &map_desc, ovrtx_timeout_infinite,
                                               &rendered_output);
            if (check_and_print_error(result, "map_rendered_output (prime)")) {
                ovrtx_destroy_results(renderer, current_step_result);
                cuda_cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                ovphysx_destroy_instance(phys_handle);
                ovrtx_destroy_renderer(renderer);
                return 1;
            }

            current_map_handle = rendered_output.map_handle;
            has_mapped_output  = true;

            CUarray cuda_array = reinterpret_cast<CUarray>(rendered_output.buffer.dl.data);
            CUevent wait_event = reinterpret_cast<CUevent>(
                rendered_output.buffer.cuda_sync.wait_event);
            int out_w = static_cast<int>(rendered_output.buffer.dl.shape[1]);
            int out_h = static_cast<int>(rendered_output.buffer.dl.shape[0]);

            if (wait_event) {
                cuda_wait_event(wait_event, cuda_stream);
            }
            cuda_copy_array_to_surface(0, cuda_array, out_w, out_h, cuda_fmt, cuda_stream);
            cuEventRecord(cuda_copy_done_event, cuda_stream);
            cuStreamSynchronize(cuda_stream);

            ovrtx_cuda_sync_t copy_done_sync = {};
            copy_done_sync.wait_event = reinterpret_cast<uintptr_t>(cuda_copy_done_event);
            result = ovrtx_unmap_rendered_output(renderer, current_map_handle,
                                                 copy_done_sync);
            if (check_and_print_error(result, "unmap_rendered_output (prime)")) {
                ovrtx_destroy_results(renderer, current_step_result);
                cuda_cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                ovphysx_destroy_instance(phys_handle);
                ovrtx_destroy_renderer(renderer);
                return 1;
            }
            result = ovrtx_destroy_results(renderer, current_step_result);
            if (check_and_print_error(result, "destroy_results (prime)")) {
                cuda_cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                ovphysx_destroy_instance(phys_handle);
                ovrtx_destroy_renderer(renderer);
                return 1;
            }
            has_mapped_output = false;

            read_idx  = 0;
            write_idx = 1;
        }

        // Timing
        double accumulated_cpu_ms    = 0.0;
        double accumulated_vulkan_ms = 0.0;
        double accumulated_cuda_ms   = 0.0;
        int    frame_count      = 0;
        int    cuda_frame_count = 0;
        int    swaps_this_second = 0;
        bool   defer_swapchain_recreate = false;
        auto   last_print_time = std::chrono::steady_clock::now();
        auto   last_step_time  = std::chrono::steady_clock::now();

        std::vector<double> all_cpu_times_ms, all_vulkan_times_ms, all_cuda_times_ms;
        if (num_frames > 0) {
            all_cpu_times_ms.reserve(num_frames);
            all_vulkan_times_ms.reserve(num_frames);
            all_cuda_times_ms.reserve(num_frames);
        }

        std::cerr << "Starting render loop (physics + rendering)...\n";
        if (num_frames > 0) {
            std::cerr << "Will render " << num_frames
                      << " frames then save out.png and exit\n";
        }

        int total_frames = 0;

        // =====================================================================
        // Main loop
        // =====================================================================
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            auto size = vk.framebuffer_size();
            if (size.x == 0 || size.y == 0) {
                glfwWaitEvents();
                continue;
            }

            auto frame_start = std::chrono::steady_clock::now();

            vk.wait_for_fence();

            if (frame_count > 0) {
                double vulkan_ms = vk.vulkan_elapsed_ms();
                accumulated_vulkan_ms += vulkan_ms;
                if (num_frames > 0) all_vulkan_times_ms.push_back(vulkan_ms);
            }

            if (defer_swapchain_recreate) {
                vk.recreate_swapchain();
                defer_swapchain_recreate = false;
                continue;
            }

            vk.reset_fence();

            uint32_t swapchain_image_index;
            auto acquire_result = vk.acquire_next_image(swapchain_image_index);

            if (acquire_result == AcquireResult::OutOfDate) {
                vk.recreate_swapchain();
                vk.reset_fence_to_signaled();
                continue;
            } else if (acquire_result == AcquireResult::Suboptimal) {
                defer_swapchain_recreate = true;
            } else if (acquire_result == AcquireResult::Minimized) {
                vk.reset_fence_to_signaled();
                continue;
            }

            if (cuda_work_pending) {
                CUresult event_status = cuEventQuery(cuda_frame_done_event);
                if (event_status == CUDA_SUCCESS) {
                    float cuda_elapsed_ms = 0.0f;
                    cuEventElapsedTime(&cuda_elapsed_ms, cuda_start_event, cuda_end_event);
                    accumulated_cuda_ms += cuda_elapsed_ms;
                    cuda_frame_count++;
                    if (num_frames > 0) all_cuda_times_ms.push_back(cuda_elapsed_ms);

                    read_timeline_value = cuda_frame_counter;
                    std::swap(read_idx, write_idx);
                    cuda_work_pending = false;
                    swaps_this_second++;
                }
            }

            // Camera update
            if (g_camera_dirty) {
                ovx_string_t prim_path_str = {};
                prim_path_str.ptr    = "/World/Camera";
                prim_path_str.length = strlen("/World/Camera");

                ovrtx_prim_list_t prim_list = {};
                prim_list.prim_paths = &prim_path_str;
                prim_list.num_paths  = 1;

                ovrtx_attribute_type_t attr_type = {};
                attr_type.dtype.code  = kDLFloat;
                attr_type.dtype.bits  = 64;
                attr_type.dtype.lanes = 16;
                attr_type.is_array    = false;
                attr_type.semantic    = OVRTX_SEMANTIC_XFORM_MAT4x4;

                ovrtx_binding_desc_t binding = {};
                binding.prim_list = prim_list;
                binding.attribute_name.string.ptr    = "omni:xform";
                binding.attribute_name.string.length = strlen("omni:xform");
                binding.attribute_type = attr_type;
                binding.prim_mode      = OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY;
                binding.flags          = OVRTX_BINDING_FLAG_NONE;

                ovrtx_binding_desc_or_handle_t binding_desc_or_handle = {};
                binding_desc_or_handle.binding_desc   = binding;
                binding_desc_or_handle.binding_handle = 0;

                glm::mat4  transform   = orbit_camera.transform_matrix();
                double     transform_data[16];
                float const* src = glm::value_ptr(transform);
                for (int i = 0; i < 16; ++i) {
                    transform_data[i] = static_cast<double>(src[i]);
                }

                DLTensor transform_dl = {};
                transform_dl.data              = transform_data;
                transform_dl.device.device_type = kDLCPU;
                transform_dl.device.device_id   = 0;
                transform_dl.ndim              = 1;
                int64_t shape[1]               = {1};
                transform_dl.shape             = shape;
                transform_dl.strides           = nullptr;
                transform_dl.byte_offset       = 0;
                transform_dl.dtype.code        = kDLFloat;
                transform_dl.dtype.bits        = 64;
                transform_dl.dtype.lanes       = 16;

                ovrtx_input_buffer_t input_buffer = {};
                input_buffer.tensors      = &transform_dl;
                input_buffer.tensor_count = 1;

                enqueue_result = ovrtx_write_attribute(renderer,
                                                       &binding_desc_or_handle,
                                                       &input_buffer,
                                                       OVRTX_DATA_ACCESS_SYNC);
                if (check_and_print_error(enqueue_result, "write_attribute (camera)")) {
                    cuda_cleanup();
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                    ovphysx_destroy_instance(phys_handle);
                    ovrtx_destroy_renderer(renderer);
                    return 1;
                }
                g_camera_dirty = false;
            }

            if (!cuda_work_pending) {
                cuEventRecord(cuda_start_event, cuda_stream);

                // ---------------------------------------------------------
                // Physics: advance simulation by one fixed step, then push
                // the resulting poses to ovrtx so the next rendered frame
                // reflects the current simulation state.
                // ---------------------------------------------------------
                {
                    // Drive character back and forth in X by setting its velocity.
                    // vx = d/dt[-5*cos(0.5t)] = 2.5*sin(0.5t). vy=0 to stay grounded.
                    float char_x = -5.0f * std::cos(phys_sim_time * 0.5f);
                    char_vel_data[0] = 2.5f * std::sin(phys_sim_time * 0.5f); // vx
                    char_vel_data[1] = 0.0f; // vy - suppress gravity accumulation
                    ovphysx_result_t cwr =
                        ovphysx_write_tensor_binding(phys_handle, phys_char_binding,
                                                     &char_vel_tensor, nullptr);
                    if (cwr.status != OVPHYSX_API_SUCCESS) {
                        std::cerr << "Failed to write character velocity\n";
                        if (cwr.error.ptr) ovphysx_destroy_error(cwr.error);
                    }

                    ovphysx_enqueue_result_t ps =
                        ovphysx_step(phys_handle, PHYS_DT, phys_sim_time);
                    phys_sim_time += PHYS_DT;

                    if (!phys_wait_op(phys_handle, ps, "step")) {
                        cuda_cleanup();
                        glfwDestroyWindow(window);
                        glfwTerminate();
                        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                        ovphysx_destroy_instance(phys_handle);
                        ovrtx_destroy_renderer(renderer);
                        return 1;
                    }

                    ovphysx_result_t pr = ovphysx_read_tensor_binding(
                        phys_handle, phys_pose_binding, &phys_poses.tensor);
                    if (pr.status != OVPHYSX_API_SUCCESS) {
                        std::cerr << "Failed to read physics poses\n";
                        if (pr.error.ptr && pr.error.length > 0) {
                            std::cerr << "  "
                                      << std::string_view(pr.error.ptr, pr.error.length)
                                      << "\n";
                            ovphysx_destroy_error(pr.error);
                        }
                        cuda_cleanup();
                        glfwDestroyWindow(window);
                        glfwTerminate();
                        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                        ovphysx_destroy_instance(phys_handle);
                        ovrtx_destroy_renderer(renderer);
                        return 1;
                    }

                    // Convert [px py pz qx qy qz qw] float32 poses to the
                    // ovrtx pos3d+rot4f+scale3f format and push to the scene.
                    ovrtx_xform_pos3d_rot4f_scale3f_t body_xforms[NUM_PHYSICS_BODIES];
                    for (int i = 0; i < NUM_PHYSICS_BODIES; ++i) {
                        float const* p = phys_poses.data + i * 7;
                        body_xforms[i].position[0]     = static_cast<double>(p[0]);
                        body_xforms[i].position[1]     = static_cast<double>(p[1]);
                        body_xforms[i].position[2]     = static_cast<double>(p[2]);
                        body_xforms[i].rot_quat_xyzw[0] = p[3]; // qx
                        body_xforms[i].rot_quat_xyzw[1] = p[4]; // qy
                        body_xforms[i].rot_quat_xyzw[2] = p[5]; // qz
                        body_xforms[i].rot_quat_xyzw[3] = p[6]; // qw
                        body_xforms[i].scale[0] = 1.0f;
                        body_xforms[i].scale[1] = 1.0f;
                        body_xforms[i].scale[2] = 1.0f;
                        body_xforms[i].padding  = 0;
                    }

                    // Write all three bodies in a single batched call
                    enqueue_result = ovrtx_set_xform_pos_rot_scale(
                        renderer, body_ovrtx_paths, NUM_PHYSICS_BODIES, body_xforms);
                    if (check_and_print_error(enqueue_result,
                                              "set_xform_pos_rot_scale (bodies)")) {
                        cuda_cleanup();
                        glfwDestroyWindow(window);
                        glfwTerminate();
                        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                        ovphysx_destroy_instance(phys_handle);
                        ovrtx_destroy_renderer(renderer);
                        return 1;
                    }

                    // Update character visual to match commanded pose.
                    ovrtx_xform_pos3d_rot4f_scale3f_t char_xform{};
                    char_xform.position[0]      = static_cast<double>(char_x);
                    char_xform.position[1]      = 1.0;
                    char_xform.position[2]      = 0.0;
                    char_xform.rot_quat_xyzw[3] = 1.0f; // identity
                    char_xform.scale[0] = char_xform.scale[1] = char_xform.scale[2] = 1.0f;
                    ovrtx_set_xform_pos_rot_scale(renderer, &char_visual_path, 1, &char_xform);
                }

                // ---------------------------------------------------------
                // ovrtx render step
                // ---------------------------------------------------------
                auto now = std::chrono::steady_clock::now();
                double delta_time =
                    std::chrono::duration<double>(now - last_step_time).count();
                last_step_time = now;

                enqueue_result = ovrtx_step(renderer, render_products, delta_time,
                                            &current_step_result);
                if (check_and_print_error(enqueue_result, "step")) {
                    cuda_cleanup();
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                    ovphysx_destroy_instance(phys_handle);
                    ovrtx_destroy_renderer(renderer);
                    return 1;
                }

                result = ovrtx_fetch_results(renderer, current_step_result,
                                             ovrtx_timeout_infinite, &outputs);
                if (check_and_print_error(result, "fetch_results")) {
                    cuda_cleanup();
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                    ovphysx_destroy_instance(phys_handle);
                    ovrtx_destroy_renderer(renderer);
                    return 1;
                }

                OutputType frame_output_type;
                color_output_handle = find_color_output(outputs, frame_output_type);

                result = ovrtx_map_rendered_output(renderer, color_output_handle,
                                                   &map_desc, ovrtx_timeout_infinite,
                                                   &rendered_output);
                if (check_and_print_error(result, "map_rendered_output")) {
                    ovrtx_destroy_results(renderer, current_step_result);
                    cuda_cleanup();
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                    ovphysx_destroy_instance(phys_handle);
                    ovrtx_destroy_renderer(renderer);
                    return 1;
                }

                current_map_handle = rendered_output.map_handle;
                has_mapped_output  = true;

                CUarray cuda_array = reinterpret_cast<CUarray>(
                    rendered_output.buffer.dl.data);
                CUevent wait_event = reinterpret_cast<CUevent>(
                    rendered_output.buffer.cuda_sync.wait_event);
                int out_w = static_cast<int>(rendered_output.buffer.dl.shape[1]);
                int out_h = static_cast<int>(rendered_output.buffer.dl.shape[0]);

                if (wait_event) {
                    cuda_wait_event(wait_event, cuda_stream);
                }

                cuda_copy_array_to_surface(write_idx, cuda_array, out_w, out_h,
                                           cuda_fmt, cuda_stream);
                cuEventRecord(cuda_copy_done_event, cuda_stream);

                ovrtx_cuda_sync_t copy_done_sync = {};
                copy_done_sync.wait_event =
                    reinterpret_cast<uintptr_t>(cuda_copy_done_event);
                result = ovrtx_unmap_rendered_output(renderer, current_map_handle,
                                                     copy_done_sync);
                if (check_and_print_error(result, "unmap_rendered_output")) {
                    ovrtx_destroy_results(renderer, current_step_result);
                    cuda_cleanup();
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                    ovphysx_destroy_instance(phys_handle);
                    ovrtx_destroy_renderer(renderer);
                    return 1;
                }
                result = ovrtx_destroy_results(renderer, current_step_result);
                if (check_and_print_error(result, "destroy_results")) {
                    cuda_cleanup();
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
                    ovphysx_destroy_instance(phys_handle);
                    ovrtx_destroy_renderer(renderer);
                    return 1;
                }
                has_mapped_output = false;

                cuEventRecord(cuda_end_event, cuda_stream);
                cuEventRecord(cuda_frame_done_event, cuda_stream);
                cuda_frame_counter++;
                cuda_signal_timeline(cuda_frame_counter, cuda_stream);
                cuda_work_pending = true;
            }

            // Vulkan present
            CommandBuffer cmd = vk.command_buffer();
            cmd.begin();
            cmd.reset_query_pool(vk.timestamp_query_pool(), 0, 2);
            cmd.write_timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                vk.timestamp_query_pool(), 0);

            VkImage read_image = vk.sampled_image(shared_images[read_idx]).image;
            cmd.image_memory_barrier(read_image,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                     VK_QUEUE_FAMILY_EXTERNAL,
                                     vk.queue_family());

            cmd.image_memory_barrier(vk.swapchain_image(swapchain_image_index),
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

            record_rendering_state(cmd, vk, swapchain_image_index,
                                   vert_shader, frag_shader, shared_images[read_idx]);

            cmd.image_memory_barrier(vk.swapchain_image(swapchain_image_index),
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                     VK_ACCESS_2_NONE);

            cmd.image_memory_barrier(read_image,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     vk.queue_family(),
                                     VK_QUEUE_FAMILY_EXTERNAL);

            cmd.write_timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                vk.timestamp_query_pool(), 1);
            cmd.end();
            PresentResult present_result = vk.submit_and_present(
                swapchain_image_index, read_timeline_value);
            if (present_result == PresentResult::OutOfDate ||
                present_result == PresentResult::Suboptimal) {
                defer_swapchain_recreate = true;
            }

            auto   frame_end = std::chrono::steady_clock::now();
            double cpu_ms    = std::chrono::duration<double, std::milli>(
                                   frame_end - frame_start).count();
            accumulated_cpu_ms += cpu_ms;
            if (num_frames > 0) all_cpu_times_ms.push_back(cpu_ms);
            frame_count++;
            total_frames++;

            if (num_frames > 0 && total_frames >= num_frames) {
                break;
            }

            auto   now2           = std::chrono::steady_clock::now();
            double elapsed_seconds =
                std::chrono::duration<double>(now2 - last_print_time).count();
            if (elapsed_seconds >= 1.0) {
                double avg_cpu_ms    = accumulated_cpu_ms / frame_count;
                double avg_vulkan_ms = accumulated_vulkan_ms / frame_count;
                double avg_cuda_ms   = (cuda_frame_count > 0)
                                           ? accumulated_cuda_ms / cuda_frame_count
                                           : 0.0;
                std::cerr << "Avg (ms): CPU=" << avg_cpu_ms
                          << "  Vulkan=" << avg_vulkan_ms
                          << "  CUDA=" << avg_cuda_ms
                          << "  FPS=" << (frame_count / elapsed_seconds)
                          << "  phys_t=" << phys_sim_time << "s\n";

                accumulated_cpu_ms    = 0.0;
                accumulated_vulkan_ms = 0.0;
                accumulated_cuda_ms   = 0.0;
                frame_count      = 0;
                cuda_frame_count = 0;
                swaps_this_second = 0;
                last_print_time = now2;
            }
        }

        cuStreamSynchronize(cuda_stream);
        vk.wait_for_fence();

        if (num_frames > 0) {
            print_frame_time_stats(total_frames, all_cpu_times_ms,
                                   all_vulkan_times_ms, all_cuda_times_ms);
        }

        if (num_frames > 0) {
            std::cerr << "Reading back frame from buffer " << read_idx << "...\n";
            std::vector<uint8_t> pixels =
                cuda_read_surface_rgba8(read_idx, tex_width, tex_height,
                                        (output_type == OutputType::HdrColor)
                                            ? CudaImageFormat::Half4
                                            : CudaImageFormat::UInt8_4);
            int ok = stbi_write_png("out.png", tex_width, tex_height, 4,
                                    pixels.data(), tex_width * 4);
            if (ok) {
                std::cerr << "Saved out.png (" << tex_width << "x" << tex_height << ")\n";
            } else {
                std::cerr << "Failed to write out.png\n";
            }
        }

        cuEventDestroy(cuda_start_event);
        cuEventDestroy(cuda_end_event);
        cuEventDestroy(cuda_frame_done_event);
        cuEventDestroy(cuda_copy_done_event);
        cuStreamDestroy(cuda_stream);
        cuda_cleanup();

    } catch (std::exception const& e) {
        std::cerr << "Vulkan interop error: " << e.what() << "\n";
        g_orbit_camera = nullptr;
        glfwDestroyWindow(window);
        glfwTerminate();
        ovphysx_destroy_tensor_binding(phys_handle, phys_char_binding);
        ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
        delete[] phys_poses.data;
        ovphysx_destroy_instance(phys_handle);
        ovrtx_destroy_renderer(renderer);
        return 1;
    }

    g_orbit_camera = nullptr;

    glfwDestroyWindow(window);
    glfwTerminate();

    // Cleanup ovphysx
    ovphysx_destroy_tensor_binding(phys_handle, phys_char_binding);
    ovphysx_destroy_tensor_binding(phys_handle, phys_pose_binding);
    delete[] phys_poses.data;
    ovphysx_destroy_instance(phys_handle);

    ovrtx_destroy_renderer(renderer);

    std::cerr << "Done!\n";
    return 0;
}

// =========================================================================
// Utility functions (identical to vulkan-interop)
// =========================================================================

static auto get_executable_dir() -> std::filesystem::path {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len == 0 || len == MAX_PATH)
        throw std::runtime_error("GetModuleFileNameW failed");
    return std::filesystem::path(buf).parent_path();
#elif defined(__linux__)
    return std::filesystem::canonical("/proc/self/exe").parent_path();
#else
#error "Unsupported platform"
#endif
}

static void
mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    (void)window; (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_mouse_pressed = (action == GLFW_PRESS);
    }
}

static void
cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (g_mouse_pressed && g_orbit_camera) {
        float delta_x = static_cast<float>(xpos - g_last_mouse_x);
        float delta_y = static_cast<float>(ypos - g_last_mouse_y);
        g_orbit_camera->update(delta_x, delta_y);
        g_camera_dirty = true;
    }
    g_last_mouse_x = xpos;
    g_last_mouse_y = ypos;
}

static void
scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window; (void)xoffset;
    if (g_orbit_camera) {
        float const dolly_factor = 0.1f;
        float new_distance = g_orbit_camera->distance() *
                             (1.0f - static_cast<float>(yoffset) * dolly_factor);
        new_distance = std::max(new_distance, 0.1f);
        g_orbit_camera->set_distance(new_distance);
        g_camera_dirty = true;
    }
}

static void print_usage(char const* program_name) {
    std::cerr << "Usage: " << program_name << " [options]\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --usd, -u <path>           USD file (default: physics_boxes.usda next to executable)\n";
    std::cerr << "  --render-product, -r <path> Render product prim path (default: "
              << DEFAULT_RENDER_PRODUCT_PATH << ")\n";
    std::cerr << "  --up-axis, -a <Y|Z>         Scene up axis (default: Y)\n";
    std::cerr << "  --units <meters|centimeters> Scene units (default: meters)\n";
    std::cerr << "  --num-frames, -n <N>         Render N frames, save out.png and exit\n";
    std::cerr << "  --help, -h                   Show this help\n";
}

static bool parse_args(int argc, char* argv[],
                       std::string& usd_file,
                       std::string& render_product,
                       UpAxis& up_axis,
                       Units& units,
                       int& num_frames) {
    usd_file       = "";  // empty = use default (physics_boxes.usda)
    render_product = DEFAULT_RENDER_PRODUCT_PATH;
    up_axis        = DEFAULT_UP_AXIS;
    units          = DEFAULT_SCENE_UNITS;
    num_frames     = 0;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if ((arg == "--usd" || arg == "-u") && i + 1 < argc) {
            usd_file = argv[++i];
        } else if ((arg == "--render-product" || arg == "-r") && i + 1 < argc) {
            render_product = argv[++i];
        } else if ((arg == "--up-axis" || arg == "-a") && i + 1 < argc) {
            std::string_view val = argv[++i];
            up_axis = (val == "Y" || val == "y") ? UpAxis::Y : UpAxis::Z;
        } else if (arg == "--units" && i + 1 < argc) {
            std::string_view val = argv[++i];
            units = (val == "centimeters") ? Units::Centimeters : Units::Meters;
        } else if ((arg == "--num-frames" || arg == "-n") && i + 1 < argc) {
            num_frames = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

template <typename ResultT>
static bool check_and_print_error(ResultT const& result, std::string_view operation) {
    if (result.status != OVRTX_API_SUCCESS) {
        std::cerr << "ERROR in " << operation << " (status=" << result.status << ")\n";
        ovx_string_t err = ovrtx_get_last_error();
        if (err.ptr && err.length > 0) {
            std::cerr << "  " << std::string_view(err.ptr, err.length) << "\n";
        }
        return true;
    }
    return false;
}

static auto vulkan_format_for_output(OutputType type) -> VkFormat {
    return (type == OutputType::HdrColor)
               ? VK_FORMAT_R16G16B16A16_SFLOAT
               : VK_FORMAT_R8G8B8A8_SRGB;
}

static auto cuda_format_for_output(OutputType type) -> CudaImageFormat {
    return (type == OutputType::HdrColor) ? CudaImageFormat::Half4
                                          : CudaImageFormat::UInt8_4;
}

static auto output_type_name(OutputType type) -> char const* {
    return (type == OutputType::HdrColor) ? "HdrColor" : "LdrColor";
}

static void print_frame_time_stats(int total_frames,
                                   std::vector<double> const& cpu_times_ms,
                                   std::vector<double> const& vulkan_times_ms,
                                   std::vector<double> const& cuda_times_ms) {
    std::cerr << "\n=== Frame time statistics (" << total_frames << " frames) ===\n";
    auto print_stats = [](char const* name, std::vector<double> const& v) {
        if (v.empty()) { std::cerr << name << ": no data\n"; return; }
        double sum = 0.0, mn = v[0], mx = v[0];
        for (double x : v) { sum += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        std::cerr << name << ": avg=" << sum / v.size()
                  << " min=" << mn << " max=" << mx << " ms\n";
    };
    print_stats("CPU  ", cpu_times_ms);
    print_stats("Vulkan", vulkan_times_ms);
    print_stats("CUDA ", cuda_times_ms);
}

static void record_rendering_state(CommandBuffer& cmd,
                                   VulkanContext& vk,
                                   uint32_t swapchain_image_index,
                                   ShaderHandle vert_shader,
                                   ShaderHandle frag_shader,
                                   SampledImageHandle read_image_handle) {
    VkRenderingAttachmentInfo color_attachment = {};
    color_attachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView   = vk.swapchain_image_view(swapchain_image_index);
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};

    VkRenderingInfo rendering_info = {};
    rendering_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.renderArea.offset    = {0, 0};
    rendering_info.renderArea.extent    = vk.swapchain_extent();
    rendering_info.layerCount           = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments    = &color_attachment;

    cmd.begin_rendering(rendering_info);

    VkExtent2D extent = vk.swapchain_extent();
    cmd.set_viewport(0.0f, 0.0f,
                     static_cast<float>(extent.width),
                     static_cast<float>(extent.height));
    cmd.set_scissor(0, 0, extent.width, extent.height);
    cmd.set_rasterizer_discard_enable(false);
    cmd.set_polygon_mode(VK_POLYGON_MODE_FILL);
    cmd.set_cull_mode(VK_CULL_MODE_NONE);
    cmd.set_front_face(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    cmd.set_depth_bias_enable(false);
    cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    cmd.set_primitive_restart_enable(false);
    cmd.set_depth_test_enable(false);
    cmd.set_depth_write_enable(false);
    cmd.set_depth_bounds_test_enable(false);
    cmd.set_stencil_test_enable(false);
    cmd.set_rasterization_samples(VK_SAMPLE_COUNT_1_BIT);
    cmd.set_sample_mask(VK_SAMPLE_COUNT_1_BIT, 0xFFFFFFFF);
    cmd.set_alpha_to_coverage_enable(false);
    cmd.set_color_blend_enable(0, false);
    cmd.set_color_write_mask(0,
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
    cmd.set_vertex_input_empty();

    vk.bind_shaders(vert_shader, frag_shader);
    VkDescriptorSet desc_set = vk.descriptor_set();
    cmd.bind_descriptor_sets(VK_PIPELINE_BIND_POINT_GRAPHICS,
                              vk.pipeline_layout(), 0, 1, &desc_set);

    uint32_t tex_idx = vk.sampled_image(read_image_handle).descriptor_index;
    cmd.push_constants(vk.pipeline_layout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(tex_idx), &tex_idx);

    cmd.draw(3);
    cmd.end_rendering();
}

static auto find_color_output(ovrtx_render_product_set_outputs_t const& outputs,
                              OutputType& output_type)
    -> ovrtx_rendered_output_handle_t {
    ovrtx_rendered_output_handle_t hdr_handle = 0;
    ovrtx_rendered_output_handle_t ldr_handle = 0;

    for (size_t i = 0; i < outputs.output_count; ++i) {
        ovrtx_render_product_output_t const& product_output = outputs.outputs[i];
        for (size_t f = 0; f < product_output.output_frame_count; ++f) {
            ovrtx_render_product_frame_output_t const& frame =
                product_output.output_frames[f];
            for (size_t v = 0; v < frame.render_var_count; ++v) {
                ovrtx_render_product_render_var_output_t const& var =
                    frame.output_render_vars[v];
                if (!var.render_var_name.ptr) {
                    continue;
                }
                if (strncmp(var.render_var_name.ptr,
                            "HdrColor",
                            var.render_var_name.length) == 0) {
                    hdr_handle = var.output_handle;
                } else if (strncmp(var.render_var_name.ptr,
                                   "LdrColor",
                                   var.render_var_name.length) == 0) {
                    ldr_handle = var.output_handle;
                }
            }
        }
    }
    if (hdr_handle != 0) {
        output_type = OutputType::HdrColor;
        return hdr_handle;
    }
    if (ldr_handle != 0) {
        output_type = OutputType::LdrColor;
        return ldr_handle;
    }
    return 0;
}
