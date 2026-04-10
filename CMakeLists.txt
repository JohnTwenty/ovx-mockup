# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

cmake_minimum_required(VERSION 3.18)
project(ovx-mockup LANGUAGES CXX)

# ============================================================
# System dependencies
# ============================================================
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found.")
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc
    HINTS $ENV{VULKAN_1_2_SDK_BIN}
)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install Vulkan SDK.")
endif()
message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")

# ovphysx - pass -DCMAKE_PREFIX_PATH=/path/to/ovphysx at configure time
find_package(ovphysx REQUIRED)
message(STATUS "Found ovphysx: ${ovphysx_VERSION}")

# ============================================================
# C++ standard
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# FetchContent base dir - set before ovrtx_fetch() so all downloaded
# artifacts land in _deps/ at the project root, not inside the submodule.
# ============================================================
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/_deps" CACHE PATH
    "Shared FetchContent download/build directory")

# ============================================================
# ovrtx - cmake module lives in the submodule; it downloads the
# pre-built binary from GitHub Releases automatically.
# ============================================================
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/third-party/ovrtx/examples/c/cmake")
include(ovrtx)
ovrtx_fetch()

# ============================================================
# Other dependencies via FetchContent
# ============================================================
set(FETCHCONTENT_QUIET FALSE)
include(FetchContent)

find_package(glfw3 QUIET)
if(glfw3_FOUND)
    message(STATUS "glfw3 found at: ${glfw3_DIR}")
else()
    message(STATUS "glfw3 not found, building from source")
    FetchContent_Declare(
        glfw3
        URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(glfw3)
endif()

FetchContent_Declare(
    unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG v4.4.0
)
FetchContent_MakeAvailable(unordered_dense)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 1.4.304
)
FetchContent_MakeAvailable(volk)

# ============================================================
# Shaders - GLSL sources reused from the submodule
# ============================================================
set(OVRTX_EXAMPLES_C
    "${CMAKE_CURRENT_LIST_DIR}/third-party/ovrtx/examples/c")
set(SHADER_SOURCE_DIR "${OVRTX_EXAMPLES_C}/vulkan-interop/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SHADER_SOURCES
    ${SHADER_SOURCE_DIR}/fullscreen.vert
    ${SHADER_SOURCE_DIR}/fullscreen.frag
)

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SHADER_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${GLSLC_EXECUTABLE} -O --target-env=vulkan1.2 --target-spv=spv1.5
                ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
        VERBATIM
    )
    list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
endforeach()

add_custom_target(shaders DEPENDS ${SHADER_OUTPUTS})

# ============================================================
# Executable
#   src/main.cpp       - this project's application code
#   VI_SRC/*           - Vulkan/CUDA helpers from the submodule
#                        (vulkan-interop example sources, not forked)
# ============================================================
set(VI_SRC "${OVRTX_EXAMPLES_C}/vulkan-interop/src")

add_executable(ovx-mockup
    src/main.cpp
    ${VI_SRC}/cuda/cuda_kernel.cpp
    ${VI_SRC}/vk/vulkan_context.cpp
    ${VI_SRC}/vk/sampled_image.cpp
    ${VI_SRC}/vk/shader.cpp
    ${VI_SRC}/vk/command_buffer.cpp
    ${VI_SRC}/camera/orbit_camera.cpp
)

add_dependencies(ovx-mockup shaders)

target_include_directories(ovx-mockup PRIVATE
    ${VI_SRC}
    ${Vulkan_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${unordered_dense_SOURCE_DIR}/include
)

target_link_libraries(ovx-mockup PRIVATE
    Vulkan::Vulkan
    glfw
    CUDA::cuda_driver
    CUDA::nvrtc
    glm::glm
    volk
    ovrtx::ovrtx
    ovphysx::ovphysx
)

# Deploy shaders next to executable
add_custom_command(TARGET ovx-mockup POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:ovx-mockup>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SHADER_OUTPUTS}
        "$<TARGET_FILE_DIR:ovx-mockup>/shaders/"
    COMMENT "Copying shaders"
)

# Deploy USD scene next to executable
add_custom_command(TARGET ovx-mockup POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/physics_boxes.usda"
        "$<TARGET_FILE_DIR:ovx-mockup>/physics_boxes.usda"
    COMMENT "Copying physics_boxes.usda"
)

# Handle runtime dependencies (RPATH on Linux, DLL + junctions on Windows)
ovrtx_setup_runtime(ovx-mockup)
