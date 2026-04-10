# ovx-mockup

Interactive rigid-body physics simulation driving GPU ray-traced rendering.
Uses [ovphysx](https://github.com/NVIDIA-Omniverse/PhysX) for CPU-mode physics
and [ovrtx](https://github.com/NVIDIA-Omniverse/ovrtx) for RTX rendering, connected
through their shared DLPack tensor interface.

Each frame:
1. A heavy dynamic capsule is driven back and forth by a velocity write.
2. `ovphysx_step` simulates contacts with the lighter dynamic objects.
3. World-space poses are read back via `ovphysx_read_tensor_binding`.
4. Poses are pushed into the renderer via `ovrtx_set_xform_pos_rot_scale`.
5. `ovrtx_step` + Vulkan/CUDA interop presents the frame.

## Prerequisites

- NVIDIA RTX GPU with a recent driver
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x
- [Vulkan SDK](https://vulkan.lunarg.com/) (provides `glslc`)
- CMake 3.18+
- C++17 compiler (GCC 11+ / MSVC 2019+)
- Git

## Getting ovphysx

Download the pre-built tarball for your platform from the
[NVIDIA-Omniverse/PhysX releases page](https://github.com/NVIDIA-Omniverse/PhysX/releases)
and extract it somewhere convenient, e.g.:

**Linux**
```bash
# Example: ovphysx-linux-x86_64-0.2.8
tar -xf ovphysx-linux-x86_64-0.2.8.tar.gz -C ~/local
```

**Windows**
```powershell
# Example: ovphysx-windows-x86_64-0.2.8
Expand-Archive ovphysx-windows-x86_64-0.2.8.zip -DestinationPath C:\local
```

## Cloning

```bash
git clone https://github.com/JohnTwenty/ovx-mockup
cd ovx-mockup
git submodule update --init --depth=1
```

The submodule (`third-party/ovrtx`, pinned to v0.2.0) provides two things:
- The cmake module that auto-downloads the ovrtx pre-built binary at configure time.
- The Vulkan/CUDA helper sources shared with the `vulkan-interop` example.

## Getting ovrtx (normally automatic)

The cmake configure step downloads the ovrtx pre-built binary from GitHub
Releases automatically into `_deps/`. No manual step is needed if your
machine can reach `github.com`.

**If the download is blocked** (corporate proxy, air-gap network): download
the zip manually from the
[ovrtx releases page](https://github.com/NVIDIA-Omniverse/ovrtx/releases/tag/v0.2.0),
extract it, and pass the path with `-DOVRTX_DIR=...` (see below).

| Platform | Package filename |
|---|---|
| Linux x86_64 | `ovrtx@0.2.0.manylinux_2_35_x86_64.zip` |
| Linux aarch64 | `ovrtx@0.2.0.manylinux_2_35_aarch64.zip` |
| Windows x86_64 | `ovrtx@0.2.0.windows-x86_64.zip` |

## Building

### Linux

```bash
cmake -B build \
    -DCMAKE_PREFIX_PATH=/path/to/ovphysx-linux-x86_64-0.2.8/ovphysx
cmake --build build --parallel
```

### Windows (Visual Studio)

```powershell
cmake -B build `
    -DCMAKE_PREFIX_PATH="C:\local\ovphysx-windows-x86_64-0.2.8\ovphysx"
cmake --build build --config Release
```

### Offline / behind a proxy

Pass both package paths so cmake never attempts a network download:

```bash
cmake -B build \
    -DCMAKE_PREFIX_PATH=/path/to/ovphysx-linux-x86_64-0.2.8/ovphysx \
    -DOVRTX_DIR=/path/to/ovrtx@0.2.0.manylinux_2_35_x86_64
cmake --build build --parallel
```

On Windows:
```powershell
cmake -B build `
    -DCMAKE_PREFIX_PATH="C:\local\ovphysx-windows-x86_64-0.2.8\ovphysx" `
    -DOVRTX_DIR="C:\local\ovrtx@0.2.0.windows-x86_64"
cmake --build build --config Release
```

`-DOVRTX_DIR` tells cmake where to find the pre-extracted ovrtx package.
`ovrtx_fetch()` detects it via `find_package` and skips the download.
Other dependencies (glfw3, glm, volk, unordered_dense) are fetched from
GitHub at configure time; if those are also blocked, install them via your
system package manager or vcpkg and cmake will find them automatically.

## Running

```bash
./build/ovrtx-interop-physx                  # Linux
.\build\Release\ovrtx-interop-physx.exe      # Windows
```

The binary expects `physics_boxes.usda` and compiled shaders in the same
directory; CMake post-build commands deploy them automatically.

Optional arguments:
```
ovrtx-interop-physx [usd_file] [render_product] [up_axis] [duration_s]
```

## Project layout

```
ovx-mockup/
├── CMakeLists.txt           main build file (~120 lines)
├── src/
│   └── main.cpp             application code (physics + render loop)
├── assets/
│   └── physics_boxes.usda   USD scene (rigid bodies + lights + camera)
└── third-party/
    └── ovrtx/               git submodule @ v0.2.0
        examples/c/
          cmake/ovrtx.cmake  fetches pre-built ovrtx binary
          vulkan-interop/
            shaders/         GLSL sources compiled at build time
            src/             Vulkan/CUDA helpers (referenced, not copied)
```

The only files this project owns are `CMakeLists.txt`, `src/main.cpp`, and
`assets/physics_boxes.usda`. Everything else is either fetched at build time
or referenced from the submodule.
