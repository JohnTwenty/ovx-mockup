# AGENTS.md — AI agent context for ovx-mockup

This file records design decisions, architecture notes, and working conventions
for AI coding agents (Claude Code or similar) contributing to this project.

## Commit authorship

**All commits made by an AI agent must use the author `Claude <noreply@anthropic.com>`.**

```bash
git commit --author="Claude <noreply@anthropic.com>" -m "..."
# or amend an existing commit:
git commit --amend --author="Claude <noreply@anthropic.com>" --no-edit
```

The project owner's corporate git identity (`amoravanszky@nvidia.com`) must
not appear in AI-generated commits.

---

## Dependency strategy

This project deliberately owns as few files as possible. The design goal is
a repo whose only authored content is `CMakeLists.txt`, `src/main.cpp`, and
`assets/physics_boxes.usda`.

### ovrtx

- **Not built from source.** ovrtx ships pre-built binaries on GitHub Releases.
- The cmake module `third-party/ovrtx/examples/c/cmake/ovrtx.cmake` (inside
  the submodule) exposes two macros:
  - `ovrtx_fetch()` — detects platform, downloads the correct zip from GitHub
    Releases via FetchContent, and makes `ovrtx::ovrtx` findable.
  - `ovrtx_setup_runtime(TARGET)` — on Linux sets RPATH; on Windows copies the
    DLL and creates junctions for the runtime plugin directories.
- The submodule is pinned to **v0.2.0** (commit `393afc6`).

### Vulkan/CUDA helper sources

The physx example shares six `.cpp` files with the `vulkan-interop` example
(Vulkan context, CUDA kernel, sampled image, shader loader, command buffer,
orbit camera). Rather than copying them, `CMakeLists.txt` references them
directly from the submodule:

```cmake
set(VI_SRC "${OVRTX_EXAMPLES_C}/vulkan-interop/src")
add_executable(ovx-mockup
    src/main.cpp
    ${VI_SRC}/cuda/cuda_kernel.cpp
    ${VI_SRC}/vk/vulkan_context.cpp
    ...
)
target_include_directories(ovx-mockup PRIVATE ${VI_SRC} ...)
```

If ovrtx releases a new version that changes these helpers, update the
submodule pin and retest.

### ovphysx

- Consumed as a pre-built tarball downloaded manually from
  `https://github.com/NVIDIA-Omniverse/PhysX/releases`.
- Version in use: **0.2.8** (package name `ovphysx-linux-x86_64-0.2.8` /
  `ovphysx-windows-x86_64-0.2.8`).
- Discovered by cmake via `find_package(ovphysx REQUIRED)` when the user
  passes `-DCMAKE_PREFIX_PATH=/path/to/extracted/ovphysx`.
- There is no automated FetchContent step for ovphysx. If one is desired in
  the future, model it after `ovrtx.cmake` with platform-specific URLs and
  SHA256 hashes.

### FetchContent cache location

`FETCHCONTENT_BASE_DIR` is set to `${PROJECT_ROOT}/_deps` *before*
`ovrtx_fetch()` is called. This pre-empts the cmake module's default, which
would otherwise place downloads inside the submodule tree. `_deps/` is
gitignored.

### Why submodule over copying files

Three alternatives were evaluated:

| Approach | Verdict |
|---|---|
| Copy helper .cpp files into the project | Rejected — any upstream fix requires manual re-sync |
| FetchContent source archive of ovrtx repo | Viable alternative; requires network at every fresh configure |
| **Git submodule pinned to v0.2.0** | **Chosen** — explicit version pinning, no forked files, source available offline after `submodule update --init` |

---

## ovphysx API notes

These were learned while developing `src/main.cpp` and are recorded here to
avoid re-discovery.

### Stage is read-only after load

ovphysx reads the USD stage only at `ovphysx_load`. Writing USD attributes at
runtime (e.g. via `ovrtx_write_attribute`) does **not** affect the running
simulation. The `physxCharacterController:moveTarget` USD approach was ruled
out for this reason.

### Tensor bindings are the runtime interface

```cpp
// Read world-space poses: [N, 7] = [px py pz qx qy qz qw]
ovphysx_create_tensor_binding(..., OVPHYSX_TENSOR_RIGID_BODY_POSE_F32, ...);
ovphysx_read_tensor_binding(handle, binding, &tensor);

// Write velocities: [N, 6] = [vx vy vz wx wy wz]
ovphysx_create_tensor_binding(..., OVPHYSX_TENSOR_RIGID_BODY_VELOCITY_F32, ...);
ovphysx_write_tensor_binding(handle, binding, &tensor, nullptr);
```

### Kinematic pose writes do not push dynamic bodies

`setGlobalPose` (teleport) generates no contact impulses. Use velocity writes
on a regular dynamic body instead — PhysX resolves contacts normally.

### Velocity write must come before ovphysx_step

The write takes effect in the *same* step it is issued, so call
`ovphysx_write_tensor_binding` before `ovphysx_step`.

### High mass prevents deflection

The character capsule is 500 kg; the dynamic objects are 1–2 kg. This
prevents the character from bouncing off what it is supposed to push.

### No raw PhysX pointers

`ovphysx_handle_t` is an opaque integer. `PxScene*`, `PxPhysics*`, and other
PhysX SDK objects are inaccessible in the current API.

---

## Possible future work

- **Automate ovphysx download**: add an `ovphysx_fetch()` cmake macro (modeled
  on `ovrtx.cmake`) with platform-specific URLs and SHA256 hashes.
- **Keyboard-driven character**: hook GLFW key callbacks to replace the
  sin-wave velocity driver.
- **Read-back character position**: add a pose read binding for
  `/World/Character` so the rendered position tracks physics rather than the
  commanded velocity integral.
- **Angular velocity / tumbling**: write non-zero `wx/wy/wz` to spin boxes
  on impact.
- **Character controller**: ask ovphysx devs whether `PxController::move()`
  or a CCT tensor type is on the roadmap.
- **Multi-environment RL**: use `ovphysx_clone` (experimental C++ API) to
  replicate the scene for parallel training runs.
- **Upgrade submodule**: when ovrtx ships a newer release, update the
  submodule pin (`cd third-party/ovrtx && git checkout <new-tag>`) and
  update the version comment in `CMakeLists.txt`.
