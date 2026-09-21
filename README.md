# omni_xpu_kernels

Independent public source home for Omni Intel XPU native kernels and PyTorch
bindings. The Python distribution and import remain **`omni_xpu_kernel`**.

The initial source snapshot comes from `intel/llm-scaler` at
`38d9a3dbbfecd3a5293a8acd75efef54f9bb3719`, directory `omni/omni_xpu_kernel`.
Runtime code, policy and package versions are unchanged. Original attribution
and the Apache-2.0 license are retained; this is not an Intel release.

## Ownership and layout

- `omni_xpu_kernel/`: native implementations, Python APIs and target policy.
- `setup.py`, `pyproject.toml`, `policy_codegen.py`: independent wheel build.
- `tests/`, `benchmarks/`, `scripts/`: imported validation and development tools.
- [ARCHITECTURE.md](ARCHITECTURE.md): four-repository boundaries and integration.
- [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json): source identity and file hashes.
- [component-sources.json](component-sources.json): initial component source map.

## Build

Use a development environment with Intel oneAPI, Python development headers,
matched Torch XPU and oneDNN runtime/development packages. The imported Linux
baseline uses Torch `2.13.0+xpu`, oneDNN packages `2026.0.0`, and sycl-tla revision
`2fc09973bfdf15755090fcb0e3b6ad236408a992`.

```bash
source /opt/intel/oneapi/setvars.sh
export CUTLASS_SYCL_ROOT=/path/to/pinned/sycl-tla
export OMNI_XPU_DEVICE=bmg  # select for the actual device
export OMNI_XPU_REQUIRE_CUTE=1
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

Select the actual device and matching Torch ABI; build success is not a device
support claim. Detailed API/build documentation is preserved in
[UPSTREAM_README.md](UPSTREAM_README.md); its parent-directory links and
[Windows build guide](WHL_BUILD_INSTALL.md) refer to the original llm-scaler
layout. For this standalone repository, the kernel root is the repository root.
The source snapshot contains no wheel, DSO, model or measured-performance claim.
