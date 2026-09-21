# Component architecture

This workspace follows the ownership boundaries in `shinosawabot/owl-xpu` and
separately maintained benchmark evidence, expressed as four independent public
repositories.

| Repository | Responsibility |
| --- | --- |
| [omni_xpu_kernels](https://github.com/shinosawabot/omni_xpu_kernels) | Native XPU kernels, Torch bindings, target capabilities and kernel policy |
| [comfy-kitchen](https://github.com/shinosawabot/comfy-kitchen) | Common operator APIs, per-call capability checks, XPU dispatch and fallback |
| [comfy-aimdo](https://github.com/shinosawabot/comfy-aimdo) | Device memory management and allocator lifecycle |
| [ComfyUI_OmniXPU](https://github.com/shinosawabot/ComfyUI_OmniXPU) | Prestartup provider selection and thin ComfyUI call-site adapters |

ComfyUI calls native kernels through Kitchen or a dedicated adapter. Native
kernels do not depend on ComfyUI, Kitchen or AIMDO. Generic operator dispatch
belongs in Kitchen; allocator policy belongs in AIMDO. ComfyUI integration must
not duplicate generic Kitchen registrations or replace workflow/model semantics.

Official `comfy-kitchen` / `comfy-aimdo` packages may coexist with separately
packaged `comfy-kitchen-xpu-runtime` / `comfy-aimdo-xpu-runtime` providers. Provider
wheels own private vendor paths, not official package files. Prestartup selects
one canonical runtime per namespace, validates compatibility, and keeps safe
fallbacks. Kitchen and AIMDO activation are independent; AIMDO additionally
requires the explicit DynamicVRAM and allocator-lifecycle gates.

Repository names do not rename Python imports or distributions. In particular,
`omni_xpu_kernels` retains the existing `omni_xpu_kernel` import/distribution.
No version bump, wheel release, model download or new device-support claim is
part of repository initialization. Each target requires its own build and
correctness evidence; the four imported snapshots are not a newly validated
end-to-end component combination.

Benchmark and tuning sources remain the owner of benchmark contracts and measured
evidence. Their target profiles and same-run ceiling rules are not copied into
runtime policy or replaced by nominal hardware specifications.

See `SOURCE_PROVENANCE.json` and `component-sources.json` for exact source identities.
