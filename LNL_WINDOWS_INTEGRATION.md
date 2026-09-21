# Windows Lunar Lake integration

The `lnl` target is an experimental Windows integration line for Intel Arc 130V. The package identity, core AOT marker, ConvRot contract, LGRF/CUTE target selection, and provider manifests must agree on `lnl`.

The implementation boundary covers native ConvRot/RMS/RoPE paths and standalone attention shapes; a clean physical LNL package receipt is still pending. Qwen Image 2.1 attention remains on the PyTorch SDPA route when its shape is outside the CUTE contract.
