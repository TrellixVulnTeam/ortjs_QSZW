# ONNX Runtime for WebAssembly

Currently, only MLAS can be built to WASM. More parts in ORT will be built into WASM in future.

## HOW TO BUILD:

### Before build

1. Install Node.js 14.x
2. Syncup git submodules (cmake/external/emsdk)
3. Perform one-time setup (This will be implicit called by build.cmd. It takes some time to download.)
    - `emsdk install latest`
    - `emsdk activate latest`

### Building

1. Build ONNXRuntime.

    Building ONNXRuntime helps to generate `onnx-ml.pb.h` and `onnx-operators-ml.pb.h` under folder `build\Windows\{BUILD_TYPE}\onnx`. This file is required for building WebAssembly.

    call `build --config {BUILD_TYPE} --minimal_build` in root folder. Supported BUILD_TYPE are Debug and Release.

2. Build WebAssembly
- Build WebAssembly MVP
   - call `build.cmd`
   - call `build.cmd debug` for debug build
- Build WebAssembly SIMD128
   - call `build.cmd simd`
   - call `build.cmd simd debug` for debug build

### Test

- Use Node.js to launch.
   - call `test.cmd`

### Output

Files `out_wasm_main.*` will be outputted.