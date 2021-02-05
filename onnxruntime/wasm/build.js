#!/usr/bin/env node

'use strict';

const child_process = require('child_process');
const fs = require('fs');
const path = require('path');

const ORT_FOLDER = path.normalize(path.join(__dirname, '../..'));
const EMSDK_FOLDER = path.normalize(path.join(ORT_FOLDER, 'cmake/external/emsdk'));
const EMSDK_BIN = path.join(EMSDK_FOLDER, 'emsdk');
const EMCC_BIN = path.normalize(path.join(EMSDK_FOLDER, 'upstream/emscripten/em++'));

// build args
const BUILD_TYPE = (process.argv.indexOf('debug') !== -1) ? 'Debug' : 'Release';
const BUILD_ENABLE_SIMD = process.argv.indexOf('simd') !== -1;
const BUILD_ENABLE_PTHREAD = process.argv.indexOf('pthread') !== -1;
const BUILD_TEST = process.argv.indexOf('test') !== -1;

const INC_SEARCH_FOLDERS = [
    `${ORT_FOLDER}/onnxruntime/core/mlas/inc`, // mlas.h
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib`, // mlasi.h
    `${ORT_FOLDER}/onnxruntime`,
    `${ORT_FOLDER}/include/onnxruntime`,

    // onnxruntime_config.h
    `${ORT_FOLDER}/build/Windows/${BUILD_TYPE}`,

    // eigen
    `${ORT_FOLDER}/cmake/external/eigen`,

    // nsync
    //`${ORT_FOLDER}/cmake/external/nsync`,

    `${ORT_FOLDER}/cmake/external/onnx`,
    `${ORT_FOLDER}/cmake/external/optional-lite/include`,
    `${ORT_FOLDER}/cmake/external/mp11/include`,
    `${ORT_FOLDER}/cmake/external/protobuf/src`,
    `${ORT_FOLDER}/cmake/external/SafeInt`,

    //'.',
];

const SOURCE_FILES = [
    // MLAS
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/threading.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/platform.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/sgemm.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/logistic.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/tanh.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/snchwc.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/activate.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/reorder.cpp`,    
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/convolve.cpp`,

    // onnx
    //`${ORT_FOLDER}/build/Windows/${BUILD_TYPE}/onnx/onnx-ml.pb.cc`,
    //`${ORT_FOLDER}/cmake/external/onnx/onnx/defs/data_type_utils.cc`,

    // PLATFORM
    `${ORT_FOLDER}/onnxruntime/core/common/denormal.cc`,
    `${ORT_FOLDER}/onnxruntime/core/common/logging/capture.cc`,
    `${ORT_FOLDER}/onnxruntime/core/common/logging/logging.cc`,
    `${ORT_FOLDER}/onnxruntime/core/common/status.cc`,
    `${ORT_FOLDER}/onnxruntime/core/common/threadpool.cc`,
    `${ORT_FOLDER}/onnxruntime/core/platform/env.cc`,
    `${ORT_FOLDER}/onnxruntime/core/platform/env_time.cc`,
    `${ORT_FOLDER}/onnxruntime/core/platform/telemetry.cc`,

    `${ORT_FOLDER}/onnxruntime/core/platform/posix/env.cc`,
    `${ORT_FOLDER}/onnxruntime/core/platform/posix/env_time.cc`,
    //`${ORT_FOLDER}/onnxruntime/core/platform/posix/ort_mutex.cc`,
    `${ORT_FOLDER}/onnxruntime/core/platform/posix/stacktrace.cc`,

    //`${ORT_FOLDER}/onnxruntime/core/framework/allocator.cc`,
    `${ORT_FOLDER}/onnxruntime/core/framework/tensor.cc`,
    `${ORT_FOLDER}/onnxruntime/core/framework/tensor_shape.cc`,
    `${ORT_FOLDER}/onnxruntime/core/framework/op_kernel.cc`,
    `${ORT_FOLDER}/onnxruntime/core/framework/op_kernel_info.cc`,
    `${ORT_FOLDER}/onnxruntime/core/framework/op_node_proto_helper.cc`,
    `${ORT_FOLDER}/onnxruntime/core/framework/data_types.cc`,

    //`${ORT_FOLDER}/onnxruntime/core/providers/cpu/cpu_provider_factory.cc`,

    // register
    `${ORT_FOLDER}/onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/cpu_execution_provider.cc`,


    // OPs
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/math/gemm.cc`,
    // Add HERE!!!
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/math/element_wise_ops.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/tensor/concat.cc`,
    `${ORT_FOLDER}/onnxruntime/core/util/math_cpu.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/tensor/gather.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/math/matmul.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/tensor/slice.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/tensor/unsqueeze.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/activation/activations.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/tensor/reshape.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/nn/conv.cc`,
    `${ORT_FOLDER}/onnxruntime/contrib_ops/cpu/fused_conv.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/tensor/resize.cc`,
    `${ORT_FOLDER}/onnxruntime/core/providers/cpu/tensor/upsample.cc`,

    // utils
    `${ORT_FOLDER}/onnxruntime/wasm/api.cc`,
    `${ORT_FOLDER}/onnxruntime/wasm/utils.cc`,

];

if (BUILD_TEST) {
    SOURCE_FILES.push(`${ORT_FOLDER}/onnxruntime/test/mlas/unittest.cpp`);
}

let args = `
${INC_SEARCH_FOLDERS.map(i => `-I${path.normalize(i)}`).join(' ')}
-DEIGEN_MPL2_ONLY                                             
-DORT_MINIMAL_BUILD
-DDISABLE_ML_OPS
-DONNX_ML=1
-DONNX_NAMESPACE=onnx
-std=c++14                                                    
--bind
-s ASYNCIFY=1
-s EXPORT_NAME=onnxjs
-s WASM=1                                                     
-s NO_EXIT_RUNTIME=0                                          
-s ALLOW_MEMORY_GROWTH=1                                      
-s SAFE_HEAP=0                                                
-s MODULARIZE=1                                               
-s SAFE_HEAP_LOG=0                                            
-s STACK_OVERFLOW_CHECK=0                                     
-s EXPORT_ALL=0                                               
-o out_wasm_main.js                                           
-s LLD_REPORT_UNDEFINED
${SOURCE_FILES.map(path.normalize).join(' ')}
`
    ;

if (BUILD_TEST) {
    args += ` -s "EXPORTED_FUNCTIONS=[_main]" `;
} else {
    args += ` --no-entry `;
}

if (BUILD_ENABLE_SIMD) {
    args += `
-msimd128 
${path.normalize(`${ORT_FOLDER}/onnxruntime/core/mlas/lib/wasm_simd/sgemmc.cpp`)}
`
} else {
    args += `
${path.normalize(`${ORT_FOLDER}/onnxruntime/core/mlas/lib/wasm/sgemmc.cpp`)}
`
}

if (BUILD_TYPE === 'Debug') {
    args += `
-s VERBOSE=1
-s ASSERTIONS=1
-g4 
`
} else {
    args += `
-DNDEBUG
-s VERBOSE=0
-s ASSERTIONS=0
-O3                                                          
`
}

if (BUILD_ENABLE_PTHREAD) {
    args += ' -pthread ';
} else {
    args += ' -DMLAS_NO_ONNXRUNTIME_THREADPOOL ';
}


if (!fs.existsSync(EMCC_BIN)) {

    // One-time installation: 'emsdk install latest'

    const install = child_process.spawnSync(`${EMSDK_BIN} install latest`, { shell: true, stdio: 'inherit', cwd: EMSDK_FOLDER });
    if (install.status !== 0) {
        if (install.error) {
            console.error(install.error);
        }
        process.exit(install.status === null ? undefined : install.status);
    }

    // 'emsdk activate latest'

    const activate = child_process.spawnSync(`${EMSDK_BIN} activate latest`, { shell: true, stdio: 'inherit', cwd: EMSDK_FOLDER });
    if (activate.status !== 0) {
        if (activate.error) {
            console.error(activate.error);
        }
        process.exit(activate.status === null ? undefined : activate.status);
    }
}

console.log(`${EMCC_BIN} ${args.split('\n').map(i => i.trim()).filter(i => i !== '').join(' ')}`);
const emccBuild = child_process.spawnSync(EMCC_BIN, args.split('\n').map(i => i.trim()), { shell: true, stdio: 'inherit', cwd: __dirname });

if (emccBuild.error) {
    console.error(emccBuild.error);
}
process.exit(emccBuild.status === null ? undefined : emccBuild.status);
