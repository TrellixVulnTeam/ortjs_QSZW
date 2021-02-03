#!/usr/bin/env node

'use strict';

const child_process = require('child_process');
const fs = require('fs');
const path = require('path');

const ORT_FOLDER = path.normalize(path.join(__dirname, '../..'));
const EMSDK_FOLDER = path.normalize(path.join('d:/src', 'emsdk'));
const EMSDK_BIN = path.join(EMSDK_FOLDER, 'emsdk');
const EMCC_BIN = path.normalize(path.join(EMSDK_FOLDER, 'upstream/emscripten/em++'));

// build args
const BUILD_TYPE = (process.argv.indexOf('debug') !== -1) ? 'Debug' : 'Release';
const BUILD_ENABLE_SIMD = process.argv.indexOf('simd') !== -1;
const BUILD_ENABLE_PTHREAD = process.argv.indexOf('pthread') !== -1;

const INC_SEARCH_FOLDERS = [
    `${ORT_FOLDER}/onnxruntime/core/mlas/inc`, // mlas.h
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib`, // mlasi.h
    `${ORT_FOLDER}/include/onnxruntime`,
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

    // MLAS Tests
    `${ORT_FOLDER}/onnxruntime/test/mlas/unittest.cpp`,
];


let args = `
${INC_SEARCH_FOLDERS.map(i => `-I${path.normalize(i)}`).join(' ')}
-DORT_MINIMAL_BUILD
-DDISABLE_ML_OPS
-DONNX_ML=1
-DONNX_NAMESPACE=onnx
-std=c++14                                         
-s WASM=1
-s ALLOW_MEMORY_GROWTH=1
-s EXPORT_ALL=0
-o out_mlas.html
-s LLD_REPORT_UNDEFINED
${SOURCE_FILES.map(path.normalize).join(' ')}
`
    ;

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
-O0
-gseparate-dwarf
`
} else {
    args += `
-DNDEBUG
-s VERBOSE=0
-s ASSERTIONS=0
-g4
-O3
-gseparate-dwarf
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
