const wasm_factory = require('../out_wasm_main');
const binary_tests = require('./test_binary');
const concat_tests = require('./test_concat');
const gather_tests = require('./test_gather');
const matmul_tests = require('./test_matmul');
const slice_tests = require('./test_slice');
const unsqueeze_tests = require('./test_unsqueeze');


function _test_gemm(o) {
    console.log("==== GEMM test starts. ====");
    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[1]: B - [4, 5]
    // val[2]: C - [3, 5]

    const A = new Float32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const B = new Float32Array(20);
    B.set([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]);
    const C = new Float32Array(15);

    const f1 = new InferenceContext(1,  // num operators
                                    3,  // num values
                                    [1, 1, 1], // value types
                                    );

    f1.setInitializer(1,             // value index
                      [4, 5]);       // dim
    
    f1.addAttribute_i(0, "transA", 0);
    f1.addAttribute_i(0, "transB", 0);
    f1.addAttribute_f(0, "alpha", 1.0);
    f1.addAttribute_f(0, "beta", 0.0);

    f1.initKernel(0, "Gemm", "", 7,  // op, opset, opset_ver
                  [0, 1], [2],    // inputs idx, output idx
                  "");

    // init set one time. 
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    new Float32Array(o.HEAPU8.buffer, offset_1, size_1).set(B);

    f1.setInput(0,        // value idx
                [3, 4]);  // shape

    // can set multiple. 
    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Float32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    

    f1.run();

    const offset_2 = f1.getTensorData(2);
    const size_2 = f1.getTensorDataSize(2);
    const c_out = new Float32Array(o.HEAPU8.buffer, offset_2, size_2);
    C.set(new Float32Array(o.HEAPU8.buffer, offset_2, size_2));
    console.log(C);
    console.log("==== GEMM test complete. ====");

}


wasm_factory().then((o) => {

    _test_gemm(o);
    binary_tests(o, "Add");
    binary_tests(o, "Mul");
    concat_tests(o);
    gather_tests(o);
    matmul_tests(o);
    slice_tests(o);
    unsqueeze_tests(o);
});

//wasm_factory();
