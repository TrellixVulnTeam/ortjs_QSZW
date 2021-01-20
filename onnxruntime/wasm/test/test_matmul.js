function _test_matmul(o) {
    console.log("==== MATMUL test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[1]: B - [4, 3]
    // val[2]: C - [3, 3]

    const A = new Float32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const B = new Float32Array(12);
    B.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const C = new Float32Array(9);

    const f1 = new InferenceContext(1,  // num operators
                                    3,  // num values
                                    [1, 1, 1] // value types
                                    );

    f1.setInitializer(1,             // value index
                      [4, 3]);       // dim
    
    f1.addAttribute_i(0, "transA", 0);
    f1.addAttribute_i(0, "transB", 0);
    f1.addAttribute_f(0, "alpha", 1.0);

    f1.initKernel(0, "MatMul", "", 7,  // op, opset, opset_ver
                  [0, 1], [2],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 4]);  // shape
    
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    new Float32Array(o.HEAPU8.buffer, offset_1, size_1).set(B);

    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Float32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    
    f1.run();

    const offset_2 = f1.getTensorData(2);
    const size_2 = f1.getTensorDataSize(2);
    const c_out = new Float32Array(o.HEAPU8.buffer, offset_2, size_2);
    C.set(new Float32Array(o.HEAPU8.buffer, offset_2, size_2));
    console.log(C);
    console.log("==== MATMUL test complete. ====");
}

module.exports = function (o) {
    _test_matmul(o);
}