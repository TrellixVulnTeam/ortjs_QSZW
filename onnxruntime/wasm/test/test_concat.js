function _test_concat(o) {
    console.log("==== CONCAT test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[1]: B - [3, 4]
    // val[2]: C - [6, 4]

    const A = new Float32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const B = new Float32Array(12);
    B.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const C = new Float32Array(24);

    const f1 = new InferenceContext(1,  // num operators
                                    3,  // num values
                                    [1, 1, 1], // value types
                                    );

    f1.addAttribute_i(0, "axis", 0);

    f1.initKernel(0, "Concat", "", 7,  // op, opset, opset_ver
                  [0, 1], [2],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 4]);  // shape

    f1.setInput(1,        // value idx
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
    console.log("==== CONCAT test complete. ====");
}

function _test_concat_3_inputs(o) {
    console.log("==== CONCAT test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[1]: B - [3, 4]
    // val[2]: C - [3, 4]
    // val[3]: D - [9, 4]

    const A = new Float32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const B = new Float32Array(12);
    B.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const C = new Float32Array(12);
    C.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const D = new Float32Array(36);

    const f1 = new InferenceContext(1,  // num operators
                                    4,  // num values
                                    [1, 1, 1, 1], // value types
                                    );

    f1.addAttribute_i(0, "axis", 0);

    f1.initKernel(0, "Concat", "", 7,  // op, opset, opset_ver
                  [0, 1, 2], [3],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 4]);  // shape

    f1.setInput(1,        // value idx
                [3, 4]);  // shape

    f1.setInput(2,        // value idx
                [3, 4]);  // shape
    
    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Float32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);

    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    new Float32Array(o.HEAPU8.buffer, offset_1, size_1).set(B);

    const offset_2 = f1.getTensorData(2);
    const size_2 = f1.getTensorDataSize(2);
    new Float32Array(o.HEAPU8.buffer, offset_2, size_2).set(C);
    
    f1.run();

    const offset_3 = f1.getTensorData(3);
    const size_3 = f1.getTensorDataSize(3);
    const d_out = new Float32Array(o.HEAPU8.buffer, offset_3, size_3);
    D.set(new Float32Array(o.HEAPU8.buffer, offset_3, size_3));
    console.log(D);
    console.log("==== CONCAT test complete. ====");
}

module.exports = function (o) {
    _test_concat(o);
    _test_concat_3_inputs(o);
}