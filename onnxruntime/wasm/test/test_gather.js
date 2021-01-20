function _test_gather_1d(o) {
    console.log("==== GATHER_1d test starts. ====")

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [6]
    // val[1]: B - [3]
    // val[2]: C - [3]

    const A = new Int32Array(6);
    A.set([4, 3, 5, 7, 6, 8]);
    const B = new Int32Array(3);
    B.set([0, 1, 4]);
    const C = new Int32Array(3);

    const f1 = new InferenceContext(1,  // num operators
                                    3,  // num values
                                    [6, 6, 6] // value types
                                    );

    f1.addAttribute_i(0, "axis", 0);

    f1.initKernel(0, "Gather", "", 11,  // op, opset, opset_ver
                  [0, 1], [2],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [6]);  // shape

    f1.setInput(1,        // value idx
                [3]);  // shape
    
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    new Int32Array(o.HEAPU8.buffer, offset_1, size_1).set(B);

    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Int32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    
    f1.run();

    const offset_2 = f1.getTensorData(2);
    const size_2 = f1.getTensorDataSize(2);
    const c_out = new Int32Array(o.HEAPU8.buffer, offset_2, size_2);
    C.set(new Int32Array(o.HEAPU8.buffer, offset_2, size_2));
    console.log(C);
    console.log("==== GATHER_1d test complete. ====")
}

function _test_gather_2d_axis_0(o) {
    console.log("==== GATHER_2d test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 2]
    // val[1]: B - [2, 2]
    // val[2]: C - [2, 2, 2]

    const A = new Float32Array(6);
    A.set([1.0, 1.2, 2.3, 3.4, 4.5, 5.7]);
    const B = new Int32Array(4);
    B.set([0, 1, 1, 2]);
    const C = new Float32Array(8);

    const f1 = new InferenceContext(1,  // num operators
                                    3,  // num values
                                    [1, 6, 1] // value types
                                    );

    f1.addAttribute_i(0, "axis", 0);

    f1.initKernel(0, "Gather", "", 11,  // op, opset, opset_ver
                  [0, 1], [2],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 2]);  // shape

    f1.setInput(1,        // value idx
                [2, 2]);  // shape
    
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    new Int32Array(o.HEAPU8.buffer, offset_1, size_1).set(B);

    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Float32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    
    f1.run();

    const offset_2 = f1.getTensorData(2);
    const size_2 = f1.getTensorDataSize(2);
    const c_out = new Float32Array(o.HEAPU8.buffer, offset_2, size_2);
    C.set(new Float32Array(o.HEAPU8.buffer, offset_2, size_2));
    console.log(C);
    console.log("==== GATHER_2d test complete. ====");
}
module.exports = function (o) {
    _test_gather_1d(o);
    _test_gather_2d_axis_0(o);
}