function _test_reshape(o) {
    console.log("==== Reshape test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[1]: B - [2]
    // val[2]: C - [12]

    const A = new Int32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const B = new BigInt64Array(2);
    B.set([BigInt(4), BigInt(3)]);
    const C = new Int32Array(12);

    const f1 = new InferenceContext(1,  // num operators
                                    3,  // num values
                                    [6, 7, 6] // value types
                                    );

    f1.initKernel(0, "Reshape", "", 11,  // op, opset, opset_ver
                  [0, 1], [2],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 4]);  // shape

    f1.setInput(1,        // value idx
                [2]);  // shape
    
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    new BigInt64Array(o.HEAPU8.buffer, offset_1, size_1).set(B);

    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Int32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    
    f1.run();

    const offset_2 = f1.getTensorData(2);
    const size_2 = f1.getTensorDataSize(2);
    const c_out = new Int32Array(o.HEAPU8.buffer, offset_2, size_2);
    C.set(new Int32Array(o.HEAPU8.buffer, offset_2, size_2));
    console.log(C);
    const a_shape = f1.getTensorShape(0);
    const c_shape = f1.getTensorShape(2);
    console.log("Input old shape: %s, Output new shape: %s", a_shape, c_shape);
    console.log("==== Reshape test complete. ====");
}

module.exports = function (o) {
    _test_reshape(o);
}