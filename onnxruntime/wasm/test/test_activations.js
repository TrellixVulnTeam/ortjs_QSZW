function _test_sigmoid(o) {
    console.log("==== Sigmoid test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3]
    // val[1]: C - [3]

    const A = new Float32Array(3);
    A.set([-1, 0, 1]);
    const C = new Float32Array(3);

    const f1 = new InferenceContext(1,  // num operators
                                    2,  // num values
                                    [1, 1] // value types
                                    );

    f1.initKernel(0, "Sigmoid", "", 11,  // op, opset, opset_ver
                  [0], [1],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3]);  // shape

    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Float32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    
    f1.run();

    const offset_2 = f1.getTensorData(1);
    const size_2 = f1.getTensorDataSize(1);
    const c_out = new Float32Array(o.HEAPU8.buffer, offset_2, size_2);
    C.set(new Float32Array(o.HEAPU8.buffer, offset_2, size_2));
    console.log(C);
    console.log("==== Sigmoid test complete. ====");
}

function _test_tanh(o) {
    console.log("==== Tanh test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3]
    // val[1]: C - [3]

    const A = new Float32Array(3);
    A.set([-1, 0, 1]);
    const C = new Float32Array(3);

    const f1 = new InferenceContext(1,  // num operators
                                    2,  // num values
                                    [1, 1] // value types
                                    );

    f1.initKernel(0, "Tanh", "", 11,  // op, opset, opset_ver
                  [0], [1],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3]);  // shape

    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Float32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    
    f1.run();

    const offset_2 = f1.getTensorData(1);
    const size_2 = f1.getTensorDataSize(1);
    const c_out = new Float32Array(o.HEAPU8.buffer, offset_2, size_2);
    C.set(new Float32Array(o.HEAPU8.buffer, offset_2, size_2));
    console.log(C);
    console.log("==== Tanh test complete. ====");
}

module.exports = function (o) {
    _test_sigmoid(o);
    _test_tanh(o);
}