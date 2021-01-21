function _test_unsqueeze(o) {
    console.log("==== Slice test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[2]: C - [4]

    const A = new Int32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const C = new Int32Array(12);

    const f1 = new InferenceContext(1,  // num operators
                                    2,  // num values
                                    [6, 6] // value types
                                    );

    f1.addAttribute_ints(0, "axes", [2]);

    f1.initKernel(0, "Unsqueeze", "", 9,  // op, opset, opset_ver
                  [0], [1],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 4]);  // shape
    


    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Int32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    const a_shape = f1.getTensorShape(0);
    console.log("a shape ", a_shape);
    
    f1.run();
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    const c_shape = f1.getTensorShape(1);
    c_out = new Int32Array(o.HEAPU8.buffer, offset_1, size_1);
    C.set(new Int32Array(o.HEAPU8.buffer, offset_1, size_1));
    console.log("Output new shape ", c_shape);
    console.log("==== Unsqueeze test complete. ====");
}

module.exports = function (o) {
    _test_unsqueeze(o);
}