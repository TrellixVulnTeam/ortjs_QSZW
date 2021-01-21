function _test_slice1(o) {
    console.log("==== Slice1 test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[1]: C - [4]

    const A = new Int32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const C = new Int32Array(4);

    const f1 = new InferenceContext(1,  // num operators
                                    2,  // num values
                                    [6, 6] // value types
                                    );

    f1.addAttribute_ints(0, "starts", [0, 0]);
    f1.addAttribute_ints(0, "ends", [2, 2]);
    f1.addAttribute_ints(0, "axes", [0, 1]);
    f1.addAttribute_ints(0, "steps", [1, 1]);

    f1.initKernel(0, "Slice", "", 9,  // op, opset, opset_ver
                  [0], [1],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 4]);  // shape
    
    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    new Int32Array(o.HEAPU8.buffer, offset_0, size_0).set(A);
    
    f1.run();
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    c_out = new Int32Array(o.HEAPU8.buffer, offset_1, size_1);
    C.set(new Int32Array(o.HEAPU8.buffer, offset_1, size_1));
    console.log(C);
    console.log("==== Slice1 test complete. ====");
}

function _test_slice10(o) {
    // TODO: This test is not correct. Fix later. 
    console.log("==== Slice10 test starts. ====");

    const InferenceContext = o.InferenceContext;

    // val[0]: A - [3, 4]
    // val[2]: C - [4]

    const A = new Int32Array(12);
    A.set([0,1,2,3,4,5,6,7,8,9,10,11]);
    const STARTS = new Int32Array(2);
    STARTS.set([0, 0]);
    const ENDS = new Int32Array(2);
    ENDS.set([2, 2]);
    const AXES = new Int32Array(2);
    AXES.set([0, 1]);
    const STEPS = new Int32Array(2);
    STEPS.set([1, 1]);
    const C = new Int32Array(4);
    const f1 = new InferenceContext(1,  // num operators
                                    6,  // num values
                                    [6, 6, 6, 6, 6, 6] // value types
                                    );

    f1.initKernel(0, "Slice", "", 10,  // op, opset, opset_ver
                  [0, 1, 2, 3, 4], [5],    // inputs idx, output idx
                  "");

    f1.setInput(0,        // value idx
                [3, 4]);  // shape
    // starts
    f1.setInput(1,        // value idx
                [2]);  // shape
    // ends
    f1.setInput(2,        // value idx
                [2]);  // shape    
    // axes
    f1.setInput(3,        // value idx
                [2]);  // shape
    // steps
    f1.setInput(4,        // value idx
                [2]);  // shape

    const offset_0 = f1.getTensorData(0);
    const size_0 = f1.getTensorDataSize(0);
    const offset_1 = f1.getTensorData(1);
    const size_1 = f1.getTensorDataSize(1);
    const offset_2 = f1.getTensorData(2);
    const size_2 = f1.getTensorDataSize(2);
    const offset_3 = f1.getTensorData(3);
    const size_3 = f1.getTensorDataSize(3);
    const offset_4 = f1.getTensorData(4);
    const size_4 = f1.getTensorDataSize(4);
    new Int32Array(o.HEAPU8.buffer, offset_0, size_0);
    new Int32Array(o.HEAPU8.buffer, offset_1, size_1).set(STARTS);
    new Int32Array(o.HEAPU8.buffer, offset_2, size_2).set(ENDS);
    new Int32Array(o.HEAPU8.buffer, offset_3, size_3).set(AXES);
    new Int32Array(o.HEAPU8.buffer, offset_4, size_4).set(STEPS);
    
    f1.run();
    const offset_5 = f1.getTensorData(5);
    const size_5 = f1.getTensorDataSize(5);
    c_out = new Int32Array(o.HEAPU8.buffer, offset_5, size_5);
    C.set(new Int32Array(o.HEAPU8.buffer, offset_5, size_5));
    console.log(C);
    console.log("==== Slice1 test complete. ====");
}
module.exports = function (o) {
    _test_slice1(o);
    //_test_slice10(o);
}