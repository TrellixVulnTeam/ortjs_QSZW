const wasm_factory = require('../out_wasm_main');
const ALL_TESTS = [
    require('./test_gemm.json'),
    require('./test_binary.json'),
    require('./test_concat.json'),
    require('./test_gather.json'),
    require('./test_matmul.json'),
    require('./test_slice.json'),
    require('./test_unsqueeze.json'),
    require('./test_reshape.json'),
    require('./test_activations.json')
];

var expect = require('chai').expect;

DATA_TYPE_TO_ARRAY_OBJECT_MAP = {
    // TODO: ADD MORE
    1: Float32Array,
    6: Int32Array,
    7: BigInt64Array
};

function runOpTestcase(o, testcase) {
    const InferenceContext = o.InferenceContext;

    const num_values = testcase.inputs.length + testcase.initializers.length + testcase.outputs.length;
    var value_types = new Array(num_values); // value_types[value index] = type
    var input_idx_to_data = {};
    var input_indices = [];
    var init_idx_to_data = {};
    var output_indices = [];
    // Create inputs
    testcase.inputs.forEach(i => {
        var input;
        const size = i.dims.reduce((a, b) => a * b);
        input = new DATA_TYPE_TO_ARRAY_OBJECT_MAP[i.type](size);
        if (i.type == 7) {
            input.set(i.data.map(i => BigInt(i)));
        } else {
            input.set(i.data);
        }
        value_types[i.value_idx] = i.type;
        input_idx_to_data[i.value_idx] = input;
        input_indices.push(i.value_idx);
    });

    // Create outputs 
    testcase.outputs.forEach(i => {
        var output;
        const size = i.dims.reduce((a, b) => a * b);
        output = new DATA_TYPE_TO_ARRAY_OBJECT_MAP[i.type](size);
        value_types[i.value_idx] = i.type;
        output_indices.push(i.value_idx);
    });

    testcase.initializers.map(i => {
        value_types[i.value_idx] = i.type;
    });

    const f1 = new InferenceContext(1,  // num operators
        num_values,  // num values
        value_types, // value types
    );

    // Create initializer 
    testcase.initializers.forEach(i => {
        f1.setInitializer(i.value_idx,             // value index
            i.dims);       // dim
        var initializer;
        const size = i.dims.reduce((a, b) => a * b);
        initializer = new DATA_TYPE_TO_ARRAY_OBJECT_MAP[i.type](size);
        if (i.type == 7) {
            initializer.set(i.data.map(i => BigInt(i)));
        } else {
            initializer.set(i.data);
        }
        input_indices.push(i.value_idx);
        init_idx_to_data[i.value_idx] = initializer;
    });
    // set attributes
    testcase.attributes.forEach(i => {
        if (i.type == "float") {
            f1.addAttribute_f(0, i.name, i.data);
        } else if (i.type == "int") {
            f1.addAttribute_i(0, i.name, i.data);
        } else if (i.type == "ints") {
            f1.addAttribute_ints(0, i.name, i.data);
        } else if (i.type == "floats") {
            f1.addAttribute_floats(0, i.name, i.data);
        } else if (i.type == "string") {
            f1.addAttribute_s(0, i.name, i.data);
        }
    });

    f1.initKernel(0, testcase.operator, "", testcase.opset,  // op, opset, opset_ver
        input_indices, output_indices,    // inputs idx, output idx
        "");

    // Set initializer
    testcase.initializers.forEach(i => {
        const offset = f1.getTensorData(i.value_idx);
        const size = f1.getTensorDataSize(i.value_idx);
        new DATA_TYPE_TO_ARRAY_OBJECT_MAP[i.type](o.HEAPU8.buffer, offset, size).set(init_idx_to_data[i.value_idx]);
    })

    // Set input
    testcase.inputs.forEach(i => {
        f1.setInput(i.value_idx, i.dims);
        const offset = f1.getTensorData(i.value_idx);
        const size = f1.getTensorDataSize(i.value_idx);
        new DATA_TYPE_TO_ARRAY_OBJECT_MAP[i.type](o.HEAPU8.buffer, offset, size).set(input_idx_to_data[i.value_idx]);
    })

    f1.run();

    var results = [];
    testcase.outputs.forEach(i => {
        const offset = f1.getTensorData(i.value_idx);
        const size = f1.getTensorDataSize(i.value_idx);
        const c_out = new DATA_TYPE_TO_ARRAY_OBJECT_MAP[i.type](o.HEAPU8.buffer, offset, size);
        results.push(c_out);
    })

    testcase.outputs.forEach((expectedOutput, i) => {
        // check output size and shape
        expect(results[i].length, 'size of output tensors').to.equal(expectedOutput.data.length);
        expect(f1.getTensorShape(expectedOutput.value_idx), 'dims of output tensors').to.eql(expectedOutput.dims);
        for (var idx = 0; idx < results[i].length; idx++) {
            // check output value
            expect(results[i][idx]).equals(expectedOutput.data[idx]);            
        }
    });
}

describe('#OPTESTS#', function () {
    var ob;
    // const buffer;
    before(() => {
        return wasm_factory().then((o) => {
            ob = o;
        });
    });

    for (const test of ALL_TESTS) {
        for (const testCase of test) {
            describe(`#OPTESTS# - ${testCase.operator}`, function () {
                it(testCase.name, function () {
                    runOpTestcase(ob, testCase);
                });
            });
        }
    }
});
