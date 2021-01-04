const wasm_factory = require('./out_wasm_main');

wasm_factory().then((o) => {
    console.log(o);

    const f = o.InferenceContext;

    const f1 = new f(3, [1,1,1], 2, 1);
});

//wasm_factory();
