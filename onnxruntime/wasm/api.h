#pragma once

#include <emscripten.h>
#include <emscripten/bind.h>
#include <string>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel.h"

extern "C" {

class InferenceContext {
public:
    InferenceContext(int num_kernels, int num_values, const emscripten::val& arr_data_types, int num_inputs, int num_outputs);
    ~InferenceContext();

    void SetInitializer(int index, const emscripten::val& arr_dims);
    void InitKernel(int index, const std::string& op, const std::string& opset, int opset_version,
                    const emscripten::val& arr_input_indices, const emscripten::val& arr_output_indices, const std::string varience);

    void AddAttribute_f(int kernel_index, const std::string& name, float value);
    void AddAttribute_floats(int kernel_index, const std::string& name, const emscripten::val& arr_values);
    void AddAttribute_i(int kernel_index, const std::string& name, int value);
    void AddAttribute_ints(int kernel_index, const std::string& name, const emscripten::val& arr_values);
    void AddAttribute_s(int kernel_index, const std::string& name, const std::string& value);

    void SetInput(int index, int value_index, const emscripten::val& arr_dims);
    void SetOutput(int index, int value_index);
    void Run();

    size_t GetTensorData(int index);
    size_t GetTensorDataSize(int index);
    std::vector<int> GetTensorShape(int index);

private:
    std::vector<OrtValue> values_;
    std::vector<onnxruntime::MLDataType> types_;
    std::vector<bool> preserve_;

    std::vector<int> input_indices_;
    std::vector<int> output_indices_;

    onnxruntime::AllocatorPtr alloc_;
    std::vector<onnxruntime::OpKernel*> kernels_;
    std::vector<std::vector<int>> kernel_input_indices_;
    std::vector<std::vector<int>> kernel_output_indices_;
    std::vector<onnxruntime::NodeAttributes> attributes_;
};
};
