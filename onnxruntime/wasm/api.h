#pragma once

#include <emscripten.h>
#include <string>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel.h"

extern "C" {

class InferenceContext {
public:
    InferenceContext(int num_values, const std::vector<int>& data_types, int num_inputs, int num_outputs);
    ~InferenceContext();

    void SetInitializer(int index, const std::vector<int>& dims);
    int AddKernel(const std::string& op, const std::string& opset, int opset_version, const std::string varience);

    void AddAttribute(int kernel_index, const std::string& name, float value);
    void AddAttribute(int kernel_index, const std::string& name, const std::vector<float>& values);
    void AddAttribute(int kernel_index, const std::string& name, int value);
    void AddAttribute(int kernel_index, const std::string& name, const std::vector<int>& values);
    void AddAttribute(int kernel_index, const std::string& name, const std::string& value);

    void SetInput(int index, int value_index, const std::vector<int>& dims);
    void SetOutput(int index, int value_index);
    void Run();

    void* GetTensorData(int index);
    std::vector<int> GetTensorShape(int index);

private:
    std::vector<OrtValue> values_;
    std::vector<onnxruntime::MLDataType> types_;
    std::vector<bool> preserve_;

    std::vector<int> input_indices_;
    std::vector<int> output_indices_;

    onnxruntime::AllocatorPtr alloc_;
    std::vector<onnxruntime::OpKernel> kernels_;
    std::vector<onnxruntime::NodeAttributes> attributes_;
};
};
