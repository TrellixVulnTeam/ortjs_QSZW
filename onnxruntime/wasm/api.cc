#include <emscripten/bind.h>

#include "api.h"

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel_info.h"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(inference_context) {
    class_<InferenceContext>("InferenceContext")
        .constructor<int, std::vector<int>, int, int>()
        .function("setInitializer", &InferenceContext::SetInitializer)
        .function("addKernel", &InferenceContext::AddKernel)
        .function("addAttribute_f", select_overload<void(int, const std::string&, float)>(&InferenceContext::AddAttribute))
        .function("addAttribute_floats", select_overload<void(int, const std::string&, const std::vector<float>&)>(&InferenceContext::AddAttribute))
        .function("addAttribute_i", select_overload<void(int, const std::string&, int)>(&InferenceContext::AddAttribute))
        .function("addAttribute_ints", select_overload<void(int, const std::string&, const std::vector<int>&)>(&InferenceContext::AddAttribute))
        .function("addAttribute_s", select_overload<void(int, const std::string&, const std::string&)>(&InferenceContext::AddAttribute))
        .function("setInput", &InferenceContext::SetInput)
        .function("setOutput", &InferenceContext::SetOutput)
        .function("run", &InferenceContext::Run)
        .function("getTensorData", &InferenceContext::GetTensorData, allow_raw_pointers())
        .function("getTensorShape", &InferenceContext::GetTensorShape);

    register_vector<int>("vector<int>");
    register_vector<float>("vector<float>");
}

namespace {
void CreateMLValue(onnxruntime::AllocatorPtr alloc, const std::vector<int>& dims, onnxruntime::MLDataType element_type, OrtValue* p_mlvalue) {
  std::vector<int64_t> shape(dims.begin(), dims.end());
  std::unique_ptr<onnxruntime::Tensor> p_tensor = onnxruntime::make_unique<onnxruntime::Tensor>(element_type,
                                                                      onnxruntime::TensorShape{shape},
                                                                      alloc);

  p_mlvalue->Init(p_tensor.release(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
}
}

InferenceContext::InferenceContext(int num_values, const std::vector<int>& data_types, int num_inputs, int num_outputs) {
    ORT_ENFORCE(num_values == data_types.size(), "num_values doesn't match size of data_types");
    values_.resize(num_values);
    types_.resize(num_values);
    preserve_.resize(num_values);

    input_indices_.resize(num_inputs);
    output_indices_.resize(num_outputs);

    alloc_ = std::make_shared<::onnxruntime::CPUAllocator>();
}

InferenceContext::~InferenceContext() {}

void InferenceContext::SetInitializer(int index, const std::vector<int>& dims) {
    CreateMLValue(alloc_, dims, types_[index], &values_[index]);
    preserve_[index] = true;
}

int InferenceContext::AddKernel(const std::string& op, const std::string& opset, int opset_version, const std::string varience) {
    // TODO
    // kernels_.emplace_back(...);
    // attributes_.emplace_back(...);

    return kernels_.size() - 1;
}

void InferenceContext::AddAttribute(int kernel_index, const std::string& name, float value) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attr.set_f(value);
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute(int kernel_index, const std::string& name, const std::vector<float>& values) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
    attr.set_floats(values);
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute(int kernel_index, const std::string& name, int value) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr.set_i(value);
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute(int kernel_index, const std::string& name, const std::vector<int>& values) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    attr.set_ints(std::vector<int64_t>(values.begin(), values.end()));
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute(int kernel_index, const std::string& name, const std::string& value) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    attr.set_s(value);
    attributes_[kernel_index][name] = attr;
}

void InferenceContext::SetInput(int index, int value_index, const std::vector<int>& dims) {
    input_indices_[index] = value_index;
    CreateMLValue(alloc_, dims, types_[index], &values_[value_index]);
}

void InferenceContext::SetOutput(int index, int value_index) {
    output_indices_[index] = value_index;
}

void InferenceContext::Run() {
    // TODO
}

void* InferenceContext::GetTensorData(int index) {
    return values_[index].GetMutable<onnxruntime::Tensor>()->MutableDataRaw();
}

std::vector<int> InferenceContext::GetTensorShape(int index) {
    auto &shape = values_[index].Get<onnxruntime::Tensor>().Shape();
    return std::vector<int>(shape.GetDims().begin(), shape.GetDims().end());
}
