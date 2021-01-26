#include "api.h"

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel_info.h"

#include "core/providers/cpu/math/gemm.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/tensor/concat.h"
#include "core/providers/cpu/tensor/gather.h"
#include "core/providers/cpu/math/matmul.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/unsqueeze.h"
#include "core/providers/cpu/tensor/reshape.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/mlas/inc/mlas.h"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(inference_context) {
    class_<InferenceContext>("InferenceContext")
        .constructor<int, int, val>()
        //.function("setTypes", &InferenceContext::SetTypes)
        .function("setInitializer", &InferenceContext::SetInitializer)
        .function("initKernel", &InferenceContext::InitKernel)
        .function("addAttribute_f", &InferenceContext::AddAttribute_f)
        .function("addAttribute_floats", &InferenceContext::AddAttribute_floats)
        .function("addAttribute_i", &InferenceContext::AddAttribute_i)
        .function("addAttribute_ints", &InferenceContext::AddAttribute_ints)
        .function("addAttribute_s", &InferenceContext::AddAttribute_s)
        .function("setInput", &InferenceContext::SetInput)
        .function("run", &InferenceContext::Run)
        .function("getTensorData", &InferenceContext::GetTensorData)
        .function("getTensorDataSize", &InferenceContext::GetTensorDataSize)
        .function("getTensorShape", &InferenceContext::GetTensorShape);

    register_vector<int>("vector<int>");
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

InferenceContext::InferenceContext(int num_kernels,
                                   int num_values,
                                   const emscripten::val& arr_data_types) {
    std::vector<int> types = convertJSArrayToNumberVector<int>(arr_data_types);
    ORT_ENFORCE(num_values == types.size(), "num_values doesn't match size of data_types");
    values_.resize(num_values);
    types_.resize(num_values);
    preserve_.resize(num_values);

    for (size_t i = 0; i < types.size(); i++) {
        types_[i] = onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(types[i])->GetElementType();
    }

    attributes_.resize(num_kernels);
    kernels_.resize(num_kernels);
    kernel_input_indices_.resize(num_kernels);
    kernel_output_indices_.resize(num_kernels);
    kernel_input_arg_count_.resize(num_kernels);

    alloc_ = std::make_shared<::onnxruntime::CPUAllocator>();
}

InferenceContext::~InferenceContext() {
    for (auto* p : kernels_) {
      delete p;
    }
}

void InferenceContext::SetInitializer(int index, const val& arr_dims) {
    std::vector<int> dims = convertJSArrayToNumberVector<int>(arr_dims);
    CreateMLValue(alloc_, dims, types_[index], &values_[index]);
    preserve_[index] = true;
}

void InferenceContext::InitKernel(int index,
                                  const std::string& op,
                                  const std::string& opset,
                                  int opset_version,
                                  const val& arr_input_indices,
                                  const val& arr_output_indices,
                                  const std::string varience) {
    ORT_ENFORCE(index >= 0 && index < kernels_.size(), "index out of range");
    // TODO
    // kernels_.emplace_back(...);
    // attributes_.emplace_back(...);
    
    // naive resolve implementation
    if (op == "Gemm") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::Gemm<float>{info};
    } else if (op == "Add") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::Add<float>{info};
    } else if (op == "Concat") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        kernel_input_arg_count_[index].push_back(kernel_input_indices_[index].size());
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::Concat{info};
    } else if (op == "Gather") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::Gather{info};
    } else if (op == "MatMul") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::MatMul<float>{info};
    } else if (op == "Mul") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::Mul<float>{info};
    }  else if (op == "Reshape") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::Reshape{info};
    } else if (op == "Sigmoid") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::ElementWiseKernel<::onnxruntime::functors::Sigmoid<float>>{info};
    } else if (op == "Slice") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        if (opset_version < 10) {
            kernels_[index] = new ::onnxruntime::Slice1{info};
        } else {
            kernels_[index] = new ::onnxruntime::Slice10{info};
        }
    } else if (op == "Tanh") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::ElementWiseKernel<::onnxruntime::functors::Tanh<float>>{info};
    } else if (op == "Unsqueeze") {
        kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
        kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
        ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                         static_cast<int>(kernel_input_indices_[index].size()),
                                         static_cast<int>(kernel_output_indices_[index].size()));
        kernels_[index] = new ::onnxruntime::Unsqueeze{info};
    } else {
        //error
    }
}

void InferenceContext::AddAttribute_f(int kernel_index, const std::string& name, float value) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attr.set_f(value);
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute_floats(int kernel_index, const std::string& name, const emscripten::val& arr_values) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
    std::vector<float> values = convertJSArrayToNumberVector<float>(arr_values);
    attr.set_floats(values);
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute_i(int kernel_index, const std::string& name, int value) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr.set_i(value);
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute_ints(int kernel_index, const std::string& name, const emscripten::val& arr_values) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    std::vector<int> values = convertJSArrayToNumberVector<int>(arr_values);
    attr.set_ints(std::vector<int64_t>(values.begin(), values.end()));
    attributes_[kernel_index][name] = attr;
}
void InferenceContext::AddAttribute_s(int kernel_index, const std::string& name, const std::string& value) {
    onnxruntime::AttributeStub attr;
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    attr.set_s(value);
    attributes_[kernel_index][name] = attr;
}

void InferenceContext::SetInput(int index, const emscripten::val& arr_dims) {
    std::vector<int> dims = convertJSArrayToNumberVector<int>(arr_dims);
    CreateMLValue(alloc_, dims, types_[index], &values_[index]);
}

void InferenceContext::Run() {
    for (size_t i = 0; i < kernels_.size(); i++) {
        onnxruntime::OpKernelContext ctx{values_.data(),
                                         types_.data(),
                                         kernels_[i],
                                         nullptr,
                                         kernel_input_indices_[i],
                                         kernel_output_indices_[i]};
#ifndef NDEBUG
        printf("running kernel %d\n", (int)(i));
        for (size_t j = 0; j < kernel_input_indices_[i].size(); j++) {
            auto t = ctx.Input<onnxruntime::Tensor>(j);
            std::cout<<"input"<<j<<" ["<<kernel_input_indices_[i][j]<<"/"<<values_.size()<<"]: "<<t->Shape().ToString()<<" "<<t->DataRaw()<<std::endl;
        }
#endif
        ORT_ENFORCE(kernels_[i]->Compute(&ctx).IsOK(),
                    "failed to run kernel");
#ifndef NDEBUG
        for (size_t j = 0; j < kernel_output_indices_[i].size(); j++) {
            auto &t = values_[kernel_output_indices_[i][j]].Get<onnxruntime::Tensor>();
            std::cout<<"output"<<j<<" ["<<kernel_output_indices_[i][j]<<"/"<<values_.size()<<"]: "<<t.Shape().ToString()<<" "<<t.DataRaw()<<std::endl;
        }
#endif
    }
}

size_t InferenceContext::GetTensorData(int index) {
    return reinterpret_cast<size_t>(values_[index].GetMutable<onnxruntime::Tensor>()->MutableDataRaw());
}

size_t InferenceContext::GetTensorDataSize(int index) {
    return values_[index].GetMutable<onnxruntime::Tensor>()->Shape().Size();
}

emscripten::val InferenceContext::GetTensorShape(int index) {
    auto &shape = values_[index].Get<onnxruntime::Tensor>().Shape().GetDims();
    return emscripten::val::array(std::vector<int>(shape.begin(), shape.end()));
}
