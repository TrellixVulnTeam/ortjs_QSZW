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
#include "core/providers/cpu/tensor/resize.h"
#include "core/providers/cpu/nn/conv.h"
#include "contrib_ops/cpu/fused_conv.h"
#include "contrib_ops/cpu/nchwc_ops.h"
#include "core/mlas/inc/mlas.h"

#include <thread>

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

    kernel_names_.resize(num_kernels);
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
    kernel_names_[index] = op;
    kernel_input_indices_[index] = convertJSArrayToNumberVector<int>(arr_input_indices);
    kernel_output_indices_[index] = convertJSArrayToNumberVector<int>(arr_output_indices);
    if (op == "Concat") {
        kernel_input_arg_count_[index].push_back(kernel_input_indices_[index].size());
    }
    ::onnxruntime::OpKernelInfo info(alloc_, attributes_[index], kernel_input_arg_count_[index],
                                        opset_version,
                                        static_cast<int>(kernel_input_indices_[index].size()),
                                        static_cast<int>(kernel_output_indices_[index].size()));
    // naive resolve implementation
    if (op == "Gemm") {
        kernels_[index] = new ::onnxruntime::Gemm<float>{info};
    } else if (op == "Add") {
        kernels_[index] = new ::onnxruntime::Add<float>{info};
    } else if (op == "Concat") {
        kernels_[index] = new ::onnxruntime::Concat{info};
    } else if (op == "Conv") {
        kernels_[index] = new ::onnxruntime::Conv<float>{info};
    } else if (op == "FusedConv") {
        kernels_[index] = new ::onnxruntime::contrib::FusedConvFloat{info};
    } else if (op == "ConvNchwc") {
        kernels_[index] = new ::onnxruntime::contrib::NchwcConv{info};
    } else if (op == "ReorderInput") {
        kernels_[index] = new ::onnxruntime::contrib::ReorderInput{info};
    } else if (op == "ReorderOutput") {
        kernels_[index] = new ::onnxruntime::contrib::ReorderOutput{info};
    } else if (op == "Gather") {
        kernels_[index] = new ::onnxruntime::Gather{info};
    } else if (op == "MatMul") {
        kernels_[index] = new ::onnxruntime::MatMul<float>{info};
    } else if (op == "Mul") {
        kernels_[index] = new ::onnxruntime::Mul<float>{info};
    } else if (op == "Relu") {
        kernels_[index] = new ::onnxruntime::ElementWiseKernel<::onnxruntime::functors::Relu<float>>{info};
    } else if (op == "Reshape") {
        kernels_[index] = new ::onnxruntime::Reshape{info};
    } else if (op == "Resize") {
        kernels_[index] = new ::onnxruntime::Resize<float>{info};
    } else if (op == "Sigmoid") {
        kernels_[index] = new ::onnxruntime::ElementWiseKernel<::onnxruntime::functors::Sigmoid<float>>{info};
    } else if (op == "Slice") {
        if (opset_version < 10) {
            kernels_[index] = new ::onnxruntime::Slice1{info};
        } else {
            kernels_[index] = new ::onnxruntime::Slice10{info};
        }
    } else if (op == "Tanh") {
        kernels_[index] = new ::onnxruntime::ElementWiseKernel<::onnxruntime::functors::Tanh<float>>{info};
    } else if (op == "Unsqueeze") {
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
    try {
        std::vector<std::pair<uint64_t, uint64_t>> op_time_stamps;
        op_time_stamps.reserve(kernels_.size());
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
            uint64_t op_start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            ORT_ENFORCE(kernels_[i]->Compute(&ctx).IsOK(),
                        "failed to run kernel");
            uint64_t op_end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            op_time_stamps.emplace_back(op_start, op_end);

    #ifndef NDEBUG
            for (size_t j = 0; j < kernel_output_indices_[i].size(); j++) {
                auto &t = values_[kernel_output_indices_[i][j]].Get<onnxruntime::Tensor>();
                std::cout<<"output"<<j<<" ["<<kernel_output_indices_[i][j]<<"/"<<values_.size()<<"]: "<<t.Shape().ToString()<<" "<<t.DataRaw()<<std::endl;
            }
    #endif
        }

        if (op_time_stamps.size()) {
            std::stringstream ss;
            ss << "Start Run() from [thread:" << std::this_thread::get_id() << "] at " <<  op_time_stamps.front().first << std::endl;
            for (size_t i = 0; i < kernels_.size(); i++) {
                const auto& op_time_stamp = op_time_stamps[i];
                ss << "    Kernel " << i << ", op_name:" << kernel_names_[i];
                ss << ", Start:" << op_time_stamp.first << ", End:" << op_time_stamp.second;
                ss << ", latency:" << (op_time_stamp.second - op_time_stamp.first) << " us";
                ss << std::endl;
                std::cout << ss.str();
                ss.str(std::string());
            }
            ss << "End Run() from [thread:" << std::this_thread::get_id() << "] at " <<  op_time_stamps.back().second;
            ss << " total Run() latency:" << (op_time_stamps.back().second - op_time_stamps.front().first) << " us" << std::endl;
            std::cout << ss.str() << std::endl;
            ss.str(std::string());
        }
    } catch (...) {
        std::cout << "Exception happended" << std::endl;
        throw;
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
