// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/graph/graph_viewer.h"
#include "gsl/gsl"
#else
#include "core/framework/allocator.h"
#endif

#include "core/framework/op_node_proto_helper.h"

namespace onnxruntime {

#if defined(__wasm__)
// TODO: replace Node
// (NOTE by yulong) Node class is a huge class that connects a lot of implementations and connects to Graph. we dont want them.
//                  we just want to support functions that operators impl need.
class NodeStub {
public:
  explicit NodeStub(const NodeAttributes& attributes, const std::vector<int>& input_arg_count): 
    attributes_(attributes), input_arg_count_(input_arg_count) {}
  ~NodeStub() = default;

  const NodeAttributes& GetAttributes() const noexcept { return attributes_; }
  /** Gets the count of arguments for each of the Node's explicit inputs. */
  const std::vector<int>& InputArgCount() const noexcept { return input_arg_count_; }

private:
  const NodeAttributes& attributes_;
  // input/output defs and arg count
  std::vector<int> input_arg_count_;
};
#endif

#if !defined(__wasm__)
class OrtValueNameIdxMap;
class FuncManager;
class DataTransferManager;
#endif

// A very light-weight class, which works as an aggregated
// view of all data needed for constructing a Kernel instance.
// NOTE: it does not own/hold any objects.
class OpKernelInfo : public OpNodeProtoHelper<ProtoHelperNodeContext> {
 public:
#if !defined(__wasm__)
  explicit OpKernelInfo(const onnxruntime::Node& node,
                        const KernelDef& kernel_def,
                        const IExecutionProvider& execution_provider,
                        const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                        const OrtValueNameIdxMap& mlvalue_name_idx_map,
                        const FuncManager& funcs_mgr,
                        const DataTransferManager& data_transfer_mgr);
#else
  explicit OpKernelInfo(AllocatorPtr allocator,
                        const onnxruntime::NodeAttributes& attributes,
                        const std::vector<int>& input_arg_count,
                        size_t num_inputs,
                        size_t num_outputs);
#endif

  OpKernelInfo(const OpKernelInfo& other);

  AllocatorPtr GetAllocator(int device_id, OrtMemType mem_type) const;

#if !defined(__wasm__)
  const OrtMemoryInfo& GetMemoryInfo(int device_id, OrtMemType mem_type) const;

  const KernelDef& GetKernelDef() const;

  const IExecutionProvider* GetExecutionProvider() const noexcept;

  const DataTransferManager& GetDataTransferManager() const noexcept;

  const onnxruntime::Node& node() const noexcept;

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const;

  common::Status GetFusedFuncs(NodeComputeInfo*& compute_info) const;
#else
  const onnxruntime::NodeStub& node() const noexcept { return node_; }
#endif

 private:
  ORT_DISALLOW_MOVE(OpKernelInfo);
  ORT_DISALLOW_ASSIGNMENT(OpKernelInfo);

#if !defined(__wasm__)
  const onnxruntime::Node& node_;
  const KernelDef& kernel_def_;
  // For non cpu/cuda case, this pointer should be set so that function kernel
  // will delegate kernel compute call to <execution_provider> compute call.
  gsl::not_null<const ::onnxruntime::IExecutionProvider*> execution_provider_;
  const std::unordered_map<int, OrtValue>& constant_initialized_tensors_;
  const OrtValueNameIdxMap& ort_value_name_idx_map_;
  const FuncManager& funcs_mgr_;
  const DataTransferManager& data_transfer_mgr_;
#else
  onnxruntime::NodeStub node_;
  AllocatorPtr allocator_;
#endif
  ProtoHelperNodeContext proto_helper_context_;
};

}  // namespace onnxruntime
