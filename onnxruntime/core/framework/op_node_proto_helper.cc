// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/op.h"
#include "gsl/gsl"

#if !defined(__wasm__)
using namespace ONNX_NAMESPACE;
#endif
using namespace ::onnxruntime::common;
namespace onnxruntime {

#if defined(__wasm__)
using AttributeProto = onnxruntime::AttributeStub;
using ONNX_NAMESPACE::AttributeProto_AttributeType;
using ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED;
using ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT;
using ONNX_NAMESPACE::AttributeProto_AttributeType_INT;
using ONNX_NAMESPACE::AttributeProto_AttributeType_STRING;
using ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR;
using ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH;
using ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSOR;
using ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS;
using ONNX_NAMESPACE::AttributeProto_AttributeType_INTS;
using ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS;
using ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS;
using ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS;
using ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSORS;

template <class T>
bool HasTyped(const AttributeProto*);

template <>
inline bool HasTyped<float>(const AttributeProto* attr) {
  return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT;
}
template <>
inline bool HasTyped<int64_t>(const AttributeProto* attr) {
  return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT;
}
template <>
inline bool HasTyped<std::string>(const AttributeProto* attr) {
  return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRINGS;
}

// template <>
// inline bool HasTyped<TensorProto>(const AttributeProto* attr) {
//   return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_TENSOR;
// }
// template <>
// inline bool HasTyped<GraphProto>(const AttributeProto* attr) {
//   return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_GRAPH;
// }

template <typename T>
inline bool HasTypedList(const AttributeProto* attr);

template <>
inline bool HasTypedList<float>(const AttributeProto* attr) {
  return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOATS;
}

template <>
inline bool HasTypedList<int64_t>(const AttributeProto* attr) {
  return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INTS;
}

template <>
inline bool HasTypedList<std::string>(const AttributeProto* attr) {
  return attr->type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRINGS;
}
#else

template <class T>
bool HasTyped(const AttributeProto*);

template <>
inline bool HasTyped<float>(const AttributeProto* attr) {
  return utils::HasFloat(*attr);
}
template <>
inline bool HasTyped<int64_t>(const AttributeProto* attr) {
  return utils::HasInt(*attr);
}
template <>
inline bool HasTyped<std::string>(const AttributeProto* attr) {
  return utils::HasString(*attr);
}

template <>
inline bool HasTyped<TensorProto>(const AttributeProto* attr) {
  return utils::HasTensor(*attr);
}
template <>
inline bool HasTyped<GraphProto>(const AttributeProto* attr) {
  return utils::HasGraph(*attr);
}

template <typename T>
inline bool HasTypedList(const AttributeProto* attr);

template <>
inline bool HasTypedList<float>(const AttributeProto* attr) {
  return utils::HasFloats(*attr);
}

template <>
inline bool HasTypedList<int64_t>(const AttributeProto* attr) {
  return utils::HasInts(*attr);
}

template <>
inline bool HasTypedList<std::string>(const AttributeProto* attr) {
  return utils::HasStrings(*attr);
}

#endif

template <typename T>
inline constexpr int ArrayTypeToAttributeType();

template <>
inline constexpr int ArrayTypeToAttributeType<float>() {
  return AttributeProto_AttributeType_FLOATS;
}

template <>
inline constexpr int ArrayTypeToAttributeType<int64_t>() {
  return AttributeProto_AttributeType_INTS;
}

template <>
inline constexpr int ArrayTypeToAttributeType<std::string>() {
  return AttributeProto_AttributeType_STRINGS;
}

#define ORT_DEFINE_GET_ATTR(IMPL_T, T, type)                                                       \
  template <>                                                                                      \
  template <>                                                                                      \
  Status OpNodeProtoHelper<IMPL_T>::GetAttr<T>(                                                    \
      const std::string& name, T* value) const {                                                   \
    const AttributeProto* attr = TryGetAttribute(name);                                            \
    if (!attr) {                                                                                   \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No attribute with name:'", name, "'is defined."); \
    }                                                                                              \
    if (!HasTyped<T>(attr)) {                                                                      \
      return Status(ONNXRUNTIME, FAIL, "Attibute name and type don't match");                      \
    } else {                                                                                       \
      *value = static_cast<T>(attr->type());                                                       \
      return Status::OK();                                                                         \
    }                                                                                              \
  }

#define ORT_DEFINE_GET_ATTRS(IMPL_T, T, list)                                      \
  template <>                                                                      \
  template <>                                                                      \
  Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                                   \
      const std::string& name, std::vector<T>& values) const {                     \
    const AttributeProto* attr = TryGetAttribute(name);                            \
    if (!attr) {                                                                   \
      return Status(ONNXRUNTIME, FAIL, "No attribute with this name is defined."); \
    }                                                                              \
    values.reserve(attr->list##_size());                                           \
    for (int i = 0; i < attr->list##_size(); ++i) {                                \
      values.push_back(static_cast<T>(attr->list(i)));                             \
    }                                                                              \
    return Status::OK();                                                           \
  }                                                                                \
  template <>                                                                      \
  template <>                                                                      \
  Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                                   \
      const std::string& name, gsl::span<T> values) const {                        \
    const AttributeProto* attr = TryGetAttribute(name);                            \
    if (!attr) {                                                                   \
      return Status(ONNXRUNTIME, FAIL, "No attribute with this name is defined."); \
    }                                                                              \
    ORT_ENFORCE(values.size() == static_cast<size_t>(attr->list##_size()));        \
    for (int i = 0; i < attr->list##_size(); ++i) {                                \
      values[i] = static_cast<T>(attr->list(i));                                   \
    }                                                                              \
    return Status::OK();                                                           \
  }

// Will not work for std::strings
#define ORT_DEFINE_GET_ATTRS_AS_SPAN(IMPL_T, T, list)                                              \
  template <>                                                                                      \
  template <>                                                                                      \
  Status OpNodeProtoHelper<IMPL_T>::GetAttrsAsSpan<T>(                                             \
      const std::string& name, gsl::span<const T>& values) const {                                 \
    const AttributeProto* attr = TryGetAttribute(name);                                            \
    if (!attr) {                                                                                   \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No attribute with name: ", name, " is defined."); \
    }                                                                                              \
    if (!HasTypedList<T>(attr)) {                                                                  \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attribute: ", name,                               \
                             " expected to be of type: ",                                          \
                             AttributeProto::AttributeType_Name(ArrayTypeToAttributeType<T>()),    \
                             " but is of type: ",                                                  \
                             AttributeProto::AttributeType_Name(attr->type()));                    \
    }                                                                                              \
    values = gsl::make_span<const T>(reinterpret_cast<const T*>(attr->list().data()),              \
                                     attr->list##_size());                                         \
    return Status::OK();                                                                           \
  }

#if !defined(ORT_MINIMAL_BUILD)
#define ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(type, list)   \
  ORT_DEFINE_GET_ATTR(ProtoHelperNodeContext, type, list) \
  ORT_DEFINE_GET_ATTR(InferenceContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(type, list)   \
  ORT_DEFINE_GET_ATTRS(ProtoHelperNodeContext, type, list) \
  ORT_DEFINE_GET_ATTRS(InferenceContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(type, list) \
  ORT_DEFINE_GET_ATTRS_AS_SPAN(ProtoHelperNodeContext, type, list)

#else
#define ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(type, list) \
  ORT_DEFINE_GET_ATTR(ProtoHelperNodeContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(type, list) \
  ORT_DEFINE_GET_ATTRS(ProtoHelperNodeContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(type, list) \
  ORT_DEFINE_GET_ATTRS_AS_SPAN(ProtoHelperNodeContext, type, list)
#endif

ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(float, f)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(int64_t, i)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(std::string, s)
#if !defined(__wasm__)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(TensorProto, t)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(GraphProto, g)
#endif
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(float, floats)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(int64_t, ints)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(std::string, strings)
#if !defined(__wasm__)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(TensorProto, tensors)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(GraphProto, graphs)
#endif

ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(float, floats)
ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(int64_t, ints)

template <typename Impl_t>
MUST_USE_RESULT Status OpNodeProtoHelper<Impl_t>::GetAttrsStringRefs(
    const std::string& name,
    std::vector<std::reference_wrapper<const std::string>>& refs) const {
  const AttributeProto* attr = TryGetAttribute(name);
  if (!attr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No attribute with name: ", name, " is defined.");
  }
  if (!HasTypedList<std::string>(attr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Requested attribute: ",
                           name, " is expected to have type: ",
                           AttributeProto::AttributeType_Name(AttributeProto_AttributeType_STRINGS),
                           " but is of type: ",
                           AttributeProto::AttributeType_Name(attr->type()));
  }
  std::vector<std::reference_wrapper<const std::string>> result;
  if (attr->strings_size() > 0) {
    result.reserve(attr->strings_size());
    std::copy(attr->strings().cbegin(), attr->strings().cend(), std::back_inserter(result));
  }
  refs.swap(result);
  return Status::OK();
}

size_t ProtoHelperNodeContext::getNumInputs() const {
#if !defined(__wasm__)
  return node_.InputDefs().size();
#else
  return num_inputs_;
#endif
}

size_t ProtoHelperNodeContext::getNumOutputs() const {
#if !defined(__wasm__)
  return node_.InputDefs().size();
#else
  return num_outputs_;
#endif
}

const AttributeProto* ProtoHelperNodeContext::getAttribute(const std::string& name) const {
#if !defined(__wasm__)
  const onnxruntime::NodeAttributes& attributes = node_.GetAttributes();
#else
  const onnxruntime::NodeAttributes& attributes = attributes_;
#endif
  auto it = attributes.find(name);
  if (it != attributes.end()) {
    return &it->second;
  }
  return nullptr;
}

#if !defined(__wasm__)
const TypeProto* ProtoHelperNodeContext::getInputType(size_t index) const {
  return node_.InputDefs()[index]->TypeAsProto();
}

const TypeProto* ProtoHelperNodeContext::getOutputType(size_t index) const {
  return node_.OutputDefs()[index]->TypeAsProto();
}
#endif

template <class Impl_t>
uint32_t OpNodeProtoHelper<Impl_t>::GetPrimitiveAttrElementCount(AttributeProto_AttributeType type,
                                                                 const std::string& name) const noexcept {
  const AttributeProto* attr = impl_->getAttribute(name);
  if (attr) {
    switch (type) {
      case AttributeProto_AttributeType_FLOAT:
      case AttributeProto_AttributeType_INT:
      case AttributeProto_AttributeType_STRING:
        return 1;

      case AttributeProto_AttributeType_FLOATS:
        return attr->floats_size();
      case AttributeProto_AttributeType_INTS:
        return attr->ints_size();
      case AttributeProto_AttributeType_STRINGS:
        return attr->strings_size();

        // The following are unsupported through this method
      case AttributeProto_AttributeType_UNDEFINED:
      case AttributeProto_AttributeType_TENSOR:
      case AttributeProto_AttributeType_GRAPH:
      case AttributeProto_AttributeType_SPARSE_TENSOR:
      case AttributeProto_AttributeType_TENSORS:
      case AttributeProto_AttributeType_GRAPHS:
      case AttributeProto_AttributeType_SPARSE_TENSORS:
      default:
        return 0;
    }
  }

  return 0;
}

template <class Impl_t>
bool OpNodeProtoHelper<Impl_t>::HasPrimitiveAttribute(AttributeProto_AttributeType type,
                                                      const std::string& name) const noexcept {
  return GetPrimitiveAttrElementCount(type, name) > 0;
}

template class OpNodeProtoHelper<ProtoHelperNodeContext>;
#if !defined(ORT_MINIMAL_BUILD)
template class OpNodeProtoHelper<InferenceContext>;
#endif

}  // namespace onnxruntime
