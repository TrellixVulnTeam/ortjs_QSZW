// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/graph/graph_viewer.h"
#include "gsl/gsl"

#ifdef __has_attribute
#define ORT_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define ORT_HAVE_ATTRIBUTE(x) 0
#endif

#if ORT_HAVE_ATTRIBUTE(nodiscard)
#define MUST_USE_RESULT [[nodiscard]]
#elif defined(__clang__) && ORT_HAVE_ATTRIBUTE(warn_unused_result)
#define MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define MUST_USE_RESULT
#endif

class IMLOpKernel;

namespace onnxruntime {

#if defined(__wasm__)
struct AttributeStub {
  //static std::unique_ptr<AttributeStub> Create() { return onnxruntime::make_unique<AttributeStub>(); }
  //void operator=(const AttributeStub& v) = default;
  //static void operator delete(void* p) { g_host->Provider_AttributeProto__operator_delete(reinterpret_cast<AttributeStub*>(p)); }

  ONNX_NAMESPACE::AttributeProto_AttributeType type() const { return type_; }
  int ints_size() const { return static_cast<int>(ints_.size()); }
  int floats_size() const { return static_cast<int>(floats_.size()); }
  int strings_size() const { return static_cast<int>(strings_.size()); }
  const std::vector<int64_t>& ints() const { return ints_; }
  int64_t ints(int i) const { return ints_[i]; }
  void set_ints(const std::vector<int64_t>& values) { ints_ = values; }
  int64_t i() const { return i_; }
  void set_i(int64_t value) { i_ = value; }
  const std::vector<float>& floats() const { return floats_; }
  float floats(int i) const { return floats_[i]; }
  void set_floats(const std::vector<float>& values) { floats_ = values; }
  float f() const { return f_; }
  void set_f(float value) { f_ = value; }
  const std::vector<std::string>& strings() const { return strings_; }
  const ::std::string& strings(int i) const { return strings_[i]; }
  void set_strings(const std::vector<std::string>& values) { strings_ = values; }
  const ::std::string& s() const { return s_; }
  void set_s(const ::std::string& value) { s_ = value; }
  // void set_name(const ::std::string& value) { return g_host->Provider_AttributeProto__set_name(this, value); }
  void set_type(ONNX_NAMESPACE::AttributeProto_AttributeType value) { type_ = value; }
  // Provider_TensorProto* add_tensors() { return g_host->Provider_AttributeProto__add_tensors(this); }

  AttributeStub() = default;
  AttributeStub(const AttributeStub&) = default;

  template<typename T>
  static inline const std::string& AttributeType_Name(T enum_t_value) {
    //return ONNX_NAMESPACE::AttributeProto::AttributeType_Name(enum_t_value);
    static std::string dummy = "";
    return dummy;
  }

 private:
  ::onnx::AttributeProto_AttributeType type_;
  std::vector<int64_t> ints_;
  int64_t i_;
  std::vector<float> floats_;
  float f_;
  std::vector<std::string> strings_;
  std::string s_;
};

#endif

#if !defined(__wasm__)
  using AttributeType = ONNX_NAMESPACE::AttributeProto;
#else
  using AttributeType = AttributeStub;
#endif


/**
   A set of wrappers with common signatures for use with both OpKernelInfo
   (as its base class) and InferenceContext.  Used by ABI kernels for both
   shape / type inference and kernel construction
*/
template <class Impl_t>
class OpNodeProtoHelper {
 public:
  explicit OpNodeProtoHelper(const Impl_t* impl) : impl_(impl) {}

  /**
     Get a single attribute
     Call this function for a required attribute or when a default value for an optional attribute is specified in the op schema
  */
  template <typename T>
  MUST_USE_RESULT Status GetAttr(const std::string& name, T* value) const;

  /**
     Get a single attribute
     Call this function only when a default value for an optional attribute isn't specified in the op schema
  */
  template <typename T>
  T GetAttrOrDefault(const std::string& name, const T& default_value) const {
    T tmp;
    return GetAttr<T>(name, &tmp).IsOK() ? tmp : default_value;
  }

  /**
     Get a single attribute
     Call this function only when a default value for an optional attribute isn't specified in the op schema
  */
  template <typename T>
  void GetAttrOrDefault(const std::string& name, T* value, const T& default_value) const {
    if (!GetAttr<T>(name, value).IsOK())
      *value = default_value;
  }

  /**
     Get repeated attributes
     Call this function only when a default value for an optional attribute isn't specified in the op schema
  */
  template <typename T>
  MUST_USE_RESULT std::vector<T> GetAttrsOrDefault(const std::string& name, const std::vector<T>& default_value = std::vector<T>{}) const {
    std::vector<T> tmp;
    return GetAttrs<T>(name, tmp).IsOK() ? tmp : default_value;
  }

  /**
     Get repeated attributes
  */
  template <typename T>
  MUST_USE_RESULT Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  template <typename T>
  MUST_USE_RESULT Status GetAttrs(const std::string& name, gsl::span<T> values) const;

  /// <summary>
  /// Return a gsl::span that points to an array of primitive types held by AttributeProto
  /// This function allows to avoid copying big attributes locally into a kernel and operate on
  /// AttributeProto data directly.
  ///
  ///  Does not apply to strings, Tensors and Sparse Tensors that require special treatment.
  /// </summary>
  /// <typeparam name="T">Primitive type contained in the array</typeparam>
  /// <param name="name">Attribute name</param>
  /// <param name="values">Attribute data in a span, out parameter</param>
  /// <returns>Status</returns>
  template <typename T>
  MUST_USE_RESULT Status GetAttrsAsSpan(const std::string& name, gsl::span<const T>& values) const;

  MUST_USE_RESULT Status GetAttrsStringRefs(const std::string& name,
                                            std::vector<std::reference_wrapper<const std::string>>& refs) const;

  uint32_t GetPrimitiveAttrElementCount(ONNX_NAMESPACE::AttributeProto_AttributeType type,
                                        const std::string& name) const noexcept;

  bool HasPrimitiveAttribute(ONNX_NAMESPACE::AttributeProto_AttributeType type,
                             const std::string& name) const noexcept;

  uint32_t GetInputCount() const {
    return gsl::narrow_cast<uint32_t>(impl_->getNumInputs());
  }

  uint32_t GetOutputCount() const {
    return gsl::narrow_cast<uint32_t>(impl_->getNumOutputs());
  }

#if !defined(__wasm__)
  const ONNX_NAMESPACE::TypeProto* GetInputType(size_t index) const {
    return impl_->getInputType(index);
  }

  const ONNX_NAMESPACE::TypeProto* GetOutputType(size_t index) const {
    // Work around lack of a const method from the onnx InferenceContext interface
    return const_cast<Impl_t*>(impl_)->getOutputType(index);
  }
#endif

  // Try to query an attribute, returning nullptr if it doesn't exist
  const AttributeType* TryGetAttribute(const std::string& name) const {
    return impl_->getAttribute(name);
  }

  const AttributeType* GetAttribute(const std::string& name) const {
    const AttributeType* attr = TryGetAttribute(name);
    ORT_ENFORCE(attr != nullptr);
    return attr;
  }

 private:
  OpNodeProtoHelper() = delete;
  const Impl_t* impl_ = nullptr;
};

// The methods on the following class are called by OpNodeProtoHelper, implementing
// the same signatures as InferenceContext other than const-ness.
class ProtoHelperNodeContext {
 public:
#if !defined(__wasm__)
  explicit ProtoHelperNodeContext(const onnxruntime::Node& node) : node_(node) {}
#else
  explicit ProtoHelperNodeContext(const onnxruntime::NodeAttributes& attributes,
                                  size_t num_inputs,
                                  size_t num_outputs) : attributes_(attributes),
                                                        num_inputs_(num_inputs),
                                                        num_outputs_(num_outputs) {}
#endif
  ProtoHelperNodeContext() = delete;

  const AttributeType* getAttribute(const std::string& name) const;
  size_t getNumInputs() const;
#if !defined(__wasm__)
  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const;
#endif
  size_t getNumOutputs() const;
#if !defined(__wasm__)
  const ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) const;
#endif

 private:
#if !defined(__wasm__)
  const onnxruntime::Node& node_;
#else
  const onnxruntime::NodeAttributes& attributes_;
  size_t num_inputs_;
  size_t num_outputs_;
#endif
};

}  // namespace onnxruntime
