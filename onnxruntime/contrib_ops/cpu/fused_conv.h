// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/conv.h"
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace contrib {

class FusedConvFloat final : public Conv<float> {
 public:
  FusedConvFloat(const OpKernelInfo& info) : Conv<float>(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }
};

}  // namespace contrib
}  // namespace onnxruntime
