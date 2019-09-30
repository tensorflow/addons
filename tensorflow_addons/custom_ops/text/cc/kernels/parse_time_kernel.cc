/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>

#include "absl/time/time.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace addons {

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::tstring;

enum OutputUnit {
  SECOND = 1,
  MILLISECOND = 2,
  MICROSECOND = 3,
  NANOSECOND = 4,
};

bool OutputUnitFromString(string output_unit_str, OutputUnit* output_unit) {
  if (output_unit_str == "SECOND") {
    *output_unit = SECOND;
  } else if (output_unit_str == "MILLISECOND") {
    *output_unit = MILLISECOND;
  } else if (output_unit_str == "MICROSECOND") {
    *output_unit = MICROSECOND;
  } else if (output_unit_str == "NANOSECOND") {
    *output_unit = NANOSECOND;
  } else {
    return false;
  }
  return true;
}

class ParseTimeOp : public OpKernel {
 public:
  explicit ParseTimeOp(OpKernelConstruction* context) : OpKernel(context) {
    string output_unit_str;
    OP_REQUIRES_OK(context, context->GetAttr("time_format", &time_format_));
    OP_REQUIRES_OK(context, context->GetAttr("output_unit", &output_unit_str));
    OP_REQUIRES(context, OutputUnitFromString(output_unit_str, &output_unit_),
                errors::InvalidArgument("Invalid output unit"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    auto output_flat = output_tensor->flat<int64>();
    const int n = input.size();
    for (int i = 0; i < n; ++i) {
      absl::Time time;
      std::string err;
      OP_REQUIRES(context, absl::ParseTime(time_format_, input(i), &time, &err),
                  errors::InvalidArgument("Parse time failed: ", err));
      switch (output_unit_) {
        case SECOND:
          output_flat(i) = absl::ToUnixSeconds(time);
          break;
        case MILLISECOND:
          output_flat(i) = absl::ToUnixMillis(time);
          break;
        case MICROSECOND:
          output_flat(i) = absl::ToUnixMicros(time);
          break;
        case NANOSECOND:
          output_flat(i) = absl::ToUnixNanos(time);
          break;
      }
    }
  }

 private:
  std::string time_format_;
  OutputUnit output_unit_;
};

REGISTER_KERNEL_BUILDER(Name("Addons>ParseTime").Device(tensorflow::DEVICE_CPU),
                        ParseTimeOp);

}  // end namespace addons
}  // end namespace tensorflow
