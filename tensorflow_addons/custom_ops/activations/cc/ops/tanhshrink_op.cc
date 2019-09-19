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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace addons {

REGISTER_OP("Addons>Tanhshrink")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Addons>TanhshrinkGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

}  // namespace addons
}  // namespace tensorflow