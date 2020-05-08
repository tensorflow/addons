# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import inspect

from tensorflow.keras.metrics import Metric
from tensorflow_addons import metrics


def test_update_state_signature():
    for name, obj in inspect.getmembers(metrics):
        if inspect.isclass(obj) and issubclass(obj, Metric):
            check_update_state_signature(obj)


def check_update_state_signature(metric_class):
    update_state_signature = inspect.signature(metric_class.update_state)
    for expected_parameter in ["y_true", "y_pred", "sample_weight"]:
        if expected_parameter not in update_state_signature.parameters.keys():
            raise ValueError(
                "Class {} is missing the parameter {} in the `update_state` "
                "method. If the method doesn't use this argument, declare "
                "it anyway and raise a UserWarning if it is "
                "not None.".format(metric_class.__name__, expected_parameter)
            )
