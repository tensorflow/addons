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
"""Implements Accuracy for Conditional Random Field."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
def crf_accuracy(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
    return crf.get_accuracy(y_true, y_pred)
