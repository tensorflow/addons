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
"""Useful extra functionality for TensorFlow maintained by SIG-addons."""
from tensorflow_addons.utils.ensure_tf_install import _check_tf_version

_check_tf_version()

# Local project imports
from tensorflow_addons import activations
from tensorflow_addons import callbacks
from tensorflow_addons import image
from tensorflow_addons import layers
from tensorflow_addons import losses
from tensorflow_addons import metrics
from tensorflow_addons import optimizers
from tensorflow_addons import rnn
from tensorflow_addons import seq2seq
from tensorflow_addons import text
from tensorflow_addons.register import register_all

from tensorflow_addons.version import __version__
