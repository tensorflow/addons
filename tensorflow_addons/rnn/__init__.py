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
"""Additional RNN cells that corform to Keras API."""

from tensorflow_addons.rnn.nas_cell import NASCell
from tensorflow_addons.rnn.layer_norm_lstm_cell import LayerNormLSTMCell
from tensorflow_addons.rnn.layer_norm_simple_rnn_cell import LayerNormSimpleRNNCell
from tensorflow_addons.rnn.esn_cell import ESNCell
from tensorflow_addons.rnn.peephole_lstm_cell import PeepholeLSTMCell
