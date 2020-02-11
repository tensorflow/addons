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
"""Additional callbacks that conform to Keras API."""

from tensorflow_addons.callbacks.average_model_checkpoint import AverageModelCheckpoint
from tensorflow_addons.callbacks.time_stopping import TimeStopping
from tensorflow_addons.callbacks.tqdm_progress_bar import TQDMProgressBar
