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
"""Additional losses that conform to Keras API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_addons.losses.contrastive import contrastive_loss, ContrastiveLoss
from tensorflow_addons.losses.focal_loss import sigmoid_focal_crossentropy, SigmoidFocalCrossEntropy
from tensorflow_addons.losses.lifted import lifted_struct_loss, LiftedStructLoss
from tensorflow_addons.losses.npairs import npairs_loss, NpairsLoss
from tensorflow_addons.losses.sparsemax_loss import sparsemax_loss, SparsemaxLoss
from tensorflow_addons.losses.triplet import triplet_semihard_loss, TripletSemiHardLoss
