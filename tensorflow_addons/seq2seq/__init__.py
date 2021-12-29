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
"""Additional layers for sequence to sequence models."""

from tensorflow_addons.seq2seq.attention_wrapper import AttentionMechanism
from tensorflow_addons.seq2seq.attention_wrapper import AttentionWrapper
from tensorflow_addons.seq2seq.attention_wrapper import AttentionWrapperState
from tensorflow_addons.seq2seq.attention_wrapper import BahdanauAttention
from tensorflow_addons.seq2seq.attention_wrapper import BahdanauMonotonicAttention
from tensorflow_addons.seq2seq.attention_wrapper import LuongAttention
from tensorflow_addons.seq2seq.attention_wrapper import LuongMonotonicAttention
from tensorflow_addons.seq2seq.attention_wrapper import hardmax
from tensorflow_addons.seq2seq.attention_wrapper import monotonic_attention
from tensorflow_addons.seq2seq.attention_wrapper import safe_cumprod

from tensorflow_addons.seq2seq.basic_decoder import BasicDecoder
from tensorflow_addons.seq2seq.basic_decoder import BasicDecoderOutput

from tensorflow_addons.seq2seq.beam_search_decoder import BeamSearchDecoder
from tensorflow_addons.seq2seq.beam_search_decoder import BeamSearchDecoderOutput
from tensorflow_addons.seq2seq.beam_search_decoder import BeamSearchDecoderState
from tensorflow_addons.seq2seq.beam_search_decoder import FinalBeamSearchDecoderOutput
from tensorflow_addons.seq2seq.beam_search_decoder import gather_tree
from tensorflow_addons.seq2seq.beam_search_decoder import gather_tree_from_array
from tensorflow_addons.seq2seq.beam_search_decoder import tile_batch

from tensorflow_addons.seq2seq.decoder import BaseDecoder
from tensorflow_addons.seq2seq.decoder import Decoder
from tensorflow_addons.seq2seq.decoder import dynamic_decode

from tensorflow_addons.seq2seq.loss import SequenceLoss
from tensorflow_addons.seq2seq.loss import sequence_loss

from tensorflow_addons.seq2seq.sampler import CustomSampler
from tensorflow_addons.seq2seq.sampler import GreedyEmbeddingSampler
from tensorflow_addons.seq2seq.sampler import InferenceSampler
from tensorflow_addons.seq2seq.sampler import SampleEmbeddingSampler
from tensorflow_addons.seq2seq.sampler import Sampler
from tensorflow_addons.seq2seq.sampler import ScheduledEmbeddingTrainingSampler
from tensorflow_addons.seq2seq.sampler import ScheduledOutputTrainingSampler
from tensorflow_addons.seq2seq.sampler import TrainingSampler
