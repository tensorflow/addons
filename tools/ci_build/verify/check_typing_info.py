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
#

from typedapi import ensure_api_is_typed

import tensorflow_addons

TUTORIAL_URL = "https://docs.python.org/3/library/typing.html"
HELP_MESSAGE = (
    "You can also take a look at the section about it in the CONTRIBUTING.md:\n"
    "https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md#about-type-hints"
)


# TODO: add types and remove all elements from
# the exception list.
EXCEPTION_LIST = [
    tensorflow_addons.losses.lifted_struct_loss,
    tensorflow_addons.losses.triplet_semihard_loss,
    tensorflow_addons.losses.LiftedStructLoss,
    tensorflow_addons.losses.TripletSemiHardLoss,
    tensorflow_addons.losses.npairs_loss,
    tensorflow_addons.losses.NpairsLoss,
    tensorflow_addons.losses.npairs_multilabel_loss,
    tensorflow_addons.losses.NpairsMultilabelLoss,
    tensorflow_addons.text.crf_binary_score,
    tensorflow_addons.text.crf_decode,
    tensorflow_addons.text.crf_decode_backward,
    tensorflow_addons.text.crf_decode_forward,
    tensorflow_addons.text.crf_forward,
    tensorflow_addons.text.crf_log_likelihood,
    tensorflow_addons.text.crf_log_norm,
    tensorflow_addons.text.crf_multitag_sequence_score,
    tensorflow_addons.text.crf_sequence_score,
    tensorflow_addons.text.crf_unary_score,
    tensorflow_addons.text.viterbi_decode,
    tensorflow_addons.text.skip_gram_sample,
    tensorflow_addons.text.skip_gram_sample_with_text_vocab,
    tensorflow_addons.text.parse_time,
]


if __name__ == "__main__":
    ensure_api_is_typed(
        tensorflow_addons,
        EXCEPTION_LIST,
        init_only=True,
        additional_message=HELP_MESSAGE,
    )
