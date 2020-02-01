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
from types import ModuleType
import inspect

import tensorflow_addons

EXAMPLE_URL = "https://github.com/tensorflow/addons/blob/fa8e966d987fd9b0d20551a666e44e2790fdf9c7/tensorflow_addons/layers/normalizations.py#L73"
TUTORIAL_URL = "https://docs.python.org/3/library/typing.html"


# TODO: add types and remove all elements from
# the exception list.
EXCEPTION_LIST = [
    tensorflow_addons.image.mean_filter2d,
    tensorflow_addons.image.median_filter2d,
    tensorflow_addons.image.resampler,
    tensorflow_addons.image.sparse_image_warp,
    tensorflow_addons.image.rotate,
    tensorflow_addons.image.transform,
    tensorflow_addons.image.translate,
    tensorflow_addons.losses.contrastive_loss,
    tensorflow_addons.losses.sigmoid_focal_crossentropy,
    tensorflow_addons.losses.giou_loss,
    tensorflow_addons.losses.lifted_struct_loss,
    tensorflow_addons.losses.sparsemax_loss,
    tensorflow_addons.losses.triplet_semihard_loss,
    tensorflow_addons.losses.pinball_loss,
    tensorflow_addons.losses.PinballLoss,
    tensorflow_addons.losses.ContrastiveLoss,
    tensorflow_addons.losses.SigmoidFocalCrossEntropy,
    tensorflow_addons.losses.GIoULoss,
    tensorflow_addons.losses.LiftedStructLoss,
    tensorflow_addons.losses.SparsemaxLoss,
    tensorflow_addons.losses.TripletSemiHardLoss,
    tensorflow_addons.losses.npairs_loss,
    tensorflow_addons.losses.NpairsLoss,
    tensorflow_addons.losses.npairs_multilabel_loss,
    tensorflow_addons.losses.NpairsMultilabelLoss,
    tensorflow_addons.metrics.CohenKappa,
    tensorflow_addons.metrics.F1Score,
    tensorflow_addons.metrics.FBetaScore,
    tensorflow_addons.metrics.HammingLoss,
    tensorflow_addons.metrics.hamming_distance,
    tensorflow_addons.metrics.MeanMetricWrapper,
    tensorflow_addons.metrics.MatthewsCorrelationCoefficient,
    tensorflow_addons.metrics.MultiLabelConfusionMatrix,
    tensorflow_addons.metrics.RSquare,
    tensorflow_addons.rnn.LayerNormLSTMCell,
    tensorflow_addons.rnn.NASCell,
    tensorflow_addons.seq2seq.AttentionMechanism,
    tensorflow_addons.seq2seq.AttentionWrapper,
    tensorflow_addons.seq2seq.AttentionWrapperState,
    tensorflow_addons.seq2seq.BahdanauAttention,
    tensorflow_addons.seq2seq.BahdanauMonotonicAttention,
    tensorflow_addons.seq2seq.LuongAttention,
    tensorflow_addons.seq2seq.LuongMonotonicAttention,
    tensorflow_addons.seq2seq.hardmax,
    tensorflow_addons.seq2seq.monotonic_attention,
    tensorflow_addons.seq2seq.safe_cumprod,
    tensorflow_addons.seq2seq.BasicDecoder,
    tensorflow_addons.seq2seq.BasicDecoderOutput,
    tensorflow_addons.seq2seq.BeamSearchDecoder,
    tensorflow_addons.seq2seq.BeamSearchDecoderOutput,
    tensorflow_addons.seq2seq.BeamSearchDecoderState,
    tensorflow_addons.seq2seq.FinalBeamSearchDecoderOutput,
    tensorflow_addons.seq2seq.gather_tree,
    tensorflow_addons.seq2seq.gather_tree_from_array,
    tensorflow_addons.seq2seq.tile_batch,
    tensorflow_addons.seq2seq.BaseDecoder,
    tensorflow_addons.seq2seq.Decoder,
    tensorflow_addons.seq2seq.dynamic_decode,
    tensorflow_addons.seq2seq.SequenceLoss,
    tensorflow_addons.seq2seq.sequence_loss,
    tensorflow_addons.seq2seq.CustomSampler,
    tensorflow_addons.seq2seq.GreedyEmbeddingSampler,
    tensorflow_addons.seq2seq.InferenceSampler,
    tensorflow_addons.seq2seq.SampleEmbeddingSampler,
    tensorflow_addons.seq2seq.Sampler,
    tensorflow_addons.seq2seq.ScheduledEmbeddingTrainingSampler,
    tensorflow_addons.seq2seq.ScheduledOutputTrainingSampler,
    tensorflow_addons.seq2seq.TrainingSampler,
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


def check_public_api_has_typing_information():
    for attribute in get_attributes(tensorflow_addons):
        if isinstance(attribute, ModuleType):
            check_module_is_typed(attribute)


def check_module_is_typed(module):
    for attribute in get_attributes(module):
        if attribute in EXCEPTION_LIST:
            continue
        if inspect.isclass(attribute):
            check_function_is_typed(attribute.__init__, class_=attribute)
        if inspect.isfunction(attribute):
            check_function_is_typed(attribute)


def check_function_is_typed(func, class_=None):
    """ If class_ is not None, func is the __init__ of the class."""
    signature = inspect.signature(func)
    for parameter_name, parameter in signature.parameters.items():
        if parameter.annotation != inspect.Signature.empty:
            continue
        if parameter_name in ("args", "kwargs", "self"):
            continue
        if class_ is None:
            function_name = func.__name__
        else:
            function_name = class_.__name__ + ".__init__"
        raise NotTypedError(
            "The function '{}' has not complete type annotations "
            "in its signature (it's missing the type hint for '{}'). "
            "We would like all the functions and "
            "class constructors in the public API to be typed and have "
            "the @typechecked decorator. \n"
            "If you are not familiar with adding type hints in "
            "functions, you can look at functions already typed in"
            "the codebase. For example: {}. \n"
            "You can also look at this tutorial: "
            "{}.".format(function_name, parameter_name, EXAMPLE_URL, TUTORIAL_URL)
        )

    if class_ is None:
        if signature.return_annotation != inspect.Signature.empty:
            return
        raise NotTypedError(
            "The function {} has no return type. Please add one. "
            "You can take a look at the gelu activation function "
            "in tensorflow_addons/activations/gelu.py "
            "if you want an example.".format(func.__name__)
        )


def get_attributes(module):
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        yield attr


class NotTypedError(Exception):
    pass


if __name__ == "__main__":
    check_public_api_has_typing_information()
