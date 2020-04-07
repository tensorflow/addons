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
import glob
import os

from typedapi import ensure_api_is_typed

import tensorflow_addons as tfa

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def test_api_typed():
    modules_list = [
        tfa,
        tfa.activations,
        tfa.callbacks,
        tfa.image,
        tfa.losses,
        tfa.metrics,
        tfa.optimizers,
        tfa.rnn,
        tfa.seq2seq,
        tfa.text,
    ]
    # Files within this list will be exempt from verification.
    exception_list = []
    help_message = (
        "You can also take a look at the section about it in the CONTRIBUTING.md:\n"
        "https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md#about-type-hints"
    )
    ensure_api_is_typed(
        modules_list, exception_list, init_only=True, additional_message=help_message,
    )


def test_case_insensitive_filesystems():
    # Make sure BASE_DIR is project root.
    # If it doesn't, we probably computed the wrong directory.
    if not os.path.isdir(os.path.join(BASE_DIR, "tensorflow_addons")):
        raise AssertionError("BASE_DIR = {} is not project root".format(BASE_DIR))

    for dirpath, dirnames, filenames in os.walk(BASE_DIR, followlinks=True):
        lowercase_directories = [x.lower() for x in dirnames]
        lowercase_files = [x.lower() for x in filenames]

        lowercase_dir_contents = lowercase_directories + lowercase_files
        if len(lowercase_dir_contents) != len(set(lowercase_dir_contents)):
            raise AssertionError(
                "Files with same name but different case detected "
                "in directory: {}".format(dirpath)
            )


def get_lines_of_source_code(blacklist=None):
    blacklist = blacklist or []
    source_dir = os.path.join(BASE_DIR, "tensorflow_addons")
    for path in glob.glob(source_dir + "/**/*.py", recursive=True):
        if in_blacklist(path, blacklist):
            continue
        with open(path) as f:
            for line_idx, line in enumerate(f):
                yield path, line_idx, line


def in_blacklist(file_path, blacklist):
    for blacklisted_file in blacklist:
        if file_path.endswith(blacklisted_file):
            return True
    return False


def test_no_private_tf_api():
    # TODO: remove all elements of the list and remove the blacklist
    blacklist = [
        "tensorflow_addons/image/cutout_ops.py",
        "tensorflow_addons/optimizers/novograd.py",
        "tensorflow_addons/optimizers/moving_average.py",
        "tensorflow_addons/metrics/r_square.py",
        "tensorflow_addons/utils/test_utils.py",
        "tensorflow_addons/losses/giou_loss.py",
        "tensorflow_addons/seq2seq/decoder.py",
        "tensorflow_addons/seq2seq/attention_wrapper.py",
    ]

    for file_path, line_idx, line in get_lines_of_source_code(blacklist):

        if "import tensorflow.python" in line or "from tensorflow.python" in line:
            raise ImportError(
                "A private tensorflow API import was found in {} at line {}.\n"
                "tensorflow.python refers to TensorFlow's internal source "
                "code and private functions/classes.\n"
                "The use of those is forbidden in Addons for stability reasons."
                "\nYou should find a public alternative or ask the "
                "TensorFlow team to expose publicly the function/class "
                "that you are using.\n"
                "If you're trying to do `import tensorflow.python.keras` "
                "it can be replaced with `import tensorflow.keras`."
                "".format(file_path, line_idx + 1)
            )


def test_no_experimental_api():
    # TODO: remove all elements of the list and remove the blacklist
    blacklist = [
        "tensorflow_addons/optimizers/weight_decay_optimizers.py",
    ]
    for file_path, line_idx, line in get_lines_of_source_code(blacklist):

        if file_path.endswith("_test.py") or file_path.endswith("conftest.py"):
            continue
        if file_path.endswith("tensorflow_addons/utils/test_utils.py"):
            continue

        if "experimental" in line:
            raise NameError(
                "The usage of a TensorFlow experimental API was found in file {} "
                "at line {}:\n\n"
                "   {}\n"
                "Experimental APIs are ok in tests but not in user-facing code. "
                "This is because Experimental APIs might have bugs and are not "
                "widely used yet.\n"
                "Addons should show how to write TensorFlow "
                "code in a stable and forward-compatible way."
                "".format(file_path, line_idx, line)
            )


def test_no_deprecated_v1():
    # TODO: remove all elements of the list and remove the blacklist
    blacklist = [
        "tensorflow_addons/text/skip_gram_ops.py",
        "tensorflow_addons/text/skip_gram_ops_test.py",
        "tensorflow_addons/optimizers/lamb_test.py",
        "tensorflow_addons/optimizers/lookahead_test.py",
        "tensorflow_addons/optimizers/moving_average_test.py",
        "tensorflow_addons/optimizers/conditional_gradient_test.py",
        "tensorflow_addons/optimizers/yogi_test.py",
        "tensorflow_addons/optimizers/stochastic_weight_averaging_test.py",
        "tensorflow_addons/optimizers/novograd_test.py",
        "tensorflow_addons/optimizers/weight_decay_optimizers_test.py",
        "tensorflow_addons/optimizers/lazy_adam_test.py",
        "tensorflow_addons/metrics/matthews_correlation_coefficient_test.py",
        "tensorflow_addons/rnn/cell_test.py",
        "tensorflow_addons/seq2seq/decoder_test.py",
        "tensorflow_addons/metrics/cohens_kappa_test.py",
        "tensorflow_addons/optimizers/cyclical_learning_rate_test.py",
        "tensorflow_addons/metrics/f_test.py",
        "tensorflow_addons/seq2seq/basic_decoder_test.py",
        "tensorflow_addons/seq2seq/beam_search_decoder_test.py",
        "tensorflow_addons/seq2seq/decoder.py",
        "tensorflow_addons/optimizers/rectified_adam_test.py",
        "tensorflow_addons/metrics/multilabel_confusion_matrix_test.py",
        "tensorflow_addons/seq2seq/attention_wrapper_test.py",
    ]
    for file_path, line_idx, line in get_lines_of_source_code(blacklist):

        if "tf.compat.v1" in line:
            raise NameError(
                "The usage of a tf.compat.v1 API was found in file {} at line {}:\n\n"
                "   {}\n"
                "TensorFlow Addons doesn't support running programs with "
                "`tf.compat.v1.disable_v2_behavior()`.\n"
                "As such, there should be no need for the compatibility module "
                "tf.compat. Please find an alternative using only the TF2.x API."
                "".format(file_path, line_idx, line)
            )
