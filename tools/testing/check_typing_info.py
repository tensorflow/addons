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

import tensorflow_addons as tfa

TUTORIAL_URL = "https://docs.python.org/3/library/typing.html"
HELP_MESSAGE = (
    "You can also take a look at the section about it in the CONTRIBUTING.md:\n"
    "https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md#about-type-hints"
)

# Files within this list will be exempt from verification.
EXCEPTION_LIST = []


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


if __name__ == "__main__":
    ensure_api_is_typed(
        modules_list, EXCEPTION_LIST, init_only=True, additional_message=HELP_MESSAGE,
    )
