# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Modified from the tfdocs example api reference docs generation script.

This script generates API reference docs.

Install pre-requisites:
$> pip install -U git+https://github.com/tensorflow/docs
$> pip install artifacts/tensorflow_addons-*.whl

Generate Docs:
$> from the repo root run: python tools/docs/build_docs.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow_addons
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

PROJECT_SHORT_NAME = 'tf_addons'
PROJECT_FULL_NAME = 'TensorFlow Addons'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'git_branch',
    default='master',
    help='The name of the corresponding branch on github.')

flags.DEFINE_string(
    'output_dir',
    default='docs/api_docs/python/',
    help='Where to write the resulting docs to.')


def main(argv):
    if argv[1:]:
        raise ValueError('Unrecognized arguments: {}'.format(argv[1:]))

    code_url_prefix = ('https://github.com/tensorflow/addons/tree/'
                       '{git_branch}/tensorflow_addons'.format(
                           git_branch=FLAGS.git_branch))

    doc_generator = generate_lib.DocGenerator(
        root_title=PROJECT_FULL_NAME,
        # Replace `tensorflow_docs` with your module, here.
        py_modules=[(PROJECT_SHORT_NAME, tensorflow_addons)],
        code_url_prefix=code_url_prefix,
        # This callback cleans up a lot of aliases caused by internal imports.
        callbacks=[public_api.local_definitions_filter])

    doc_generator.build(FLAGS.output_dir)

    print('Output docs to: ', FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
