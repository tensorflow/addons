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
$> from the repo root run: python docs/build_docs.py
"""

from absl import app
from absl import flags

import tensorflow_addons as tfa

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

PROJECT_SHORT_NAME = "tfa"
PROJECT_FULL_NAME = "TensorFlow Addons"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "git_branch", default=None, help="The name of the corresponding branch on github."
)

CODE_PREFIX_TEMPLATE = (
    "https://github.com/tensorflow/addons/tree/{git_branch}/tensorflow_addons"
)
flags.DEFINE_string("code_url_prefix", None, "The url prefix for links to the code.")
flags.mark_flags_as_mutual_exclusive(["code_url_prefix", "git_branch"])

flags.DEFINE_string("output_dir", "/tmp/addons_api", "Where to output the docs")

flags.DEFINE_bool(
    "search_hints", True, "Include metadata search hints in the generated files"
)

flags.DEFINE_string(
    "site_path", "addons/api_docs/python", "Path prefix in the _toc.yaml"
)


def main(argv):
    if argv[1:]:
        raise ValueError("Unrecognized arguments: {}".format(argv[1:]))

    if FLAGS.code_url_prefix:
        code_url_prefix = FLAGS.code_url_prefix
    elif FLAGS.git_branch:
        code_url_prefix = CODE_PREFIX_TEMPLATE.format(git_branch=FLAGS.git_branch)
    else:
        code_url_prefix = CODE_PREFIX_TEMPLATE.format(git_branch="master")

    doc_generator = generate_lib.DocGenerator(
        root_title=PROJECT_FULL_NAME,
        py_modules=[(PROJECT_SHORT_NAME, tfa)],
        code_url_prefix=code_url_prefix,
        private_map={
            "tfa": ["__version__", "utils", "version"],
            "tfa.options": ["warn_fallback"],
        },
        # These callbacks usually clean up a lot of aliases caused by internal imports.
        callbacks=[
            public_api.local_definitions_filter,
            public_api.explicit_package_contents_filter,
        ],
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
    )

    doc_generator.build(FLAGS.output_dir)

    print("Output docs to: ", FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
