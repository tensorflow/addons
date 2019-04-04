# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""A tool to generate api_docs for TensorFlow Addons.

```
Run from root folder of source code
python tools/docs/generate2.py --output_dir=docs/
```

Requires a local installation of:
  Docs Generator - pip install git+https://github.com/tensorflow/docs
  tensorflow_addons
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

from absl import app
from absl import flags
import tensorflow_addons as tfa

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import parser

from tensorflow.python.util import tf_inspect

# Use tensorflow's `tf_inspect`, which is aware of `tf_decorator`.
parser.tf_inspect = tf_inspect

# So patch `tfa.__all__` to list everything.
tfa.__all__ = [item_name for item_name, value in tf_inspect.getmembers(tfa)]

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "code_url_prefix", "/code/stable/tensorflow_addons",
    "A url to prepend to code paths when creating links to defining code")

flags.DEFINE_string("output_dir", "/tmp/out",
                    "A directory, where the docs will be output to.")

flags.DEFINE_bool("search_hints", True,
                  "Include meta-data search hints at the top of each file.")

flags.DEFINE_string(
    "site_path", "",
    "The prefix ({site-path}/api_docs/python/...) used in the "
    "`_toc.yaml` and `_redirects.yaml` files")


def build_docs(output_dir, code_url_prefix, search_hints=True):
    """Build api docs for tensorflow v2.

    Args:
      output_dir: A string path, where to put the files.
      code_url_prefix: prefix for "Defined in" links.
      search_hints: Bool. Include meta-data search hints at the top of file.
    """
    try:
        doc_controls.do_not_generate_docs(tfa.tools)
    except AttributeError:
        pass

    base_dir = path.dirname(tfa.__file__)
    base_dirs = (base_dir)

    code_url_prefixes = (
        code_url_prefix,
        # External packages source repositories
        "https://github.com/tensorflow/addons/tree/master/tensorflow_addons")

    doc_generator = generate_lib.DocGenerator(
        root_title="TensorFlow Addons 2.0 Preview",
        py_modules=[("tfa", tfa)],
        base_dir=base_dirs,
        search_hints=search_hints,
        code_url_prefix=code_url_prefixes,
        site_path=FLAGS.site_path)

    doc_generator.build(output_dir)


def main(argv):
    del argv
    build_docs(
        output_dir=FLAGS.output_dir,
        code_url_prefix=FLAGS.code_url_prefix,
        search_hints=FLAGS.search_hints)


if __name__ == "__main__":
    app.run(main)
