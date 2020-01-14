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
# Test that checks if we have any issues with case insensitive filesystems.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))


def main():
    # Make sure BASE_DIR is project root.
    # If it doesn't, we probably computed the wrong directory.
    if not os.path.isdir(os.path.join(BASE_DIR, 'tensorflow_addons')):
        raise AssertionError(
            'BASE_DIR = {} is not project root'.format(BASE_DIR))

    for dirpath, dirnames, filenames in os.walk(BASE_DIR, followlinks=True):
        lowercase_directories = [x.lower() for x in dirnames]
        lowercase_files = [x.lower() for x in filenames]

        lowercase_dir_contents = lowercase_directories + lowercase_files
        if len(lowercase_dir_contents) != len(set(lowercase_dir_contents)):
            raise AssertionError(
                'Files with same name but different case detected '
                'in directory: {}'.format(dirpath))


if __name__ == '__main__':
    main()
