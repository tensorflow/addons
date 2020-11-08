# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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


import inspect


def discover_classes(module, parent, class_exceptions):
    """
    Args:
        module: a module in which to search for classes that inherit from the parent class
        parent: the parent class that identifies classes in the module that should be tested
        class_exceptions: a list of specific classes that should be excluded when discovering classes in a module

    Returns:
        a list of classes for testing using pytest for parameterized tests
    """

    classes = [
        class_info[1]
        for class_info in inspect.getmembers(module, inspect.isclass)
        if issubclass(class_info[1], parent) and not class_info[0] in class_exceptions
    ]

    return classes
