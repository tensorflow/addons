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

EXAMPLE_URL = 'https://github.com/tensorflow/addons/blob/fa8e966d987fd9b0d20551a666e44e2790fdf9c7/tensorflow_addons/layers/normalizations.py#L73'
TUTORIAL_URL = 'https://docs.python.org/3/library/typing.html'


def check_public_api_has_typing_information():
    for attribute in get_attributes(tensorflow_addons):
        if isinstance(attribute, ModuleType):
            check_module_is_typed(attribute)


def check_module_is_typed(module):
    for attribute in get_attributes(module):
        if inspect.isclass(attribute):
            check_function_is_typed(attribute.__init__, class_=attribute)
        if inspect.isfunction(attribute):
            check_function_is_typed(attribute)


def check_function_is_typed(func, class_=None):
    """ If class_ is not None, func is the __init__ of the class."""
    list_args = inspect.getfullargspec(func).args
    if class_ is not None:
        list_args.pop(0) # we remove 'self'
    if len(list_args) != len(func.__annotations__):
        if class_ is None:
            function_name = func.__name__
        else:
            function_name = class_.__name__ + '.__init__'
        raise NotImplementedError(
            "The function {} has not complete type annotations "
            "in its signature. We would like all the functions and "
            "class constructors in the public API to be typed and have "
            "the typechecked decorator. \n"
            "If you are not familiar with adding type hints in "
            "functions, you can look at functions already typed in"
            "the codebase. For example: {}. \n"
            "You can also look at this tutorial: {}.".format(function_name,
                                                             EXAMPLE_URL,
                                                             TUTORIAL_URL)
        )


def get_attributes(module):
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        yield attr


if __name__ == '__main__':
    check_public_api_has_typing_information()