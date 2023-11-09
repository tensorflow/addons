# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TFDecorator-aware replacements for the inspect module."""
import collections
import functools
import inspect as _inspect

import tensorflow as tf

if hasattr(_inspect, "ArgSpec"):
    ArgSpec = _inspect.ArgSpec
else:
    ArgSpec = collections.namedtuple(
        "ArgSpec",
        [
            "args",
            "varargs",
            "keywords",
            "defaults",
        ],
    )

if hasattr(_inspect, "FullArgSpec"):
    FullArgSpec = _inspect.FullArgSpec
else:
    FullArgSpec = collections.namedtuple(
        "FullArgSpec",
        [
            "args",
            "varargs",
            "varkw",
            "defaults",
            "kwonlyargs",
            "kwonlydefaults",
            "annotations",
        ],
    )


def _convert_maybe_argspec_to_fullargspec(argspec):
    if isinstance(argspec, FullArgSpec):
        return argspec
    return FullArgSpec(
        args=argspec.args,
        varargs=argspec.varargs,
        varkw=argspec.keywords,
        defaults=argspec.defaults,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={},
    )


if hasattr(_inspect, "getfullargspec"):
    _getfullargspec = _inspect.getfullargspec

    def _getargspec(target):
        """A python3 version of getargspec.

        Calls `getfullargspec` and assigns args, varargs,
        varkw, and defaults to a python 2/3 compatible `ArgSpec`.

        The parameter name 'varkw' is changed to 'keywords' to fit the
        `ArgSpec` struct.

        Args:
          target: the target object to inspect.

        Returns:
          An ArgSpec with args, varargs, keywords, and defaults parameters
          from FullArgSpec.
        """
        fullargspecs = getfullargspec(target)
        argspecs = ArgSpec(
            args=fullargspecs.args,
            varargs=fullargspecs.varargs,
            keywords=fullargspecs.varkw,
            defaults=fullargspecs.defaults,
        )
        return argspecs

else:
    _getargspec = _inspect.getargspec

    def _getfullargspec(target):
        """A python2 version of getfullargspec.

        Args:
          target: the target object to inspect.

        Returns:
          A FullArgSpec with empty kwonlyargs, kwonlydefaults and annotations.
        """
        return _convert_maybe_argspec_to_fullargspec(getargspec(target))


def currentframe():
    """TFDecorator-aware replacement for inspect.currentframe."""
    return _inspect.stack()[1][0]


def getargspec(obj):
    """TFDecorator-aware replacement for `inspect.getargspec`.

    Note: `getfullargspec` is recommended as the python 2/3 compatible
    replacement for this function.

    Args:
      obj: A function, partial function, or callable object, possibly decorated.

    Returns:
      The `ArgSpec` that describes the signature of the outermost decorator that
      changes the callable's signature, or the `ArgSpec` that describes
      the object if not decorated.

    Raises:
      ValueError: When callable's signature can not be expressed with
        ArgSpec.
      TypeError: For objects of unsupported types.
    """
    if isinstance(obj, functools.partial):
        return _get_argspec_for_partial(obj)

    decorators, target = tf.__internal__.decorator.unwrap(obj)

    spec = next(
        (d.decorator_argspec for d in decorators if d.decorator_argspec is not None),
        None,
    )
    if spec:
        return spec

    try:
        # Python3 will handle most callables here (not partial).
        return _getargspec(target)
    except TypeError:
        pass

    if isinstance(target, type):
        try:
            return _getargspec(target.__init__)
        except TypeError:
            pass

        try:
            return _getargspec(target.__new__)
        except TypeError:
            pass

    # The `type(target)` ensures that if a class is received we don't return
    # the signature of its __call__ method.
    return _getargspec(type(target).__call__)


def _get_argspec_for_partial(obj):
    """Implements `getargspec` for `functools.partial` objects.

    Args:
      obj: The `functools.partial` object
    Returns:
      An `inspect.ArgSpec`
    Raises:
      ValueError: When callable's signature can not be expressed with
        ArgSpec.
    """
    # When callable is a functools.partial object, we construct its ArgSpec with
    # following strategy:
    # - If callable partial contains default value for positional arguments (ie.
    # object.args), then final ArgSpec doesn't contain those positional
    # arguments.
    # - If callable partial contains default value for keyword arguments (ie.
    # object.keywords), then we merge them with wrapped target. Default values
    # from callable partial takes precedence over those from wrapped target.
    #
    # However, there is a case where it is impossible to construct a valid
    # ArgSpec. Python requires arguments that have no default values must be
    # defined before those with default values. ArgSpec structure is only valid
    # when this presumption holds true because default values are expressed as a
    # tuple of values without keywords and they are always assumed to belong to
    # last K arguments where K is number of default values present.
    #
    # Since functools.partial can give default value to any argument, this
    # presumption may no longer hold in some cases. For example:
    #
    # def func(m, n):
    #   return 2 * m + n
    # partialed = functools.partial(func, m=1)
    #
    # This example will result in m having a default value but n doesn't. This
    # is usually not allowed in Python and can not be expressed in ArgSpec
    # correctly.
    #
    # Thus, we must detect cases like this by finding first argument with
    # default value and ensures all following arguments also have default
    # values. When this is not true, a ValueError is raised.

    n_prune_args = len(obj.args)
    partial_keywords = obj.keywords or {}

    args, varargs, keywords, defaults = getargspec(obj.func)

    # Pruning first n_prune_args arguments.
    args = args[n_prune_args:]

    # Partial function may give default value to any argument, therefore length
    # of default value list must be len(args) to allow each argument to
    # potentially be given a default value.
    no_default = object()
    all_defaults = [no_default] * len(args)

    if defaults:
        all_defaults[-len(defaults) :] = defaults

    # Fill in default values provided by partial function in all_defaults.
    for kw, default in partial_keywords.items():
        if kw in args:
            idx = args.index(kw)
            all_defaults[idx] = default
        elif not keywords:
            raise ValueError(
                "Function does not have **kwargs parameter, but "
                "contains an unknown partial keyword."
            )

    # Find first argument with default value set.
    first_default = next(
        (idx for idx, x in enumerate(all_defaults) if x is not no_default), None
    )

    # If no default values are found, return ArgSpec with defaults=None.
    if first_default is None:
        return ArgSpec(args, varargs, keywords, None)

    # Checks if all arguments have default value set after first one.
    invalid_default_values = [
        args[i]
        for i, j in enumerate(all_defaults)
        if j is no_default and i > first_default
    ]

    if invalid_default_values:
        raise ValueError(
            f"Some arguments {invalid_default_values} do not have "
            "default value, but they are positioned after those with "
            "default values. This can not be expressed with ArgSpec."
        )

    return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))


def getfullargspec(obj):
    """TFDecorator-aware replacement for `inspect.getfullargspec`.

    This wrapper emulates `inspect.getfullargspec` in[^)]* Python2.

    Args:
      obj: A callable, possibly decorated.

    Returns:
      The `FullArgSpec` that describes the signature of
      the outermost decorator that changes the callable's signature. If the
      callable is not decorated, `inspect.getfullargspec()` will be called
      directly on the callable.
    """
    decorators, target = tf.__internal__.decorator.unwrap(obj)

    for d in decorators:
        if d.decorator_argspec is not None:
            return _convert_maybe_argspec_to_fullargspec(d.decorator_argspec)
    return _getfullargspec(target)
