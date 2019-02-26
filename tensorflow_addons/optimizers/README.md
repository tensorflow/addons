# Addons - Optimizers

## Contents
| Optimizer  | Reference                                   |
|:----------------------- |:-------------------------------|
| LazyAdamOptimizer | https://arxiv.org/abs/1412.6980      |


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all optimizers
must:
 * Inherit from either `keras.optimizer_v2.OptimizerV2` or its subclasses.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/python/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's README

#### Testing Requirements
 * When applicable, run all tests with TensorFlow's
 `@run_all_in_graph_and_eager_modes` decorator
 * Add a `py_test` to this sub-package's BUILD file

#### Documentation Requirements
 * Update the table of contents in the project's central README
 * Update the table of contents in this sub-project's README
  