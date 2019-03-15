# Addons - Losses

## Maintainers

## Contents
| Loss  | Reference                                              |
|:----------------------- |:-------------------------------------|
| LiftedStructLoss | https://arxiv.org/abs/1511.06452       |
| SparsemaxLoss |  https://arxiv.org/abs/1602.02068 | 
| TripletSemiHardLoss | https://arxiv.org/abs/1503.03832       |


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all losses
must:
 * Inherit from `keras.losses.LossFunctionWrapper`.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/python/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the loss is behaving as expected on
 some set of known inputs and outputs.
 * When applicable, run all tests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-package's README.
