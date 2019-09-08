# Addons - Activations

## Maintainers
| Submodule  | Maintainers  | Contact Info   |
|:---------- |:------------- |:--------------|
| sparsemax | @AndreasMadsen | amwwebdk+github@gmail.com |

## Contents
| Submodule | Activation  | Reference                               |
|:----------------------- |:-------------------|:---------------|
| sparsemax | Sparsemax | https://arxiv.org/abs/1602.02068           |


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all activations
must:
 * Be a `tf.function`.
 * Have the signature `fn(input, axis=-1, name=None)`.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/python/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the layer is behaving as expected.
 * When applicable, run all unittests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-package's README.
