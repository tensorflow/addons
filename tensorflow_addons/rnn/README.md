# Addons - RNN

## Maintainers
| Submodule  | Maintainers  | Contact Info  |
|:---------- |:------------ |:------------- |
| cell       | Google       | @qlzh727      |

## Components
| Submodule | Class              | Reference |
|:----------|:------------------ |:--------- |
| cell      | NASCell            | https://arxiv.org/abs/1611.01578  |
| cell      | LayerNormLSTMCell  | https://arxiv.org/abs/1607.06450  |


## Contribution Guidelines
#### Prerequisites
 * For any cell based on research paper, the original paper has to be well recognized.
   The criteria here is >= 100 citation based on Google scholar. If the contributor feels
   this requirement need to be overruled, please specify the detailed justification in the
   PR.

#### Standard API
In order to conform with the current API standard, all cells must:
 * Inherit from either `keras.layers.AbstractRNNCell` or `keras.layers.Layer` with
   required properties.
 * [Register as a Keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * When applicable, run all tests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `@run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-packages's README.
