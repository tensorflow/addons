# Addons - Activations

## Maintainers
| Submodule | Maintainers               | Contact Info                             |
|:----------|:--------------------------|:-----------------------------------------|
| gelu      | @AakashKumarNain @WindQAQ | aakashnain@outlook.com windqaq@gmail.com |
| hardshrink| @WindQAQ                  | windqaq@gmail.com                        |
| lisht     | @WindQAQ                  | windqaq@gmail.com                        |
| mish      | @digantamisra98 @WindQAQ  | mishradiganta91@gmail.com, windqaq@gmail.com |
| softshrink| @WindQAQ                  | windqaq@gmail.com                        |
| sparsemax | @AndreasMadsen            | amwwebdk+github@gmail.com                |
| tanhshrink| @fsx950223                | fsx950223@gmail.com                      |
| rrelu     | @fsx950223                | fsx950223@gmail.com                      |

## Contents
| Submodule | Activation | Reference                        |
|:----------|:-----------|:---------------------------------|
| gelu      | gelu       | https://arxiv.org/abs/1606.08415 |
| hardshrink| hardshrink |                                  |
| lisht     | lisht      | https://arxiv.org/abs/1901.05894 | 
| mish      | mish       | https://arxiv.org/abs/1908.08681 |
| softshrink| softshrink |                                  |
| sparsemax | sparsemax  | https://arxiv.org/abs/1602.02068 |
| tanhshrink| tanhshrink |                                  |
| rrelu     | rrelu      | https://arxiv.org/abs/1505.00853 |

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all activations
must:
 * Be a `tf.function`.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the layer is behaving as expected.
 * When applicable, run all unittests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.
 * Add activation name to [activations_test.py](https://github.com/tensorflow/addons/tree/master/tensorflow_addons/activations/activations_test.py) to test serialization.

#### Documentation Requirements
 * Update the table of contents in this sub-package's README.
