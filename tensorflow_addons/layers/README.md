# Addons - Layers

## Maintainers
| Submodule  |  Maintainers  | Contact Info   |
|:---------- |:----------- |:------------- |
| gelu       | @AakashKumarNain | aakashnain@outlook.com |  
| maxout | @fsx950223 | fsx950223@gmail.com |
| normalizations | @smokrow | moritz.kroeger@tu-dortmund.de |
| opticalflow | @fsx950223 | fsx950223@gmail.com |
| poincare | @rahulunair | rahulunair@gmail.com  |
| sparsemax | @AndreasMadsen | amwwebdk+github@gmail.com |
| wrappers | @seanpmorgan | seanmorgan@outlook.com |

## Components
| Submodule  | Layer |  Reference  |
|:---------- |:----------- |:------------- |
| gelu   | GeLU   | https://arxiv.org/abs/1606.08415 | 
| maxout | Maxout | https://arxiv.org/abs/1302.4389    |
| normalizations | GroupNormalization | https://arxiv.org/abs/1803.08494 |
| normalizations | InstanceNormalization | https://arxiv.org/abs/1607.08022 |
| opticalflow | CorrelationCost | https://arxiv.org/abs/1504.06852   |
| poincare | PoincareNormalize | https://arxiv.org/abs/1705.08039    |
| sparsemax| Sparsemax | https://arxiv.org/abs/1602.02068 |
| wrappers | WeightNormalization | https://arxiv.org/abs/1602.07868 |

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all layers
must:
 * Inherit from either `keras.layers.Layer` or its subclasses.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the layer is behaving as expected.
 * When applicable, run all unittests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Run `layer_test` on the layer.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-package's README.
