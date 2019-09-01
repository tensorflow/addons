# Addons - Optimizers

## Maintainers
| Submodule  | Maintainers  | Contact Info   |
|:---------- |:------------- |:--------------|
| conditional_gradient | Pengyu Kan | pkan2@wisc.edu |
| lazy_adam | Saishruthi Swaminathan  | saishruthi.tn@gmail.com  |
| moving_average | Dheeraj R. Reddy | dheeraj98reddy@gmail.com |
| weight_decay_optimizers |  Phil Jund | ijund.phil@googlemail.com   |


## Components
| Submodule | Optimizer  | Reference                                   |
|:--------- |:---------- |:---------|
| conditional_gradient | ConditionalGradient | https://arxiv.org/pdf/1803.06453.pdf |
| lazy_adam | LazyAdam | https://arxiv.org/abs/1412.6980      |
| moving_average | MovingAverage | |
| weight_decay_optimizers | SGDW, AdamW, extend_with_decoupled_weight_decay | https://arxiv.org/pdf/1711.05101.pdf |


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all optimizers
must:
 * Inherit from either `keras.optimizer_v2.OptimizerV2` or its subclasses.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * When applicable, run all tests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-packages's README.
