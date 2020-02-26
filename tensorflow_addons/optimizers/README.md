# Addons - Optimizers

## Maintainers
| Submodule  | Maintainers  | Contact Info   |
|:---------- |:------------- |:--------------|
| conditional_gradient | Pengyu Kan, Vishnu Lokhande | pkan2@wisc.edu, lokhande@cs.wisc.edu |
| cyclical_learning_rate | Raphael Meudec | raphael.meudec@gmail.com |
| lamb | Jing Li, Junjie Ke | jingli@google.com, junjiek@google.com |
| lazy_adam | Saishruthi Swaminathan  | saishruthi.tn@gmail.com  |
| lookahead | Zhao Hanguang | cyberzhg@gmail.com |
| moving_average | Dheeraj R. Reddy | dheeraj98reddy@gmail.com |
| novograd | Shreyash Patodia | patodiashreyash32@gmail.com |
| rectified_adam | Zhao Hanguang | cyberzhg@gmail.com |
| stochastic_weight_averaging | Shreyash Patodia | patodiashreyash32@gmail.com |
| weight_decay_optimizers |  Phil Jund | ijund.phil@googlemail.com   |
| yogi | Manzil Zaheer | manzilz@google.com |



## Components
| Submodule | Optimizer  | Reference                                   |
|:--------- |:---------- |:---------|
| conditional_gradient | ConditionalGradient | https://arxiv.org/pdf/1803.06453.pdf |
| cyclical_learning_rate | Cyclical Learning Rate | https://arxiv.org/abs/1506.01186 |
| lamb | LAMB | https://arxiv.org/abs/1904.00962      |
| lazy_adam | LazyAdam | https://arxiv.org/abs/1412.6980      |
| lookahead | Lookahead | https://arxiv.org/abs/1907.08610v1 |
| moving_average | MovingAverage | |
| novograd | NovoGrad | https://nvidia.github.io/OpenSeq2Seq/html/optimizers.html |
| rectified_adam | RectifiedAdam | https://arxiv.org/pdf/1908.03265v1.pdf |
| stochastic_weight_averaging | SWA | https://arxiv.org/abs/1803.05407.pdf |
| weight_decay_optimizers | SGDW, AdamW, extend_with_decoupled_weight_decay | https://arxiv.org/pdf/1711.05101.pdf |
| yogi | Yogi | https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf      |



## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all optimizers
must:
 * Inherit from either `keras.optimizer_v2.OptimizerV2` or its subclasses.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * When applicable, run all tests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-packages's README.
