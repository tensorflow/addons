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

#### Testing Requirements
 * When applicable, run all tests with TensorFlow's
 `@run_all_in_graph_and_eager_modes` decorator