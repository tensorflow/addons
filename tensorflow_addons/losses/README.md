# Addons - Losses

## Contents
| Loss  | Reference                                              |
|:----------------------- |:-------------------------------------|
| TripletLoss | https://arxiv.org/abs/1503.03832 |


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all losses
must:
 * Inherit from `keras.losses.LossFunctionWrapper`.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/python/keras_utils.py)
  so it can be serialized properly.

#### Testing Requirements
 * Simple unittests that demonstrate the loss is behaving as expected on
 some set of known inputs and outputs.
 * When applicable, run all tests with TensorFlow's
 `@run_all_in_graph_and_eager_modes` decorator.