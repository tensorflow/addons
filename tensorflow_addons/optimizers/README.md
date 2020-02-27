# Addons - Optimizers

## Components
https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers

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
