# Addons - Activations

## Contents
https://www.tensorflow.org/addons/api_docs/python/tfa/activations

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all activations
must:
 * Be a `tf.function` unless it is a straightforward call to a custom op or likely to be retraced.
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
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
