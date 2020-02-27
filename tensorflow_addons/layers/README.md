# Addons - Layers

## Components
https://www.tensorflow.org/addons/api_docs/python/tfa/layers

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all layers
must:
 * Inherit from either `keras.layers.Layer` or its subclasses.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`
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
 * Update the [CODEOWNERS file](.github/CODEOWNERS)
