# Addons - Layers

## Components
https://www.tensorflow.org/addons/api_docs/python/tfa/layers

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all layers
must:
 * Inherit from either `keras.layers.Layer` or its subclasses.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`

#### Testing Requirements
 * Simple unittests that demonstrate the layer is behaving as expected.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.run_functions_eagerly(True)`.
 * Run `layer_test` on the layer.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
