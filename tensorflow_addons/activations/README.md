# Addons - Activations

## Contents
https://www.tensorflow.org/addons/api_docs/python/tfa/activations

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all activations
must:
 * Be a `tf.function` unless it is a straightforward call to a custom op or likely to be retraced.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`

#### Testing Requirements
 * Simple unittests that demonstrate the layer is behaving as expected.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.run_functions_eagerly(True)`.
 * Add activation name to [activations_test.py](https://github.com/tensorflow/addons/tree/master/tensorflow_addons/activations/tests/activations_test.py) to test serialization.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
