# Addons - Metrics

## Contents
https://www.tensorflow.org/addons/api_docs/python/tfa/metrics

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all metrics
must:
 * Inherit from `tf.keras.metrics.Metric`.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the metric is behaving as expected.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.experimental_run_functions_eagerly(True)`.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
