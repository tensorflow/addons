# Addons - Training

## Components 
https://www.tensorflow.org/addons/api_docs/python/tfa/training


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all training ops
must:
 * Preferrably be implemented as decorator that modifies the default TF training. 
 * Must be impossible to implement in one of the other API
 standards (Layers, Losses, etc.).
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the training op is behaving as
    expected.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.experimental_run_functions_eagerly(True)`.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)