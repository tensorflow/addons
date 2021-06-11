# Addons - Text

## Components 
https://www.tensorflow.org/addons/api_docs/python/tfa/text

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all text ops
must:
 * Be impossible to implement in one of the other API
 standards (Layers, Losses, etc.).
 * Be related to text processing.

#### Testing Requirements
 * Simple unittests that demonstrate the text op is behaving as
    expected.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.run_functions_eagerly(True)`.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
