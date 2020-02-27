# Addons - Image

## Components 
https://www.tensorflow.org/addons/api_docs/python/tfa/image


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all image ops
must:
 * Be a standard image processing technique 
 * Must be impossible to implement in one of the other API
 standards (Layers, Losses, etc.).
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the image op is behaving as
    expected.
 * When applicable, run all unittests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
