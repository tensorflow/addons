# Addons - RNN

## Components
https://www.tensorflow.org/addons/api_docs/python/tfa/rnn

## Contribution Guidelines
#### Prerequisites
 * For any cell based on research paper, the original paper has to be well recognized.
   The criteria here is >= 100 citation based on Google scholar. If the contributor feels
   this requirement need to be overruled, please specify the detailed justification in the
   PR.

#### Standard API
In order to conform with the current API standard, all cells must:
 * Inherit from either `keras.layers.AbstractRNNCell` or `keras.layers.Layer` with
   required properties.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`

#### Testing Requirements
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.run_functions_eagerly(True)`.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
