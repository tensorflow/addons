# Addons - Callbacks

## Maintainers
| Submodule  | Maintainers  | Contact Info   |
|:---------- |:------------- |:--------------|
| average_model_checkpoint | @squadrick | dheeraj98reddy@gmail.com |
|  time_stopping | @shun-lin | shunlin@google.com |
|  tqdm_progress_bar | @shun-lin | shunlin@google.com |

## Contents
| Submodule | Callback  | Reference                               |
|:----------------------- |:-------------------|:---------------|
| average_model_checkpoint | AverageModelCheckpoint | N/A |
| time_stopping | TimeStopping | N/A |
| tqdm_progress_bar | TQDMProgressBar | https://tqdm.github.io/ |

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all callbacks
must:
 * Inherit from `tf.keras.callbacks.Callback`.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the callback is behaving as expected.
 * When applicable, run all unittests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-package's README.
