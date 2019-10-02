# Addons - Text

## Maintainers
| Submodule  |  Maintainers  | Contact Info   |
|:---------- |:----------- |:------------- |
| skip_gram_ops |  |  |
| crf | Dheeraj R. Reddy | dheeraj98reddy@gmail.com |
| parse_time_op | Helin Wang | helin@google.com |

## Components 
| Submodule  | Text Processing Function |  Reference  |
|:---------- |:----------- |:----------- |
| skip_gram_ops |  skip_gram_sample | https://arxiv.org/abs/1301.3781 |
| skip_gram_ops |  skip_gram_sample_with_text_vocab | https://arxiv.org/abs/1301.3781 |
| parse_time_op | parse_time ||

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all text ops
must:
 * Be impossible to implement in one of the other API
 standards (Layers, Losses, etc.).
 * Be related to text processing.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the text op is behaving as
    expected.
 * When applicable, run all unittests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-packages's README.
