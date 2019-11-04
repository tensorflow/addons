# Addons - Metrics

## Maintainers
| Submodule  | Maintainers  | Contact Info   |
|:---------- |:------------- |:--------------|
|  cohens_kappa| Aakash Nain   |  aakashnain@outlook.com|
|  f_scores| Saishruthi Swaminathan | saishruthi.tn@gmail.com|
|  r_square| Saishruthi Swaminathan| saishruthi.tn@gmail.com|
|  multilabel_confusion_matrix | Saishruthi Swaminathan | saishruthi.tn@gmail.com|
|  matthews_correlation_coefficient | I-Hong Jhuo | ihibmjhuo@gmail.com|

## Contents
| Submodule | Metric  | Reference                               |
|:----------------------- |:-------------------|:---------------|
| cohens_kappa| CohenKappa|[Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)|
| f1_scores| F1 micro, macro and weighted| [F1 Score](https://en.wikipedia.org/wiki/F1_score)|
| r_square| RSquare|[R-Sqaure](https://en.wikipedia.org/wiki/Coefficient_of_determination)|
| multilabel_confusion_matrix | Multilabel Confusion Matrix | [mcm](https://en.wikipedia.org/wiki/Confusion_matrix)|
| matthews_correlation_coefficient | Matthews Correlation Coefficient | [MCC](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)|

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all metrics
must:
 * Inherit from `tf.keras.metrics.Metric`.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/python/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the metric is behaving as expected.
 * When applicable, run all unittests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in this sub-package's README.
