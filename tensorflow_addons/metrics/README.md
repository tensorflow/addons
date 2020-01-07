# Addons - Metrics

## Maintainers
| Submodule  | Maintainers  | Contact Info   |
|:---------- |:------------- |:--------------|
|  cohens_kappa| Aakash Nain   |  aakashnain@outlook.com|
|  f_scores| Saishruthi Swaminathan | saishruthi.tn@gmail.com|
|  Hamming | Saishruthi Swaminathan | saishruthi.tn@gmail.com|
|  r_square| Saishruthi Swaminathan| saishruthi.tn@gmail.com|
|  matthews_correlation_coefficient | I.H. Jhuo | ihibmjhuo@gmail.com|
|  multilabel_confusion_matrix | Saishruthi Swaminathan | saishruthi.tn@gmail.com|

## Contents
| Submodule | Metric  | Reference                               |
|:----------------------- |:-------------------|:---------------|
| cohens_kappa| CohenKappa|[Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)|
| f_scores| F1Score | [F1 Score](https://en.wikipedia.org/wiki/F1_score)|
| f_scores| FBetaScore | |
| hamming | HammingLoss and hamming_distance | [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)|
| matthews_correlation_coefficient | Matthews Correlation Coefficient | [MCC](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)|
| multilabel_confusion_matrix | Multilabel Confusion Matrix | [mcm](https://en.wikipedia.org/wiki/Confusion_matrix)|
| r_square| RSquare|[R-Square](https://en.wikipedia.org/wiki/Coefficient_of_determination)|

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all metrics
must:
 * Inherit from `tf.keras.metrics.Metric`.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`
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
