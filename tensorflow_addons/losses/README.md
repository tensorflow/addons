# Addons - Losses

## Maintainers
| Submodule  |  Maintainers  | Contact Info   |
|:---------- |:----------- |:------------- |
| contrastive |  @WindQAQ | windqaq@gmail.com |
| focal_loss | @AakashKumarNain<br> @SSaishruthi  | aakashnain@outlook.com<br> saishruthi.tn@gmail.com |
| giou_loss | @fsx950223  | fsx950223@gmail.com |
| lifted | @rahulunair | rahulunair@gmail.com  |
| npairs | @WindQAQ | windqaq@gmail.com |
| sparsemax_loss | @AndreasMadsen | amwwebdk+github@gmail.com |
| triplet |  @rahulunair | rahulunair@gmail.com  |
| quantiles | @RomainBrault | mail@romainbrault.com |

## Components
| Submodule | Loss  | Reference               |
|:----------------------- |:---------------------|:--------------------------|
| contrastive | ContrastiveLoss | http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf |
| focal_loss | SigmoidFocalCrossEntropy | https://arxiv.org/abs/1708.02002  |
| giou_loss | GIoULoss | https://giou.stanford.edu/GIoU.pdf       |
| lifted | LiftedStructLoss | https://arxiv.org/abs/1511.06452       |
| npairs | NpairsLoss | http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf |
| npairs | NpairsMultilabelLoss | http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf |
| sparsemax_loss | SparsemaxLoss |  https://arxiv.org/abs/1602.02068 |
| triplet | TripletSemiHardLoss | https://arxiv.org/abs/1503.03832       |
| quantiles | Pinball | https://en.wikipedia.org/wiki/Quantile_regression |


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all losses
must:
 * Inherit from `keras.losses.Loss`.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the loss is behaving as expected on
 some set of known inputs and outputs.
 * When applicable, run all tests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass)
   decorator.
 * Add a `py_test` to this sub-package's BUILD file.

#### Documentation Requirements
 * Update the table of contents in the project's central README.
 * Update the table of contents in this sub-package's README.
