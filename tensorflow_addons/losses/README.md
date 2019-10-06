# Addons - Losses

## Maintainers
| Submodule  |  Maintainers  | Contact Info   |
|:---------- |:----------- |:------------- |
| contrastive |  @WindQAQ | windqaq@gmail.com |
| focal_loss | @SSaishruthi  | saishruthi.tn@gmail.com |
| lifted |  |  |
| npairs | @WindQAQ | windqaq@gmail.com |
| sparsemax_loss | @AndreasMadsen | amwwebdk+github@gmail.com |
| triplet |   |  |

## Components
| Submodule | Loss  | Reference               |
|:----------------------- |:---------------------|:--------------------------|
| contrastive | ContrastiveLoss | http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf |
| focal_loss | SigmoidFocalCrossEntropy | https://arxiv.org/abs/1708.02002  |
| lifted | LiftedStructLoss | https://arxiv.org/abs/1511.06452       |
| npairs | NpairsLoss | http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf |
| npairs | NpairsMultilabelLoss | http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf |
| sparsemax_loss | SparsemaxLoss |  https://arxiv.org/abs/1602.02068 |
| triplet | TripletSemiHardLoss | https://arxiv.org/abs/1503.03832       |


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all losses
must:
 * Inherit from `keras.losses.Loss`.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py)
  so it can be serialized properly.
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
