# Addons - Image

## Maintainers
| Submodule  |  Maintainers  | Contact Info   |
|:---------- |:----------- |:--------------|
| dense_image_warp | @WindQAQ | windqaq@gmail.com |
| distance_transform_ops | @mels630 | mels630@gmail.com |
| distort_image_ops | @WindQAQ | windqaq@gmail.com |
| filters | @Mainak431 | mainakdutta76@gmail.com |
| transform_ops | @mels630 | mels630@gmail.com | 
| translate_ops | @sayoojbk	| sayoojbk@gmail.com |

## Components 
| Submodule  | Image Processing Function |  Reference  |
|:---------- |:----------- |:----------- |
| dense_image_warp | dense_image_warp |  |
| dense_image_warp | interpolate_bilinear |  |
| distance_transform_ops | euclidean_distance_transform | |
| distort_image_ops |  adjust_hsv_in_yiq |  |
| distort_image_ops | random_hsv_in_yiq |  |
| filters | mean_filter2d |  |
| filters | median_filter2d |  |
| transform_ops | angles_to_projective_transforms | | 
| transform_ops | compose_transforms | | 
| transform_ops | matrices_to_flat_transforms | | 
| transform_ops | rotate | | 
| transform_ops | transform |  | 
| translate_ops | translate | |
| translate_ops | translations_to_projective_transforms | |

Along with these operations, there are a few important image operations implemented along with some examples [here](https://colab.research.google.com/drive/1Qf3ixXAQ2PJOD75f_RptkaRZ3DOCkrm8). This includes implementation of following operations:
 - Histogram Equalization
 - Gaussian Blurring
 - Laplacian Filter
 - Laplacian of Gaussian Filter

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
 * Update the table of contents in this sub-packages's README.
