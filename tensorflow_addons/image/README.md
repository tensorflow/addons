# Addons - Image

## Maintainers
| Submodule  |  Maintainers  | Contact Info   |
|:---------- |:----------- |:--------------|
| dense_image_warp |  |  |
| distance_transform_ops |  |  |
| distort_image_ops |  |  | 
| transform_ops |  |  | 

## Components 
| Submodule  | Image Processing Function |  Reference  |
|:---------- |:----------- |:----------- |
| dense_image_warp | dense_image_warp |  |
| dense_image_warp | interpolate_bilinear |  |
| distance_transform_ops | euclidean_distance_transform | |
| distort_image_ops |  adjust_hsv_in_yiq |  |
| distort_image_ops | random_hsv_in_yiq |  |
| transform_ops | angles_to_projective_transforms | | 
| transform_ops | matrices_to_flat_transforms | | 
| transform_ops | rotate | | 
| transform_ops | transform |  | 

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
