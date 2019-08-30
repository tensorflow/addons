<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.connected_components" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.connected_components


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/image/connected_components.py#L29-L96">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Labels the connected components in a batch of images.

``` python
tfa.image.connected_components(
    images,
    name=None
)
```



<!-- Placeholder for "Used in" -->

A component is a set of pixels in a single input image, which are
all adjacent     and all have the same non-zero value. The components
using a squared connectivity of one (all True entries are joined with
their neighbors above,below, left, and right). Components across all
images have consecutive ids 1 through n.
Components are labeled according to the first pixel of the
component appearing in row-major order (lexicographic order by
image_index_in_batch, row, col).
Zero entries all have an output id of 0.
This op is equivalent with `scipy.ndimage.measurements.label`
on a 2D array with the default structuring element
(which is the connectivity used here).
Args:
  images: A 2D (H, W) or 3D (N, H, W) Tensor of boolean image(s).
  name: The name of the op.
Returns:
  Components with the same shape as `images`.
  False entries in `images` have value 0, and
  all True entries map to a component id > 0.
Raises:
  TypeError: if `images` is not 2D or 3D.