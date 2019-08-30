<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.layers.optical_flow.correlation_cost" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.layers.optical_flow.correlation_cost


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/layers/optical_flow.py#L29-L109">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Correlation Cost Volume computation.

``` python
tfa.layers.optical_flow.correlation_cost(
    input_a,
    input_b,
    kernel_size,
    max_displacement,
    stride_1,
    stride_2,
    pad,
    data_format='channels_last',
    name=None
)
```



<!-- Placeholder for "Used in" -->

"FlowNet: Learning Optical Flow with Convolutional Networks"
Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Hausser,
Caner Hazirbas, Vladimir Golkov, Patrick van der Smagt,
Daniel Cremers, Thomas Brox. https://arxiv.org/abs/1504.06852

Computes a cost volume using correlation for two inputs. For feature
maps A, B with spatial dimensions w, h, c it computes

  output(a, b) = sum_{l in [-k,k]**2}  < I(a+l), J(b+l) >

where the patches of size K=2d + 1 are centered in position a resp. b.

The output shape is [B, C', H', W'], where

  r = max_displacement / stride_2;
  bd = max_displacement + (kernel_size - 1) / 2
  C' = (2 * r + 1) ** 2
  H' = H + 2 * (pad - bd) / stride_1
  W' = W + 2 * (pad - bd) / stride_1

Note: When the data_format requests "channels_last", an additional explicit
  transpose operation is executed.

#### Args:


* <b>`input_a`</b>: A `Tensor` of the format specified by `data_format`.
* <b>`input_b`</b>: A `Tensor` of the format specified by `data_format`.
* <b>`kernel_size`</b>: An integer specifying the height and width of the
    patch used to compute the per-patch costs.
* <b>`max_displacement`</b>: An integer specifying the maximum search radius
    for each position.
* <b>`stride_1`</b>: An integer specifying the stride length in the input.
* <b>`stride_2`</b>: An integer specifying the stride length in the patch.
* <b>`pad`</b>: An integer specifying the paddings in height and width.
* <b>`data_format`</b>: Specifies the data format.
    Possible values are:
    "channels_last" float [batch, height, width, channels]
    "channels_first" float [batch, channels, height, width]
    Defaults to `"channels_last"`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of the format specified by `data_format`.
