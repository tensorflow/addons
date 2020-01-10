<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.metric_learning.pairwise_distance" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.metric_learning.pairwise_distance

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/losses/metric_learning.py#L23-L73">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes the pairwise distance matrix with numerical stability.

``` python
tfa.losses.metric_learning.pairwise_distance(
    feature,
    squared=False
)
```



<!-- Placeholder for "Used in" -->

output[i, j] = || feature[i, :] - feature[j, :] ||_2

#### Args:


* <b>`feature`</b>: 2-D Tensor of size [number of data, feature dimension].
* <b>`squared`</b>: Boolean, whether or not to square the pairwise distances.


#### Returns:


* <b>`pairwise_distances`</b>: 2-D Tensor of size [number of data, number of data].

