<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.SampleEmbeddingSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="sample_ids_dtype"/>
<meta itemprop="property" content="sample_ids_shape"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next_inputs"/>
<meta itemprop="property" content="sample"/>
</div>

# tfa.seq2seq.SampleEmbeddingSampler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L620-L662">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `SampleEmbeddingSampler`

A sampler for use during inference.

Inherits From: [`GreedyEmbeddingSampler`](../../tfa/seq2seq/GreedyEmbeddingSampler.md)

**Aliases**: `tfa.seq2seq.sampler.SampleEmbeddingSampler`

<!-- Placeholder for "Used in" -->

Uses sampling (from a distribution) instead of argmax and passes the
result through an embedding layer to get the next input.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L627-L648">View source</a>

``` python
__init__(
    embedding_fn=None,
    softmax_temperature=None,
    seed=None
)
```

Initializer.


#### Args:


* <b>`embedding_fn`</b>: (Optional) A callable that takes a vector tensor of
  `ids` (argmax ids), or the `params` argument for
  `embedding_lookup`. The returned tensor will be passed to the
  decoder input.
* <b>`softmax_temperature`</b>: (Optional) `float32` scalar, value to divide the
  logits by before computing the softmax. Larger values (above 1.0)
  result in more random samples, while smaller values push the
  sampling distribution towards the argmax. Must be strictly greater
  than 0. Defaults to 1.0.
* <b>`seed`</b>: (Optional) The sampling seed.


#### Raises:


* <b>`ValueError`</b>: if `start_tokens` is not a 1D tensor or `end_token` is
  not a scalar.



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>

Batch size of tensor returned by `sample`.

Returns a scalar int32 tensor. The return value might not
available before the invocation of initialize(), in this case,
ValueError is raised.

<h3 id="sample_ids_dtype"><code>sample_ids_dtype</code></h3>

DType of tensor returned by `sample`.

Returns a DType. The return value might not available before the
invocation of initialize().

<h3 id="sample_ids_shape"><code>sample_ids_shape</code></h3>

Shape of tensor returned by `sample`, excluding the batch dimension.

Returns a `TensorShape`. The return value might not available
before the invocation of initialize().



## Methods

<h3 id="initialize"><code>initialize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L562-L595">View source</a>

``` python
initialize(
    embedding,
    start_tokens=None,
    end_token=None
)
```

Initialize the GreedyEmbeddingSampler.


#### Args:


* <b>`embedding`</b>: tensor that contains embedding states matrix. It will be
  used to generate generate outputs with start_tokens and end_tokens.
  The embedding will be ignored if the embedding_fn has been provided
  at __init__().
* <b>`start_tokens`</b>: `int32` vector shaped `[batch_size]`, the start tokens.
* <b>`end_token`</b>: `int32` scalar, the token that marks end of decoding.


#### Returns:

Tuple of two items: `(finished, self.start_inputs)`.


#### Raises:


* <b>`ValueError`</b>: if `start_tokens` is not a 1D tensor or `end_token` is
  not a scalar.

<h3 id="next_inputs"><code>next_inputs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L607-L617">View source</a>

``` python
next_inputs(
    time,
    outputs,
    state,
    sample_ids
)
```

next_inputs_fn for GreedyEmbeddingHelper.


<h3 id="sample"><code>sample</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L650-L662">View source</a>

``` python
sample(
    time,
    outputs,
    state
)
```

sample for SampleEmbeddingHelper.






