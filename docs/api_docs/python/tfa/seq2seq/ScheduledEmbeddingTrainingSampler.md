<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.ScheduledEmbeddingTrainingSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="sample_ids_dtype"/>
<meta itemprop="property" content="sample_ids_shape"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next_inputs"/>
<meta itemprop="property" content="sample"/>
</div>

# tfa.seq2seq.ScheduledEmbeddingTrainingSampler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L292-L389">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `ScheduledEmbeddingTrainingSampler`

A training sampler that adds scheduled sampling.

Inherits From: [`TrainingSampler`](../../tfa/seq2seq/TrainingSampler.md)

**Aliases**: `tfa.seq2seq.sampler.ScheduledEmbeddingTrainingSampler`

<!-- Placeholder for "Used in" -->

Returns -1s for sample_ids where no sampling took place; valid
sample id values elsewhere.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L299-L335">View source</a>

``` python
__init__(
    sampling_probability,
    embedding_fn=None,
    time_major=False,
    seed=None,
    scheduling_seed=None
)
```

Initializer.


#### Args:


* <b>`sampling_probability`</b>: A `float32` 0-D or 1-D tensor: the probability
  of sampling categorically from the output ids instead of reading
  directly from the inputs.
* <b>`embedding_fn`</b>: A callable that takes a vector tensor of `ids`
  (argmax ids), or the `params` argument for `embedding_lookup`.
* <b>`time_major`</b>: Python bool. Whether the tensors in `inputs` are time
  major. If `False` (default), they are assumed to be batch major.
* <b>`seed`</b>: The sampling seed.
* <b>`scheduling_seed`</b>: The schedule decision rule sampling seed.


#### Raises:


* <b>`ValueError`</b>: if `sampling_probability` is not a scalar or vector.



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

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L337-L350">View source</a>

``` python
initialize(
    inputs,
    sequence_length=None,
    mask=None,
    embedding=None
)
```

Initialize the TrainSampler.


#### Args:


* <b>`inputs`</b>: A (structure of) input tensors.
* <b>`sequence_length`</b>: An int32 vector tensor.
* <b>`mask`</b>: A boolean 2D tensor.


#### Returns:

(finished, next_inputs), a tuple of two items. The first item is a
  boolean vector to indicate whether the item in the batch has
  finished. The second item is the first slide of input data based on
  the timestep dimension (usually the second dim of the input).


<h3 id="next_inputs"><code>next_inputs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L364-L389">View source</a>

``` python
next_inputs(
    time,
    outputs,
    state,
    sample_ids
)
```

Returns `(finished, next_inputs, next_state)`.


<h3 id="sample"><code>sample</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L352-L362">View source</a>

``` python
sample(
    time,
    outputs,
    state
)
```

Returns `sample_ids`.






