<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.TrainingSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="sample_ids_dtype"/>
<meta itemprop="property" content="sample_ids_shape"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next_inputs"/>
<meta itemprop="property" content="sample"/>
</div>

# tfa.seq2seq.TrainingSampler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L168-L289">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `TrainingSampler`

A Sampler for use during training.

Inherits From: [`Sampler`](../../tfa/seq2seq/Sampler.md)

**Aliases**: `tfa.seq2seq.sampler.TrainingSampler`

<!-- Placeholder for "Used in" -->

Only reads inputs.

Returned sample_ids are the argmax of the RNN output logits.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L176-L188">View source</a>

``` python
__init__(time_major=False)
```

Initializer.


#### Args:


* <b>`time_major`</b>: Python bool.  Whether the tensors in `inputs` are time
  major. If `False` (default), they are assumed to be batch major.


#### Raises:


* <b>`ValueError`</b>: if `sequence_length` is not a 1D tensor or `mask` is
  not a 2D boolean tensor.



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

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L205-L270">View source</a>

``` python
initialize(
    inputs,
    sequence_length=None,
    mask=None
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

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L277-L289">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L272-L275">View source</a>

``` python
sample(
    time,
    outputs,
    state
)
```

Returns `sample_ids`.






