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

## Class `TrainingSampler`

A Sampler for use during training.

Inherits From: [`Sampler`](../../tfa/seq2seq/Sampler.md)

### Aliases:

* Class `tfa.seq2seq.TrainingSampler`
* Class `tfa.seq2seq.sampler.TrainingSampler`



Defined in [`seq2seq/sampler.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/seq2seq/sampler.py).

<!-- Placeholder for "Used in" -->

Only reads inputs.

Returned sample_ids are the argmax of the RNN output logits.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(time_major=False)
```

Initializer.

#### Args:

* <b>`time_major`</b>: Python bool.  Whether the tensors in `inputs` are time
    major. If `False` (default), they are assumed to be batch major.


#### Raises:

* <b>`ValueError`</b>: if `sequence_length` is not a 1D tensor.



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>



<h3 id="sample_ids_dtype"><code>sample_ids_dtype</code></h3>



<h3 id="sample_ids_shape"><code>sample_ids_shape</code></h3>





## Methods

<h3 id="initialize"><code>initialize</code></h3>

``` python
initialize(
    inputs,
    sequence_length=None
)
```

Initialize the TrainSampler.

#### Args:

* <b>`inputs`</b>: A (structure of) input tensors.
* <b>`sequence_length`</b>: An int32 vector tensor.


#### Returns:

(finished, next_inputs), a tuple of two items. The first item is a
  boolean vector to indicate whether the item in the batch has
  finished. The second item is the first slide of input data based on
  the timestep dimension (usually the second dim of the input).

<h3 id="next_inputs"><code>next_inputs</code></h3>

``` python
next_inputs(
    time,
    outputs,
    state,
    sample_ids
)
```



<h3 id="sample"><code>sample</code></h3>

``` python
sample(
    time,
    outputs,
    state
)
```





