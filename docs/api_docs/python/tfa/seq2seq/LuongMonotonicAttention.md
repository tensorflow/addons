<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.LuongMonotonicAttention" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="alignments_size"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="dynamic"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="memory_initialized"/>
<meta itemprop="property" content="metrics"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="state_size"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="deserialize_inner_layer_from_config"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_input_at"/>
<meta itemprop="property" content="get_input_mask_at"/>
<meta itemprop="property" content="get_input_shape_at"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_output_at"/>
<meta itemprop="property" content="get_output_mask_at"/>
<meta itemprop="property" content="get_output_shape_at"/>
<meta itemprop="property" content="get_updates_for"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="initial_alignments"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="setup_memory"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfa.seq2seq.LuongMonotonicAttention


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L1178-L1310">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `LuongMonotonicAttention`

Monotonic attention mechanism with Luong-style energy function.



### Aliases:

* Class `tfa.seq2seq.attention_wrapper.LuongMonotonicAttention`


<!-- Placeholder for "Used in" -->

This type of attention enforces a monotonic constraint on the attention
distributions; that is once the model attends to a given point in the
memory it can't attend to any prior points at subsequence output timesteps.
It achieves this by using the _monotonic_probability_fn instead of softmax
to construct its attention distributions.  Otherwise, it is equivalent to
LuongAttention.  This approach is proposed in

[Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
"Online and Linear-Time Attention by Enforcing Monotonic Alignments."
ICML 2017.](https://arxiv.org/abs/1704.00784)

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L1193-L1258">View source</a>

``` python
__init__(
    units,
    memory=None,
    memory_sequence_length=None,
    scale=False,
    sigmoid_noise=0.0,
    sigmoid_noise_seed=None,
    score_bias_init=0.0,
    mode='parallel',
    dtype=None,
    name='LuongMonotonicAttention',
    **kwargs
)
```

Construct the Attention mechanism.


#### Args:


* <b>`units`</b>: The depth of the query mechanism.
* <b>`memory`</b>: The memory to query; usually the output of an RNN encoder.
  This tensor should be shaped `[batch_size, max_time, ...]`.
* <b>`memory_sequence_length`</b>: (optional): Sequence lengths for the batch
  entries in memory.  If provided, the memory tensor rows are masked
  with zeros for values past the respective sequence lengths.
* <b>`scale`</b>: Python boolean.  Whether to scale the energy term.
* <b>`sigmoid_noise`</b>: Standard deviation of pre-sigmoid noise.  See the
  docstring for `_monotonic_probability_fn` for more information.
* <b>`sigmoid_noise_seed`</b>: (optional) Random seed for pre-sigmoid noise.
* <b>`score_bias_init`</b>: Initial value for score bias scalar.  It's
  recommended to initialize this to a negative value when the length
  of the memory is large.
* <b>`mode`</b>: How to compute the attention distribution.  Must be one of
  'recursive', 'parallel', or 'hard'.  See the docstring for
  <a href="../../tfa/seq2seq/monotonic_attention.md"><code>tfa.seq2seq.monotonic_attention</code></a> for more information.
* <b>`dtype`</b>: The data type for the query and memory layers of the attention
  mechanism.
* <b>`name`</b>: Name to use when creating ops.
* <b>`**kwargs`</b>: Dictionary that contains other common arguments for layer
  creation.



## Properties

<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.


<h3 id="alignments_size"><code>alignments_size</code></h3>




<h3 id="dtype"><code>dtype</code></h3>




<h3 id="dynamic"><code>dynamic</code></h3>




<h3 id="input"><code>input</code></h3>

Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.

#### Returns:

Input tensor or list of input tensors.



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.
* <b>`AttributeError`</b>: If no inbound nodes are found.

<h3 id="input_mask"><code>input_mask</code></h3>

Retrieves the input mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Input mask tensor (potentially None) or list of input
mask tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to
more than one incoming layers.

<h3 id="input_shape"><code>input_shape</code></h3>

Retrieves the input shape(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer, or if all inputs
have the same shape.

#### Returns:

Input shape, as an integer shape tuple
(or list of shape tuples, one tuple per input tensor).



#### Raises:


* <b>`AttributeError`</b>: if the layer has no defined input_shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="input_spec"><code>input_spec</code></h3>




<h3 id="losses"><code>losses</code></h3>

Losses which are associated with this `Layer`.

Variable regularization tensors are created when this property is accessed,
so it is eager safe: accessing `losses` under a `tf.GradientTape` will
propagate gradients back to the corresponding variables.

#### Returns:

A list of tensors.


<h3 id="memory_initialized"><code>memory_initialized</code></h3>

Returns `True` if this attention mechanism has been initialized with
a memory.

<h3 id="metrics"><code>metrics</code></h3>




<h3 id="name"><code>name</code></h3>




<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>




<h3 id="non_trainable_weights"><code>non_trainable_weights</code></h3>




<h3 id="output"><code>output</code></h3>

Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.

#### Returns:

Output tensor or list of output tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to more than one incoming
  layers.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="output_mask"><code>output_mask</code></h3>

Retrieves the output mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Output mask tensor (potentially None) or list of output
mask tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to
more than one incoming layers.

<h3 id="output_shape"><code>output_shape</code></h3>

Retrieves the output shape(s) of a layer.

Only applicable if the layer has one output,
or if all outputs have the same shape.

#### Returns:

Output shape, as an integer shape tuple
(or list of shape tuples, one tuple per output tensor).



#### Raises:


* <b>`AttributeError`</b>: if the layer has no defined output shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="state_size"><code>state_size</code></h3>




<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
a = tf.Module()
b = tf.Module()
c = tf.Module()
a.b = b
b.c = c
assert list(a.submodules) == [b, c]
assert list(b.submodules) == [c]
assert list(c.submodules) == []
```

#### Returns:

A sequence of all submodules.


<h3 id="trainable"><code>trainable</code></h3>




<h3 id="trainable_variables"><code>trainable_variables</code></h3>




<h3 id="trainable_weights"><code>trainable_weights</code></h3>




<h3 id="updates"><code>updates</code></h3>




<h3 id="variables"><code>variables</code></h3>

Returns the list of all layer variables/weights.

Alias of `self.weights`.

#### Returns:

A list of variables.


<h3 id="weights"><code>weights</code></h3>

Returns the list of all layer variables/weights.


#### Returns:

A list of variables.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L162-L196">View source</a>

``` python
__call__(
    inputs,
    **kwargs
)
```

Preprocess the inputs before calling `base_layer.__call__()`.

Note that there are situation here, one for setup memory, and one with
actual query and state.
1. When the memory has not been configured, we just pass all the param
   to base_layer.__call__(), which will then invoke self.call() with
   proper inputs, which allows this class to setup memory.
2. When the memory has already been setup, the input should contain
   query and state, and optionally processed memory. If the processed
   memory is not included in the input, we will have to append it to
   the inputs and give it to the base_layer.__call__(). The processed
   memory is the output of first invocation of self.__call__(). If we
   don't add it here, then from keras perspective, the graph is
   disconnected since the output from previous call is never used.

#### Args:


* <b>`inputs`</b>: the inputs tensors.
* <b>`**kwargs`</b>: dict, other keyeword arguments for the `__call__()`

<h3 id="build"><code>build</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L1260-L1270">View source</a>

``` python
build(input_shape)
```




<h3 id="compute_mask"><code>compute_mask</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L321-L325">View source</a>

``` python
compute_mask(
    inputs,
    mask=None
)
```




<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

``` python
compute_output_shape(input_shape)
```

Computes the output shape of the layer.

If the layer has not been built, this method will call `build` on the
layer. This assumes that the layer will later be used with inputs that
match the input shape provided here.

#### Arguments:


* <b>`input_shape`</b>: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.


#### Returns:

An input shape tuple.


<h3 id="count_params"><code>count_params</code></h3>

``` python
count_params()
```

Count the total number of scalars composing the weights.


#### Returns:

An integer count.



#### Raises:


* <b>`ValueError`</b>: if the layer isn't yet built
  (in which case its weights aren't yet defined).

<h3 id="deserialize_inner_layer_from_config"><code>deserialize_inner_layer_from_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L359-L389">View source</a>

``` python
deserialize_inner_layer_from_config(
    cls,
    config,
    custom_objects
)
```

Helper method that reconstruct the query and memory from the config.

In the get_config() method, the query and memory layer configs are
serialized into dict for persistence, this method perform the reverse
action to reconstruct the layer from the config.

#### Args:


* <b>`config`</b>: dict, the configs that will be used to reconstruct the
  object.
* <b>`custom_objects`</b>: dict mapping class names (or function names) of
  custom (non-Keras) objects to class/functions.

#### Returns:


* <b>`config`</b>: dict, the config with layer instance created, which is ready
  to be used as init parameters.

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L1306-L1310">View source</a>

``` python
@classmethod
from_config(
    cls,
    config,
    custom_objects=None
)
```




<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L1294-L1304">View source</a>

``` python
get_config()
```




<h3 id="get_input_at"><code>get_input_at</code></h3>

``` python
get_input_at(node_index)
```

Retrieves the input tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple inputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_input_mask_at"><code>get_input_mask_at</code></h3>

``` python
get_input_mask_at(node_index)
```

Retrieves the input mask tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple inputs).


<h3 id="get_input_shape_at"><code>get_input_shape_at</code></h3>

``` python
get_input_shape_at(node_index)
```

Retrieves the input shape(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple inputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_losses_for"><code>get_losses_for</code></h3>

``` python
get_losses_for(inputs)
```

Retrieves losses relevant to a specific set of inputs.


#### Arguments:


* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of loss tensors of the layer that depend on `inputs`.


<h3 id="get_output_at"><code>get_output_at</code></h3>

``` python
get_output_at(node_index)
```

Retrieves the output tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple outputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_output_mask_at"><code>get_output_mask_at</code></h3>

``` python
get_output_mask_at(node_index)
```

Retrieves the output mask tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple outputs).


<h3 id="get_output_shape_at"><code>get_output_shape_at</code></h3>

``` python
get_output_shape_at(node_index)
```

Retrieves the output shape(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple outputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_updates_for"><code>get_updates_for</code></h3>

``` python
get_updates_for(inputs)
```

Retrieves updates relevant to a specific set of inputs.


#### Arguments:


* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of update ops of the layer that depend on `inputs`.


<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```

Returns the current weights of the layer.


#### Returns:

Weights values as a list of numpy arrays.


<h3 id="initial_alignments"><code>initial_alignments</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L985-L1001">View source</a>

``` python
initial_alignments(
    batch_size,
    dtype
)
```

Creates the initial alignment values for the monotonic attentions.

Initializes to dirac distributions, i.e.
[1, 0, 0, ...memory length..., 0] for all entries in the batch.

#### Args:


* <b>`batch_size`</b>: `int32` scalar, the batch_size.
* <b>`dtype`</b>: The `dtype`.


#### Returns:

A `dtype` tensor shaped `[batch_size, alignments_size]`
(`alignments_size` is the values' `max_time`).


<h3 id="initial_state"><code>initial_state</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L419-L437">View source</a>

``` python
initial_state(
    batch_size,
    dtype
)
```

Creates the initial state values for the `AttentionWrapper` class.

This is important for AttentionMechanisms that use the previous
alignment to calculate the alignment at the next time step
(e.g. monotonic attention).

The default behavior is to return the same output as
initial_alignments.

#### Args:


* <b>`batch_size`</b>: `int32` scalar, the batch_size.
* <b>`dtype`</b>: The `dtype`.


#### Returns:

A structure of all-zero tensors with shapes as described by
`state_size`.


<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```

Sets the weights of the layer, from Numpy arrays.


#### Arguments:


* <b>`weights`</b>: a list of Numpy arrays. The number
    of arrays and their shape must match
    number of the dimensions of the weights
    of the layer (i.e. it should match the
    output of `get_weights`).


#### Raises:


* <b>`ValueError`</b>: If the provided weights list does not match the
    layer's specifications.

<h3 id="setup_memory"><code>setup_memory</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/attention_wrapper.py#L262-L315">View source</a>

``` python
setup_memory(
    memory,
    memory_sequence_length=None,
    memory_mask=None
)
```

Pre-process the memory before actually query the memory.

This should only be called once at the first invocation of call().

#### Args:


* <b>`memory`</b>: The memory to query; usually the output of an RNN encoder.
  This tensor should be shaped `[batch_size, max_time, ...]`.
memory_sequence_length (optional): Sequence lengths for the batch
  entries in memory. If provided, the memory tensor rows are masked
  with zeros for values past the respective sequence lengths.
* <b>`memory_mask`</b>: (Optional) The boolean tensor with shape `[batch_size,
  max_time]`. For any value equal to False, the corresponding value
  in memory should be ignored.

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
class MyModule(tf.Module):
  @tf.Module.with_name_scope
  def __call__(self, x):
    if not hasattr(self, 'w'):
      self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
    return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
mod = MyModule()
mod(tf.ones([8, 32]))
# ==> <tf.Tensor: ...>
mod.w
# ==> <tf.Variable ...'my_module/w:0'>
```

#### Args:


* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.




