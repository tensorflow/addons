<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.ConditionalGradient" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="iterations"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_slot"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_gradients"/>
<meta itemprop="property" content="get_slot"/>
<meta itemprop="property" content="get_slot_names"/>
<meta itemprop="property" content="get_updates"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="variables"/>
</div>

# tfa.optimizers.ConditionalGradient


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/optimizers/conditional_gradient.py#L25-L139">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `ConditionalGradient`

Optimizer that implements the Conditional Gradient optimization.



### Aliases:

* Class `tfa.optimizers.conditional_gradient.ConditionalGradient`


<!-- Placeholder for "Used in" -->

This optimizer helps handle constraints well.

Currently only supports frobenius norm constraint.
See https://arxiv.org/pdf/1803.06453.pdf

```
variable -= (1-learning_rate) * (variable + lambda_ * gradient
    / (frobenius_norm(gradient) + epsilon))
```

Note that we choose "lambda_" here to refer to the constraint "lambda" in
the paper.
And 'epsilon' is constant with tiny value as compared to the value of
frobenius_norm of gradient. The purpose of 'epsilon' here is to avoid the
case that the value of frobenius_norm of gradient is 0.

In this implementation, we choose 'epsilon' with value of 10^-7.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/optimizers/conditional_gradient.py#L47-L71">View source</a>

``` python
__init__(
    learning_rate,
    lambda_,
    epsilon=1e-07,
    use_locking=False,
    name='ConditionalGradient',
    **kwargs
)
```

Construct a conditional gradient optimizer.


#### Args:


* <b>`learning_rate`</b>: A `Tensor` or a floating point value.
            The learning rate.
* <b>`lambda_`</b>: A `Tensor` or a floating point value. The constraint.
* <b>`epsilon`</b>: A `Tensor` or a floating point value. A small constant
        for numerical stability when handling the case of norm of
        gradient to be zero.
* <b>`use_locking`</b>: If `True` use locks for update operations.
* <b>`name`</b>: Optional name prefix for the operations created when
    applying gradients.  Defaults to 'ConditionalGradient'



## Properties

<h3 id="iterations"><code>iterations</code></h3>

Variable. The number of training steps this Optimizer has run.


<h3 id="weights"><code>weights</code></h3>

Returns variables of this Optimizer based on the order created.




## Methods

<h3 id="add_slot"><code>add_slot</code></h3>

``` python
add_slot(
    var,
    slot_name,
    initializer='zeros'
)
```

Add a new slot variable for `var`.


<h3 id="add_weight"><code>add_weight</code></h3>

``` python
add_weight(
    name,
    shape,
    dtype=None,
    initializer='zeros',
    trainable=None,
    synchronization=tf_variables.VariableSynchronization.AUTO,
    aggregation=tf_variables.VariableAggregation.NONE
)
```




<h3 id="apply_gradients"><code>apply_gradients</code></h3>

``` python
apply_gradients(
    grads_and_vars,
    name=None
)
```

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

#### Args:


* <b>`grads_and_vars`</b>: List of (gradient, variable) pairs.
* <b>`name`</b>: Optional name for the returned operation.  Default to the name
  passed to the `Optimizer` constructor.


#### Returns:

An `Operation` that applies the specified gradients. If `global_step`
was not None, that operation also increments `global_step`.



#### Raises:


* <b>`TypeError`</b>: If `grads_and_vars` is malformed.
* <b>`ValueError`</b>: If none of the variables have gradients.

<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config,
    custom_objects=None
)
```

Creates an optimizer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same optimizer from the config
dictionary.

#### Arguments:


* <b>`config`</b>: A Python dictionary, typically the output of get_config.
* <b>`custom_objects`</b>: A Python dictionary mapping names to additional Python
  objects used to create this optimizer, such as a function used for a
  hyperparameter.


#### Returns:

An optimizer instance.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/optimizers/conditional_gradient.py#L73-L81">View source</a>

``` python
get_config()
```




<h3 id="get_gradients"><code>get_gradients</code></h3>

``` python
get_gradients(
    loss,
    params
)
```

Returns gradients of `loss` with respect to `params`.


#### Arguments:


* <b>`loss`</b>: Loss tensor.
* <b>`params`</b>: List of variables.


#### Returns:

List of gradient tensors.



#### Raises:


* <b>`ValueError`</b>: In case any gradient cannot be computed (e.g. if gradient
  function not implemented).

<h3 id="get_slot"><code>get_slot</code></h3>

``` python
get_slot(
    var,
    slot_name
)
```




<h3 id="get_slot_names"><code>get_slot_names</code></h3>

``` python
get_slot_names()
```

A list of names for this optimizer's slots.


<h3 id="get_updates"><code>get_updates</code></h3>

``` python
get_updates(
    loss,
    params
)
```




<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```




<h3 id="minimize"><code>minimize</code></h3>

``` python
minimize(
    loss,
    var_list,
    grad_loss=None,
    name=None
)
```

Minimize `loss` by updating `var_list`.

This method simply computes gradient using `tf.GradientTape` and calls
`apply_gradients()`. If you want to process the gradient before applying
then call `tf.GradientTape` and `apply_gradients()` explicitly instead
of using this function.

#### Args:


* <b>`loss`</b>: A callable taking no arguments which returns the value to minimize.
* <b>`var_list`</b>: list or tuple of `Variable` objects to update to minimize
  `loss`, or a callable returning the list or tuple of `Variable` objects.
  Use callable when the variable list would otherwise be incomplete before
  `minimize` since the variables are created at the first time `loss` is
  called.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.
* <b>`name`</b>: Optional name for the returned operation.


#### Returns:

An Operation that updates the variables in `var_list`.  If `global_step`
was not `None`, that operation also increments `global_step`.



#### Raises:


* <b>`ValueError`</b>: If some of the variables are not `Variable` objects.

<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```




<h3 id="variables"><code>variables</code></h3>

``` python
variables()
```

Returns variables of this Optimizer based on the order created.




