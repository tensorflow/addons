<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="minimize"/>
</div>

# tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension

## Class `DecoupledWeightDecayExtension`

This class allows to extend optimizers with decoupled weight decay.





Defined in [`optimizers/weight_decay_optimizers.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/optimizers/weight_decay_optimizers.py).

<!-- Placeholder for "Used in" -->

It implements the decoupled weight decay described by Loshchilov & Hutter
(https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
decoupled from the optimization steps w.r.t. to the loss function.
For SGD variants, this simplifies hyperparameter search since it decouples
the settings of weight decay and learning rate.
For adaptive gradient algorithms, it regularizes variables with large
gradients more than L2 regularization would, which was shown to yield
better training loss and generalization error in the paper above.

This class alone is not an optimizer but rather extends existing
optimizers with decoupled weight decay. We explicitly define the two
examples used in the above paper (SGDW and AdamW), but in general this
can extend any OptimizerX by using
`extend_with_decoupled_weight_decay(
    OptimizerX, weight_decay=weight_decay)`.
In order for it to work, it must be the first class the Optimizer with
weight decay inherits from, e.g.

```python
class AdamW(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
  def __init__(self, weight_decay, *args, **kwargs):
    super(AdamW, self).__init__(weight_decay, *args, **kwargs).
```

Note: this extension decays weights BEFORE applying the update based
on the gradient, i.e. this extension only has the desired behaviour for
optimizers which do not depend on the value of'var' in the update step!

Note: when applying a decay to the learning rate, be sure to manually apply
the decay to the `weight_decay` as well. For example:

```python
step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
# lr and wd can be a function or a tensor
lr = 1e-1 * schedule(step)
wd = lambda: 1e-4 * schedule(step)

# ...

optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    weight_decay,
    **kwargs
)
```

Extension class that adds weight decay to an optimizer.

#### Args:

* <b>`weight_decay`</b>: A `Tensor` or a floating point value, the factor by
        which a variable is decayed in the update step.
* <b>`**kwargs`</b>: Optional list or tuple or set of `Variable` objects to
        decay.



## Methods

<h3 id="apply_gradients"><code>apply_gradients</code></h3>

``` python
apply_gradients(
    grads_and_vars,
    name=None,
    decay_var_list=None
)
```

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

#### Args:

* <b>`grads_and_vars`</b>: List of (gradient, variable) pairs.
* <b>`name`</b>: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.
* <b>`decay_var_list`</b>: Optional list of variables to be decayed. Defaults
        to all variables in var_list.

#### Returns:

An `Operation` that applies the specified gradients. If
`global_step` was not None, that operation also increments
`global_step`.

#### Raises:

* <b>`TypeError`</b>: If `grads_and_vars` is malformed.
* <b>`ValueError`</b>: If none of the variables have gradients.

<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```



<h3 id="minimize"><code>minimize</code></h3>

``` python
minimize(
    loss,
    var_list,
    grad_loss=None,
    name=None,
    decay_var_list=None
)
```

Minimize `loss` by updating `var_list`.

This method simply computes gradient using `tf.GradientTape` and calls
`apply_gradients()`. If you want to process the gradient before
applying then call `tf.GradientTape` and `apply_gradients()` explicitly
instead of using this function.

#### Args:

* <b>`loss`</b>: A callable taking no arguments which returns the value to
        minimize.
* <b>`var_list`</b>: list or tuple of `Variable` objects to update to
        minimize `loss`, or a callable returning the list or tuple of
        `Variable` objects. Use callable when the variable list would
        otherwise be incomplete before `minimize` since the variables
        are created at the first time `loss` is called.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for
        `loss`.
* <b>`decay_var_list`</b>: Optional list of variables to be decayed. Defaults
        to all variables in var_list.
* <b>`name`</b>: Optional name for the returned operation.

#### Returns:

An Operation that updates the variables in `var_list`.  If
`global_step` was not `None`, that operation also increments
`global_step`.

#### Raises:

* <b>`ValueError`</b>: If some of the variables are not `Variable` objects.



