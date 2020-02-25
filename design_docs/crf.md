# Some technical selection in implementation of CRF layer
## About CRF loss function
currently the crf loss function is desinged as a seperated method/Class. 
### Solution 1: standalone loss
In usage it look like below

```python
from tensorflow_addons.layers import CRF
from tensorflow_addons.losses import crf_loss

model = Sequential()
model.add(Embedding(3001, 300, mask_zero=True)

crf = CRF(10)
model.add(crf)

model.compile('adam', loss=crf_loss)

model.fit(x, y)
```

#### pros ####
the standard way to use loss

#### cons ####
in the eager mode, there need override a private of base layer to make this solution works. 

code:
```python
def __call__(self, inputs, *args, **kwargs):
    outputs = super(CRF, self).__call__(inputs, *args, **kwargs)

    # A hack that add _keras_history to EagerTensor, make it more like normal Tensor
    for tensor in tf.nest.flatten(outputs):
        if not hasattr(tensor, '_keras_history'):
            tensor._keras_history = (self, 0, 0)

    return outputs
```

Maybe this patch should submit to tensorflow-core which can also help others to implement a loss function easier for a complicated layer (such like CRF) 

### Solution 2: get from crf layer ###
In usage it look like below

```python
from tensorflow_addons.layers import CRF

model = Sequential()
model.add(Embedding(3001, 300, mask_zero=True)

crf = CRF(10)
model.add(crf)

crf_loss = crf.get_keras_loss()

model.compile('adam', loss=crf_loss)

model.fit(x, y)
```

#### pros ####
easy to implement and no more need patch

#### cons ####

This solution has a shortage that load model from disk will be difficult.

##### TensorFlow's default load process don't work #####

```python
# Save the model
model.save('path_to_my_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')
```

The reason is when Keras core reconstruct the model from disk, it will construct layer and loss from disk independently, so the new loss instance don't have the reference to the new CRF layer instance, therefore the loss function don't work anymore.

##### A workaround solution (not prefect) #####
TODO: add a PoC code for this

This a workaround solution for loading CRF model from disk.

1. Load the model without compile
```python
new_model = keras.models.load_model('path_to_my_model.h5', compile=Flase)
```

2. Get the CRF layer instance
```python
# normally, crf layer is the last layer
crf_layer_instance = new_model.get_layer(index=-1)
```

3. Get the CRF loss instance from layer instance
```python
crf_loss_instance = crf_layer_instance.get_keras_loss()
```

4. Compile the model
```python
new_model.compile(loss=crf_loss_instance)
```

The shortage of this method is user need to add extract code to load the model and all the arguments except the loss passed to model's compile method before will not longer remembered, user need to pass to it again (if their still remember it)

## About CRF loss

### Solution 1: inherit from tf.keras.losses.Loss

#### pros
the recommended way to implement a "normal" loss

#### cons
according to the code around `tensorflow_core/python/keras/engine/training.py:1651`
`per_sample_losses` returned by `loss_fn.call(y_true, y_pred)` must (or can be converted to) have the same shape with `sample_weight` which default to output `mask` (tensorflow_core/python/keras/engine/training.py:1642) of CRF layer.

but that is not possible because `per_sample_losses` is a 1d tensor and `mask` of CRF is a 2d tensor.

One way to fix it is set output `mark` of crf layer to a 1d tensor, which make the mark is considered as not the same meaning as it's name.

Other way is modified the output of loss class to make `per_sample_losses` to a 2d tensor and properly set the reduce property of the class. It so wired and break the semantic meaning of the interface, should considered to a bad idea.


### Solution 2: implement as a function ###

#### pros ####
This is a old but standard (keras style) way to implement the loss function

#### cons ####
TensorFlow will convert a loss function into a subclass of `tf.keras.losses.Loss` in `` file by `` (call chain: `tf.keras.Model::compile()` [Line: 314] > `tensorflow/python/keras/engine/training_utils.py::prepare_loss_functions` [Line: 1501] > `tensorflow/python/keras/engine/training_utils.py::get_loss_function` [Line: 1186]).

```python
  # For losses which are given as strings/functions in the compile API,
  # we always set the loss reduction type to be `SUM_OVER_BATCH_SIZE`
  # (both in distribution strategy context and otherwise).
  return losses.LossFunctionWrapper(
      loss_fn,
      name=loss_fn.__name__,
      reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)
```

So it has same issue that solution 1.

### Solution 3: implement loss as a callable class

#### pros
Nothing breaks. `mark` property is still a meaningful tensor which standard as a mark.

#### cons
this solution need understanding how keras process a loss function, which is not documented and not recommend way in TF 2.x.
