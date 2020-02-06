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

This solution has a shortage that this model can not be save and load from disk anymore.

```python
# Save the model
model.save('path_to_my_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')
```

key code snippet of how load loss from disk (as h5 file) in the function `tensorflow_core.python.keras.saving.saving_utils.compile_args_from_training_config`.

loss function must be a class or function that can load from default losses, global custom losses registry or custom_objects passed by user.

Since the layer object was constructed in side the `load_model` function, there is no way to pass a loss object generated from a layer object though custom_objects.

Also I think it even can not be saved to disk. TODO(howl-anderson): add more detailed code later.

## About CRF loss

### Solution 1: inherit from tf.keras.losses.Loss

#### pros
the recommended way to implement a "normal" loss

#### cons

according to the code around tensorflow_core/python/keras/engine/training.py:1651 
`per_sample_losses` returned by `loss_fn.call(y_true, y_pred)` must (or can be converted to) have the same shape with `sample_weight` which default to output `mask` (tensorflow_core/python/keras/engine/training.py:1642) of CRF layer.

but that is not possible because `per_sample_losses` is a 1d tensor and `mask` of CRF is a 2d tensor.

One way to fix it is set output `mark` of crf layer to a 1d tensor, which make the mark is considered as not the same meaning as it's name.

Other way is modified the output of loss class to make `per_sample_losses` to a 2d tensor and properly set the reduce property of the class. It so wired and break the semantic meaning of the interface, should considered to a bad idea.

### Solution 2: implement loss as a function

#### pros

easy to implement and nothing breaks. `mark` property is still a meaningful tensor which standard as a mark.

#### cons

this is a old style way to implement a loss function, which is not the recommend way in TF 2.x.