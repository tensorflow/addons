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
in the eager mode, there need a complicated patch to make this solution works. 

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
