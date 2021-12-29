# Addons - Metrics

## Contents
https://www.tensorflow.org/addons/api_docs/python/tfa/metrics

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all metrics
must:
 * Inherit from `tf.keras.metrics.Metric`.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`

#### Metric Requirements

Any PR which adds a new metric must ensure that:

1. It inherits from the `tf.keras.metrics.Metric` class.
2. Overrides the `update_state()`, `result()`, and `reset_state()` methods.
3. Implements a `get_config()` method.

The implementation must also ensure that the following cases are well tested and supported:

#### Case I: Evaluate results for a given set of `y_true` and `y_pred` tensors
If you are given a set of `predictions` and the corresponding `ground-truth`, then the end-user should be able to create an instance of the metric and call the instance with the given set to evaluate the quality of predictions. For example, if a PR implements `my_metric`, and you have two tensors `y_pred` and `y_true`, then the end-user should be able to call the metric on this set in the following way:

```python

y_pred = [...]   # tensor representing the predicted values
y_true = [...]   # tensor representing the corresponding ground-truth

m = my_metric(..)
m.update_state(y_true, y_pred)
print("Results: ", m.result().numpy())
```

**Note**: The tensor can be a single example or it can represent a batch.


#### Case II: Classification/Regression models, etc.
Different metrics have different use cases depending on the problem set. If the metric being implemented is valid for more than one scenario, then we suggest splitting the `PR` into multiple small `PRs`. For example, `cross-entropy` implemented as `binary_crossentropy` and `categorical_crossentropy`. 

We are providing a simple example for the same if the above scenario applies to the functionality you are contributing to.
(Please note that this is just a sample and can differ from metric to metric.)

1. **Binary classification**: should work with or without `One-hot encoded labels`

```python

# with no OHE
y_pred = [[0.7], [0.5], [0.3]]   
y_true = [[0.], [1], [0]]

m = my_metric(..)
m.update_state(y_true, y_pred)
print("Results: ", m.result().numpy())

# with OHE
y_pred = [[0.7, 0.3], [0.6, 0.4], [0.2, 0.8]]   
y_true = [[1, 0], [0, 1], [1, 0]]

m = my_metric(..)
m.update_state(y_true, y_pred)
print("Results: ", m.result().numpy())
```

2. **Multiclass-classification**: should work with `One-hot encoded` or `sparse` labels

```python

# with OHE
y_pred = [[0.7, 0.2, 0.1], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]   
y_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

m = my_metric(..)
m.update_state(y_true, y_pred)
print("Results: ", m.result().numpy())

# with sparse labels
y_pred = [[0.7, 0.2, 0.1], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]   
y_true = [[0], [1], [2]]

m = my_metric(..)
m.update_state(y_true, y_pred)
print("Results: ", m.result().numpy())
```
3. **Regression**: (need to discuss any special case if applicable apart from general scenario)

**Note**: The `naming` convention and the `semantics` of the separate implementations for a user should be the same ideally.

#### Case III: `model.fit()` with the `Sequential` or the `Model` API

The metric should work with the `Model` and `Sequential` API in Keras. For example:

```python

model = Model(..)

m = my_metric(...)
model.compile(..., metric=[m])
model.fit(...)
```
For more examples on `metric` in Keras, please check out this [guide](https://keras.io/api/metrics/)

#### Testing Requirements
 * Simple unittests that demonstrate the metric is behaving as expected.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.run_functions_eagerly(True)`.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/tensorflow/addons/blob/master/.github/CODEOWNERS)
