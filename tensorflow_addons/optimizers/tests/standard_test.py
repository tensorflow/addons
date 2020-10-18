import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons import optimizers
import inspect
import sys

classes_to_test = ['RectifiedAdam', 'LazyAdam']

def discover_classes(module, parent):

    classes = [m[1] for m in inspect.getmembers(module, inspect.isclass)
               if issubclass(m[1], parent) and m[0] in classes_to_test ]

    return classes

@pytest.mark.parametrize("optimizer", discover_classes(optimizers, tf.keras.optimizers.Optimizer))
def test_optimizer_minimize(optimizer):

    model = tf.keras.Sequential(
        [tf.keras.Input(shape=[1]), tf.keras.layers.Dense(1)]
    )

    x = np.array(np.ones([1]))
    y = np.array(np.zeros([1]))

    opt = optimizer()
    loss = tf.keras.losses.MSE

    model.compile(optimizer=opt, loss=loss)

    history = model.fit(x, y, batch_size=1, epochs=10)

    loss_values = history.history['loss']

    np.testing.assert_array_less(loss_values[-1], loss_values[0])

if __name__ == '__main__':
    sys.exit(pytest.main(['standard_test.py']))