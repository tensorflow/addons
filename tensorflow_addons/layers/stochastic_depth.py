import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class StochasticDepth(tf.keras.layers.Layer):
    r"""Stochastic Depth layer.

    Implements Stochastic Depth as described in
    [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382), to randomly drop residual branches
    in residual architectures.

    Usage:
    Residual architectures with fixed depth, use residual branches that are merged back into the main network
    by adding the residual branch back to the input:

    ```python
    residual = tf.keras.layers.Conv2D(...)(input)

    return tf.keras.layers.Add()([input, residual])
    ```

    StochasticDepth acts as a drop-in replacement for the addition:

    ```python
    residual = tf.keras.layers.Conv2D(...)(input)

    return tfa.layers.StochasticDepth()([input, residual])
    ```

    At train time, StochasticDepth returns:

    ```python
    x[0] + b_l * x[1]
    ```

    , where b_l is a random Bernoulli variable with probability p(b_l == 1) == p_l

    At test time, StochasticDepth rescales the activations of the residual branch based on the survival probability:

    ```python
    x[0] + p_l * x[1]
    ```

    Arguments:
        p_l: float, the probability of the residual branch being kept.

    Call Arguments:
        inputs:  List of `[shortcut, residual]` where
            * `shortcut`, and `residual` are tensors of equal shape.

    Output shape:
        Equal to the shape of inputs `shortcut`, and `residual`
    """

    @typechecked
    def __init__(self, p_l: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.p_l = p_l

    def call(self, x, training=None):
        assert isinstance(x, list)

        shortcut, residual = x

        # Random bernoulli variable with probability p_l, indiciathing wheter the branch should be kept or not or not
        b_l = tf.keras.backend.random_binomial([], p=self.p_l)

        def _call_train():
            return shortcut + b_l * residual

        def _call_test():
            return shortcut + self.p_l * residual

        return tf.keras.backend.in_train_phase(
            _call_train, _call_test, training=training
        )

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()

        config = {"p_l": self.p_l}

        return {**base_config, **config}
