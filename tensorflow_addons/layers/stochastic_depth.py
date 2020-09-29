import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class StochasticDepth(tf.keras.layers.Layer):
    """Stochastic Depth layer.

    Implements Stochastic Depth as described in
    [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382), to randomly drop residual branches
    in residual architectures.

    Usage:
    Residual architectures with fixed depth, use residual branches that are merged back into the main network
    by adding the residual branch back to the input:

    >>> input = np.ones((1, 3, 3, 1), dtype = np.float32)
    >>> residual = tf.keras.layers.Conv2D(1, 1)(input)
    >>> output = tf.keras.layers.Add()([input, residual])
    >>> output.shape
    TensorShape([1, 3, 3, 1])

    StochasticDepth acts as a drop-in replacement for the addition:

    >>> input = np.ones((1, 3, 3, 1), dtype = np.float32)
    >>> residual = tf.keras.layers.Conv2D(1, 1)(input)
    >>> output = tfa.layers.StochasticDepth()([input, residual])
    >>> output.shape
    TensorShape([1, 3, 3, 1])

    At train time, StochasticDepth returns:

    $$
    x[0] + b_l * x[1],
    $$

    where $b_l$ is a random Bernoulli variable with probability $P(b_l = 1) = p_l$

    At test time, StochasticDepth rescales the activations of the residual branch based on the survival probability ($p_l$):

    $$
    x[0] + p_l * x[1]
    $$

    Arguments:
        survival_probability: float, the probability of the residual branch being kept.

    Call Arguments:
        inputs:  List of `[shortcut, residual]` where `shortcut`, and `residual` are tensors of equal shape.

    Output shape:
        Equal to the shape of inputs `shortcut`, and `residual`
    """

    @typechecked
    def __init__(self, survival_probability: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.survival_probability = survival_probability

    def call(self, x, training=None):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = x

        # Random bernoulli variable indicating whether the branch should be kept or not or not
        b_l = tf.keras.backend.random_bernoulli([], p=self.survival_probability)

        def _call_train():
            return shortcut + b_l * residual

        def _call_test():
            return shortcut + self.survival_probability * residual

        return tf.keras.backend.in_train_phase(
            _call_train, _call_test, training=training
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()

        config = {"survival_probability": self.survival_probability}

        return {**base_config, **config}
