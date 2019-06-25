import tensorflow as tf

from tensorflow_addons.text.crf import crf_decode


class CRF(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CRF, self).__init__()
        self.units = units  # numbers of tags

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.chain_kernel = self.add_weight(shape=(self.units, self.units),
                                            name='chain_kernel',
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = 0

    def call(self, input, **kwargs):
        logits = self._dense_layer(input)
        pred_ids, _ = crf_decode(logits, self.chain_kernel, nwords)

    def _dense_layer(self, input):
        # TODO: can simply use tf.keras.layers.dense ?
        return self.activation(tf.matmul(input, self.kernel) + self.bias)


if __name__ == "__main__":
    layer = CRF(10)
    print(layer(tf.zeros([10, 5])))
    print(layer.trainable_variables)
