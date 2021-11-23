import tensorflow as tf

from tensorflow_addons.text import crf_log_likelihood
from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
class CRFModelWrapper(tf.keras.Model):
    def __init__(
        self,
        base_model: tf.keras.Model,
        units: int,
        chain_initializer: types.Initializer = "orthogonal",
        use_boundary: bool = True,
        boundary_initializer: types.Initializer = "zeros",
        use_kernel: bool = True,
        **kwargs,
    ):
        super().__init__()

        # lazy import to solve circle import issue:
        # tfa.layers.CRF -> tfa.text.__init__ -> tfa.text.crf_wrapper -> tfa.layers.CRF
        from tensorflow_addons.layers.crf import CRF  # noqa

        self.crf_layer = CRF(
            units=units,
            chain_initializer=chain_initializer,
            use_boundary=use_boundary,
            boundary_initializer=boundary_initializer,
            use_kernel=use_kernel,
            **kwargs,
        )

        self.base_model = base_model

    def unpack_training_data(self, data):
        # override me, if this is not suit for your task
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        return x, y, sample_weight

    def call(self, inputs, training=None, mask=None, return_crf_internal=False):
        base_model_outputs = self.base_model(inputs, training, mask)

        # change next line, if your model has more outputs
        crf_input = base_model_outputs

        decode_sequence, potentials, sequence_length, kernel = self.crf_layer(crf_input)

        # change next line, if your base model has more outputs
        # Aways keep `(potentials, sequence_length, kernel), decode_sequence, `
        # as first two outputs of model.
        # current `self.train_step()` expected such settings
        outputs = (potentials, sequence_length, kernel), decode_sequence

        if return_crf_internal:
            return outputs
        else:
            # outputs[0] is the crf internal, skip it
            output_without_crf_internal = outputs[1:]

            # it is nicer to return a tensor instead of an one tensor list
            if len(output_without_crf_internal) == 1:
                return output_without_crf_internal[0]
            else:
                return output_without_crf_internal

    def compute_crf_loss(
        self, potentials, sequence_length, kernel, y, sample_weight=None
    ):
        crf_likelihood, _ = crf_log_likelihood(potentials, y, sequence_length, kernel)
        # convert likelihood to loss
        flat_crf_loss = -1 * crf_likelihood
        if sample_weight is not None:
            flat_crf_loss = flat_crf_loss * sample_weight
        crf_loss = tf.reduce_mean(flat_crf_loss)

        return crf_loss

    def train_step(self, data):
        x, y, sample_weight = self.unpack_training_data(data)
        with tf.GradientTape() as tape:
            (potentials, sequence_length, kernel), decoded_sequence, *_ = self(
                x, training=True, return_crf_internal=True
            )
            crf_loss = self.compute_crf_loss(
                potentials, sequence_length, kernel, y, sample_weight
            )
            loss = crf_loss + tf.reduce_sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, decoded_sequence)
        # Return a dict mapping metric names to current value
        orig_results = {m.name: m.result() for m in self.metrics}
        crf_results = {"loss": loss, "crf_loss": crf_loss}
        return {**orig_results, **crf_results}

    def test_step(self, data):
        x, y, sample_weight = self.unpack_training_data(data)
        (potentials, sequence_length, kernel), decode_sequence, *_ = self(
            x, training=False, return_crf_internal=True
        )
        crf_loss = self.compute_crf_loss(
            potentials, sequence_length, kernel, y, sample_weight
        )
        loss = crf_loss + tf.reduce_sum(self.losses)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, decode_sequence)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "crf_loss": crf_loss})  # append loss
        return results

    def get_config(self):
        base_model_config = self.base_model.get_config()
        crf_config = self.crf_layer.get_config()

        return {**{"base_model": base_model_config}, **crf_config}

    @classmethod
    def from_config(cls, config):
        base_model_config = config.pop("base_model")
        base_model = tf.keras.Model.from_config(base_model_config)

        return cls(base_model=base_model, **config)
