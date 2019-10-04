from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
def crf_accuracy(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
    return crf.get_accuracy(y_true, y_pred)
