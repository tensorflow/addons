import tensorflow as tf

BASE_OPTIMIZER_CLASS = tf.keras.optimizers.legacy.Optimizer
if tf.__version__[:3] <= "2.8":
    BASE_OPTIMIZER_CLASS = tf.keras.optimizers.Optimizer
