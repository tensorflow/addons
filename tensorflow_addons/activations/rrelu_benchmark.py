import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import rrelu
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class RreluBenchmark(tf.test.Benchmark):
    def benchmarkRreluOp(self):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        lower = 0.1
        upper = 0.2
        with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session(
                config=tf.test.benchmark_config()) as sess:
            result = rrelu(x, lower, upper, training=True)
            self.run_op_benchmark(
                sess, result, min_iters=25, name="rrelu_test")


if __name__ == "__main__":
    tf.test.main()
