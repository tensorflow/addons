"""Tests for Yogi optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.optimizers import yogi
from tensorflow_addons.utils import test_utils


def yogi_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.01,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-3,
                      l1reg=0.0,
                      l2reg=0.0):
  """Performs Yogi parameter update using numpy.

  Args:
    param: An numpy ndarray of the current parameter.
    g_t: An numpy ndarray of the current gradients.
    t: An numpy ndarray of the current time step.
    m: An numpy ndarray of the 1st moment estimates.
    v: An numpy ndarray of the 2nd moment estimates.
    alpha: A float value of the learning rate.
    beta1: A float value of the exponential decay rate for the 1st moment
      estimates.
    beta2: A float value of the exponential decay rate for the 2nd moment
       estimates.
    epsilon: A float of a small constant for numerical stability.
    l1reg: A float value of L1 regularization
    l2reg: A float value of L2 regularization

  Returns:
    A tuple of numpy ndarrays (param_t, m_t, v_t) representing the
    updated parameters for `param`, `m`, and `v` respectively.
  """
  beta1 = np.array(beta1, dtype=param.dtype)
  beta2 = np.array(beta2, dtype=param.dtype)

  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  g2_t = g_t * g_t
  v_t = v - (1 - beta2) * np.sign(v-g2_t) * g2_t

  per_coord_lr = alpha_t / (np.sqrt(v_t) + epsilon)
  param_t = param - per_coord_lr * m_t

  if l1reg > 0:
    param_t = (param_t - l1reg * per_coord_lr * np.sign(param_t)) / (
        1 + l2reg * per_coord_lr)
    print(param_t.dtype)
    param_t[np.abs(param_t) < l1reg * per_coord_lr] = 0.0
  elif l2reg > 0:
    param_t = param_t/(1 + l2reg * per_coord_lr)
  return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
    local_step = tf.cast(opt.iterations + 1, dtype)
    beta_1_t = tf.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = tf.math.pow(beta_1_t, local_step)
    beta_2_t = tf.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = tf.math.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)

  
@test_utils.run_all_in_graph_and_eager_modes
class YogiOptimizerTest(tf.test.TestCase):
  def _DtypesToTest(self, use_gpu):
    if use_gpu:
      return [tf.float32, tf.float64]
    else:
      return [tf.half, tf.float32, tf.float64]

  def doTestSparse(self, beta1=0.0, l1reg=0.0, l2reg=0.0, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([2]))
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([2]))
        opt = proximal_yogi.ProximalYogiOptimizer(
            beta1=beta1,
            l1_regularization_strength=l1reg,
            l2_regularization_strength=l2reg)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power, beta2_power = opt._get_beta_accumulators()
        self.assertIsNotNone(beta1_power)
        self.assertIsNotNone(beta2_power)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertLen(opt.variables(), 0)
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values.
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Yogi.
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(beta1**(t+1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**(t+1),
                                             self.evaluate(beta2_power))

          var0_np, m0, v0 = yogi_update_numpy(
              var0_np, grads0_np, t, m0, v0,
              beta1=beta1, l1reg=l1reg, l2reg=l2reg)
          var1_np, m1, v1 = yogi_update_numpy(
              var1_np, grads1_np, t, m1, v1,
              beta1=beta1, l1reg=l1reg, l2reg=l2reg)

          # Validate updated params.
          self.assertAllCloseAccordingToType(
              var0_np,
              self.evaluate(var0),
              msg="Updated params 0 do not match in NP and TF")
          self.assertAllCloseAccordingToType(
              var1_np,
              self.evaluate(var1),
              msg="Updated params 1 do not match in NP and TF")

  def testSparse(self):
    self.doTestSparse(use_resource=False)

  def testSparseRegularization(self):
    self.doTestSparse(l1reg=0.1, l2reg=0.2, use_resource=False)

  def testSparseMomentum(self):
    self.doTestSparse(beta1=0.9, use_resource=False)

  def testSparseMomentumRegularization(self):
    self.doTestSparse(beta1=0.9, l1reg=0.1, l2reg=0.2, use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceSparse(self):
    self.doTestSparse(use_resource=True)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceSparseRegularization(self):
    self.doTestSparse(l1reg=0.1, l2reg=0.2, use_resource=True)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceSparseMomentum(self):
    self.doTestSparse(beta1=0.9, use_resource=True)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceSparseMomentumRegularization(self):
    self.doTestSparse(beta1=0.9, l1reg=0.1, l2reg=0.2, use_resource=True)

  def testSparseDevicePlacement(self):
    for index_dtype in [dtypes.int32, dtypes.int64]:
      with self.test_session(force_gpu=test.is_gpu_available()):
        # If a GPU is available, tests that all optimizer ops can be placed on
        # it (i.e. they have GPU kernels).
        var = variables.Variable([[1.0], [2.0]])
        indices = constant_op.constant([0, 1], dtype=index_dtype)
        gathered_sum = math_ops.reduce_sum(array_ops.gather(var, indices))
        optimizer = proximal_yogi.ProximalYogiOptimizer(3.0)
        minimize_op = optimizer.minimize(gathered_sum)
        variables.global_variables_initializer().run()
        minimize_op.run()

  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        repeated_update = proximal_yogi.ProximalYogiOptimizer().apply_gradients(
            [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = proximal_yogi.ProximalYogiOptimizer(
            ).apply_gradients([(grad_aggregated, aggregated_update_var)])
        variables.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(),
                            repeated_index_update_var.eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var.eval(),
                              repeated_index_update_var.eval())

  def doTestBasic(self, beta1=0.0, l1reg=0.0, l2reg=0.0, use_resource=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.test_session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = proximal_yogi.ProximalYogiOptimizer(
            beta1=beta1,
            l1_regularization_strength=l1reg,
            l2_regularization_strength=l2reg)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power, beta2_power = opt._get_beta_accumulators()
        self.assertIsNotNone(beta1_power)
        self.assertIsNotNone(beta2_power)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values.
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Yogi.
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(beta1**(t + 1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta2_power))

          var0_np, m0, v0 = yogi_update_numpy(
              var0_np, grads0_np, t, m0, v0,
              beta1=beta1, l1reg=l1reg, l2reg=l2reg)
          var1_np, m1, v1 = yogi_update_numpy(
              var1_np, grads1_np, t, m1, v1,
              beta1=beta1, l1reg=l1reg, l2reg=l2reg)

          # Validate updated params.
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
          if use_resource:
            self.assertEqual("var0_%d/ProximalYogi:0" % (i,),
                             opt.get_slot(var=var0, name="v").name)

  def testBasic(self):
    self.doTestBasic(use_resource=False)

  def testBasicRegularization(self):
    self.doTestBasic(l1reg=0.1, l2reg=0.2, use_resource=False)

  def testBasicMomentum(self):
    self.doTestBasic(beta1=0.9, use_resource=False)

  def testBasicMomentumRegularization(self):
    self.doTestBasic(beta1=0.9, l1reg=0.1, l2reg=0.2, use_resource=False)

  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = proximal_yogi.ProximalYogiOptimizer(constant_op.constant(0.01))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values.
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Yogi.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = yogi_update_numpy(
              var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = yogi_update_numpy(
              var1_np, grads1_np, t, m1, v1)

          # Validate updated params.
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSharing(self):
    for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
      # Initialize variables for numpy implementation.
      m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

      var0 = tf.Variable(var0_np)
      var1 = tf.Variable(var1_np)
      grads0 = tf.constant(grads0_np)
      grads1 = tf.constant(grads1_np)
      opt = yogi.Yogi()
      
      if not tf.executing_eagerly():
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(tf.compat.v1.global_variables_initializer())

      # Fetch params to validate initial values.
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())
        
        

        # Run 3 steps of intertwined Yogi1 and Yogi2.
        for t in range(1, 4):
          beta1_power, beta2_power = get_beta_accumulators()
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t, self.evaluate(beta2_power))
          if not tf.executing_eagerly():
            if t % 2 == 0:
              update1.run()
            else:
              update2.run()
          else:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))


          var0_np, m0, v0 = yogi_update_numpy(
              var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = yogi_update_numpy(
              var1_np, grads1_np, t, m1, v1)

          # Validate updated params.
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def test_get_config(self):
    opt = yogi.Yogi(1e-4)
    config = opt.get_config()
    self.assertEqual(config['learning_rate'], 1e-4)



if __name__ == "__main__":
  tf.test.main()
