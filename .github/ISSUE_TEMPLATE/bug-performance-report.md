---
name: Bug/Performance report
about: Create a report to help us improve
title: "[Bug/Performance]"
labels:
assignees: ''

---

**Describe the bug**

A clear and concise description of what the bug is.

**System information**
- Have I written custom code (as opposed to using a stock example script provided in TensorFlow):
- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- TensorFlow installed from (source or binary):
- TensorFlow version (use command below):
- TensorFlow Addons installed from (source, PyPi):
- TensorFlow Addons version:
- Python version and type(eg. Anaconda Python, Stock Python as in Mac, or homebrew installed Python etc):
- Bazel version (if compiling from source):
- GCC/Compiler version (if compiling from source):
- Is GPU used? : [yes/no]
- CUDA/cuDNN version (if compiling/running with GPGPU):
- GPGPU model and memory:

You can collect some of this information using our environment capture [script](https://github.com/tensorflow/tensorflow/tree/master/tools/tf_env_collect.sh)
You can also obtain the TensorFlow version with
python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"

**Describe the expected behavior**

**Code to reproduce the issue**

Provide a reproducible test case that is the bare minimum necessary to generate the problem.

**Other info / logs**

Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.
