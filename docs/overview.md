<div align="center">
  <img src="https://tensorflow.org/images/SIGAddons.png" width="60%"><br><br>
</div>

-----------------

# TensorFlow Addons

**TensorFlow Addons** is a repository of contributions that conform to
well-established API patterns, but implement new functionality
not available in core TensorFlow. TensorFlow natively supports
a large number of operators, layers, metrics, losses, and optimizers.
However, in a fast moving field like ML, there are many interesting new
developments that cannot be integrated into core TensorFlow
(because their broad applicability is not yet clear, or it is mostly
 used by a smaller subset of the community).


## Installation

#### Stable Builds
To install the latest version, run the following:

```
pip install tensorflow-addons
```

To use addons:

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### Nightly Builds
There are also nightly builds of TensorFlow Addons under the pip package
`tfa-nightly`, which is built against the latest stable version of TensorFlow. Nightly builds
include newer features, but may be less stable than the versioned releases.

```
pip install tfa-nightly
```

#### Installing from Source
You can also install from source. This requires the [Bazel](
https://bazel.build/) build system.

```
git clone https://github.com/tensorflow/addons.git
cd addons

# If building GPU Ops (Requires CUDA 10.0 and CuDNN 7)
export TF_NEED_CUDA=1
export CUDA_TOOLKIT_PATH="/path/to/cuda10" (default: /usr/local/cuda)
export CUDNN_INSTALL_PATH="/path/to/cudnn" (default: /usr/lib/x86_64-linux-gnu)

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```


## Core Concepts

#### Standardized API within Subpackages
User experience and project maintainability are core concepts in
TF-Addons. In order to achieve these we require that our additions
conform to established API patterns seen in core TensorFlow.

#### GPU/CPU Custom-Ops
A major benefit of TensorFlow Addons is that there are precompiled ops. Should 
a CUDA 10 installation not be found then the op will automatically fall back to 
a CPU implementation.

#### Proxy Maintainership
Addons has been designed to compartmentalize subpackages and submodules so 
that they can be maintained by users who have expertise and a vested interest 
in that component. 

Subpackage maintainership will only be granted after substantial contribution 
has been made in order to limit the number of users with write permission. 
Contributions can come in the form of issue closings, bug fixes, documentation, 
new code, or optimizing existing code. Submodule maintainership can be granted 
with a lower barrier for entry as this will not include write permissions to 
the repo.

For more information see [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md) 
on this topic.

#### Periodic Evaluation of Subpackages
Given the nature of this repository, subpackages and submodules may become less 
and less useful to the community as time goes on. In order to keep the 
repository sustainable, we'll be performing bi-annual reviews of our code to 
ensure everything still belongs within the repo. Contributing factors to this 
review will be:

1. Number of active maintainers
2. Amount of OSS use
3. Amount of issues or bugs attributed to the code
4. If a better solution is now available

Functionality within TensorFlow Addons can be categorized into three groups:

* **Suggested**: well-maintained API; use is encouraged.
* **Discouraged**: a better alternative is available; the API is kept for 
historic reasons; or the API requires maintenance and is the waiting period 
to be deprecated.
* **Deprecated**: use at your own risk; subject to be deleted.

The status change between these three groups is: 
Suggested <-> Discouraged -> Deprecated.

The period between an API being marked as deprecated and being deleted will be 
90 days. The rationale being:

1. In the event that TensorFlow Addons releases monthly, there will be 2-3 
releases before an API is deleted. The release notes could give user enough 
warning.

2. 90 days gives maintainers ample time to fix their code.


## Contributing
TF-Addons is a community led open source project. As such, the project
depends on public contributions, bug-fixes, and documentation. Please see 
[contribution guidelines](https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md) 
for a guide on how to contribute. This project adheres to [TensorFlow's code of conduct](https://github.com/tensorflow/addons/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Community
* [Public Mailing List](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
* [SIG Monthly Meeting Notes](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    * Join our mailing list and receive calendar invites to the meeting

## License
[Apache License 2.0](LICENSE)
