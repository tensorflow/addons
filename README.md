**Addons** is a repository of contributions that conform to
well-established API patterns, but implement new functionality
not available in core TensorFlow. TensorFlow natively supports
a larger number of operators, layers, metrics, losses, and optimizers.
However, in a fast moving field like ML, there are many interesting new
developments that cannot be integrated into core TensorFlow
(because their broad applicability is not yet clear, or it is mostly used by a smaller
subset of the community).

# Scope
The tensorflow/addons repository, will contain additional functionality fitting the following criteria:

* The functionality is not otherwise available in TensorFlow
* The functionality conforms to an established API pattern in TensorFlow. For instance, it could be an additional subclass of an existing interface (new Layer, Metric, or Optimizer subclasses), or an additional Op or OpKernel implementation.
* Addons have to be compatible with TensorFlow 2.x.
* The addon conforms to the code and documentation standards defined by the group.
* The addon is useful for a large number of users (e.g., an implementation used in widely cited paper, or a utility with broad applicability)


# Developing

## Docker
```
docker run --rm -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:nightly-custom-op /bin/bash
```

## Packaging
```
# In docker
./configure.sh
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
```

A package file artifacts/tensorflow_addons-*.whl will be generated after a build is successful.


## Testing
```
# In docker
./configure.sh
bazel test //tensorflow_addons/...
```