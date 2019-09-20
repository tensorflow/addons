# Contributing

Interested in contributing to TensorFlow Addons? We appreciate all kinds
of help and are working to make this guide as comprehensive as possible.
Please [let us know](https://github.com/tensorflow/addons/issues) if
you think of something we could do to help lower the barrier to
contributing.

## Pull Requests

We gladly welcome [pull requests](
https://help.github.com/articles/about-pull-requests/).

Before making any changes, we recommend opening an issue (if it
doesn't already exist) and discussing your proposed changes. This will
let us give you advice on the proposed changes. If the changes are
minor, then feel free to make them without discussion.

Want to contribute but not sure of what? Here are a few suggestions:
1. Add a new tutorial. Located in [`docs/tutorials/`](docs/tutorials),
  these are a great way to familiarize yourself and others with TF-Addons. See 
  [the guidelines](docs/tutorials/README.md) for more information on how to add 
  examples.
2. Solve an [existing issue](https://github.com/tensorflow/addons/issues).
  These range from low-level software bugs to higher-level design problems.
  Check out the label [help wanted](https://github.com/tensorflow/addons/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22). If you're a new contributor, the label [good first issue](https://github.com/tensorflow/addons/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) can be a good place to start.

All submissions, including submissions by project members, require
review.

## Requirements for New Contributions to the Repository
The tensorflow/addons repository contains additional functionality
fitting the following criteria:

* The functionality is not otherwise available in TensorFlow
* Addons have to be compatible with TensorFlow 2.x.
* The addon conforms to the code and documentation standards
* The addon is impactful to the community (e.g. an implementation used
 in widely cited paper)
 * Lastly, the functionality conforms to the contribution guidelines of
 its subpackage.

**Note: New contributions often require team-members to read a research
paper and understand how it fits into the TensorFlow community. This
process can take longer than typical commit reviews so please bare with
us**


## Development Tips
Try these useful commands below:

* Format code automatically: `bash tools/run_docker.sh -c 'make code-format'`
* Run sanity check: `bash tools/run_docker.sh -c 'make sanity-check'`
* Run CPU unit tests: `bash tools/run_docker.sh -c 'make unit-test'`
* Run GPU unit tests: `bash tools/run_docker.sh -d gpu -c 'make gpu-unit-test'`
* All of the above: `bash tools/run_docker.sh -d gpu -c 'make'`

## Coding style

Addons provides `make code-format` command to format your changes
automatically, don't forget to use it before pushing your code.

Please see our [Style Guide](STYLE_GUIDE.md) for more details.

## Code Testing
#### CI Testing
Nighly CI tests are ran and results can be found on the central README. To 
subscribe for alerts please join the [addons-testing mailing list](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons-testing).

#### Locally Testing CPU

```bash
bash tools/run_docker.sh -c 'make unit-test'
```

or run manually:

```bash
docker run --rm -it -v ${PWD}:/addons -w /addons tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
./configure.sh  # Links project with TensorFlow dependency

bazel test -c opt -k \
--test_timeout 300,450,1200,3600 \
--test_output=all \
//tensorflow_addons/...
```

#### Locally Testing GPU
```bash
bash tools/run_docker.sh -d gpu -c 'make gpu-unit-test'
```

or run manually:

```bash
docker run --runtime=nvidia --rm -it -v ${PWD}:/addons -w /addons tensorflow/tensorflow:custom-op-gpu-ubuntu16 /bin/bash
./configure.sh  # Links project with TensorFlow dependency

bazel test -c opt -k \
--test_timeout 300,450,1200,3600 \
--crosstool_top=//build_deps/toolchains/gcc7_manylinux2010-nvcc-cuda10.0:toolchain \
--test_output=all \
--jobs=1 \
//tensorflow_addons/...
```

## Code Reviews

All submissions, including submissions by project members, require review. We
use Github pull requests for this purpose.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to https://cla.developers.google.com/ to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.
