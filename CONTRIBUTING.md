# Contributing

Interested in contributing to TensorFlow Addons? We appreciate all kinds
of help and are working to make this guide as comprehensive as possible.
Please [let us know](https://github.com/tensorflow/addons/issues) if
you think of something we could do to help lower the barrier to
contributing.

## Pull Requests

We gladly welcome [pull requests](
https://help.github.com/articles/about-pull-requests/).

Have you ever done a pull request with GitHub? 
If not we recommend you to read 
[this guide](https://github.com/gabrieldemarmiesse/getting_started_open_source) 
to get your started.

Before making any changes, we recommend opening an issue (if it
doesn't already exist) and discussing your proposed changes. This will
let us give you advice on the proposed changes. If the changes are
minor, then feel free to make them without discussion.

All submissions, including submissions by project members, require
review.

## Requirements for New Contributions to the Repository

**All new components/features to Addons need to first be submitted as a feature 
request issue. This will allow the team to check with our counterparts in the TF
ecosystem and ensure it is not roadmapped internally for Keras or TF core. These 
feature requests will be labeled with `ecosystem-review` while we determine if it 
should be included in Addons.**

The tensorflow/addons repository contains additional functionality
fitting the following criteria:

* The functionality is not otherwise available in TensorFlow
* Addons have to be compatible with TensorFlow 2.x.
* The addon conforms to the code and documentation standards
* The addon is impactful to the community (e.g. an implementation used
 in widely cited paper)
 * Lastly, the functionality conforms to the contribution guidelines of
 its subpackage.

Suggested guidelines for new feature requests:

* The feature contains an official reference implementation.
* Should be able to reproduce the same results in a published paper.
* The academic paper exceeds 50 citations.

**Note: New contributions often require team-members to read a research
paper and understand how it fits into the TensorFlow community. This
process can take longer than typical commit reviews so please bare with
us**

## Tools needed for developement

#### Linux

* Docker (code formatting / testing)
* Nvidia-docker (for GPU testing, optional)
* Bazel installed locally (to build custom ops locally, optional)
* NVCC/Cuda installed locally (to build custom ops with gpu locally, optional)

#### MacOS

* Docker (code formatting / testing)
* Bazel installed locally (to build custom ops locally, optional)

#### Windows

For Windows, you have two options:

##### WSL 2

WSL 2 is a very light virtual machine running with hyper-V. When running in 
WSL 2, you're in a full linux environment, with a real linux kernel. 
WSL 2 networking is shared with Windows and your Windows files can be found under 
`/mnt/c`. When working with WSL 2, you can just follow the linux guides and tutorials
and everything will work as in linux, including 
Docker (that means you install docker with apt-get), git, ssh...
See [the WSL 2 install guide](https://docs.microsoft.com/en-us/windows/wsl/wsl2-install).

##### Powershell in Windows

This is if you want to stay in Windows world. In this case, you need:

* Git with git bash install on the PATH (meaning that you can run the `sh` command from Powershell).
* Docker desktop with Linux containers (code format, testing on linux, etc...)
* A local Python installation
* Bazel (if you want to compile custom ops on Windows, optional)
* Visual Studio build tools 2019
[install with chocolatey](https://chocolatey.org/packages/visualstudio2019buildtools) or
 [install manually](https://www.tensorflow.org/install/source_windows#install_visual_c_build_tools_2019)
 (if you want to compile custom ops on windows, optional).
 
 If you develop on Windows and you encounter issues, we'd be happy to have your feedback!
 [This link](https://github.com/tensorflow/addons/issues/1134) might help you.

## Development Tips
Try these useful commands below, they only use Docker and 
don't require anything else (not even python installed):

* Format code automatically: `bash tools/pre-commit.sh`
* Run sanity check: `bash tools/run_sanity_check.sh`
* Run CPU unit tests: `bash tools/run_cpu_tests.sh`
* Run GPU unit tests: `bash tools/run_gpu_tests.sh`

If you're running Powershell on Windows, use `sh` instead of `bash` when typing the commands.

## Coding style

We provide a pre-commit hook to format your code automatically before each
commit, so that you don't have to read our style guide. Install it on Linux/MacOS with

```bash
cd .git/hooks && ln -s -f ../../tools/pre-commit.sh pre-commit
```

and you're good to go.

On Windows, in powershell, do:

```bash
cd .git/hooks
cmd /c mklink pre-commit ..\..\tools\pre-commit.sh
```

Note that this pre-commit needs Docker to run. 
If you have docker 19.03+, it uses
[Docker buildkit](https://docs.docker.com/develop/develop-images/build_enhancements/) 
to make the build step much faster.

See our [Style Guide](STYLE_GUIDE.md) for more details.

## Code Testing
### CI Testing
Nightly CI tests are ran and results can be found on the central README. To
subscribe for alerts please join the [addons-testing mailing list](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons-testing).


### Testing locally, without Docker

When running outside Docker, you can use your IDE to debug, and use your local tools to work.

If you're just modifying Python code (as opposed to C++/CUDA code), 
then you don't need to use Bazel to run your tests. 
And you don't need to compile anything.

#### Optional but recommended, creating a virtual environment

If you want to work in 
a [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/):

```bash
pip install virtualenv
venv my_dev_environement
source my_dev_environement/bin/activate  # Linux/macos/WSL2
.\my_dev_environement\Scripts\activate   # PowerShell
```

If you want to work in 
a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
```bash
conda create --name my_dev_environement
conda activate my_dev_environement
```


#### Install TensorFlow Addons in editable mode

Just run from the root:

```bash
pip install tensorflow==2.12.0
# you can use "pip install tensorflow-cpu==2.12.0" too if you're not testing on gpu.
pip install -e ./
```

It's going to install Addons in editable mode without compiling anything.
You can modify source files and changes will be seen at the next Python 
interpreter startup. This command needs to be executed only once. 
Now, anywhere on your system, if you do `import tensorflow_addons`, it's 
going to import the code in this git repository.

#### Uninstall TensorFlow Addons

To undo this operation, for example, you want to later on 
install TensorFlow Addons from PyPI, the release version, do:

```bash
pip uninstall tensorflow-addons
```

#### Run the tests with pytest

If TensorFlow Addons is installed in editable mode, you can then just run your tests by 
running Pytest. For example:
```bash
pip install -r tools/install_deps/pytest.txt
python -m pytest tensorflow_addons/rnn/tests/cell_test.py
# or even
python -m pytest tensorflow_addons/rnn/
# or even 
python -m pytest tensorflow_addons/
# or even if pytest is in the PATH
pytest tensorflow_addons/
```

Pytest has many cool options to help you make great tests:

```bash
# Use multiprocessing to run the tests, 3 workers
pytest -n 3 tensorflow_addons/
pytest -n auto tensorflow_addons/

# Run the whole test suite without compiling any custom ops (.so files).
pytest -v --skip-custom-ops tensorflow_addons/

# Open the debugger to inspect variables and execute code when 
# an exception is raised.
pytest --pdb tensorflow_addons/ 

# or if you prefer the Ipython debugger
pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb --capture no tensorflow_addons/

# by defaults print() aren't displayed with pytest
# if you like to debug with prints (you might get 
# the output scrambled)
pytest -s tensorflow_addons/

# get the list of functions you ran
pytest -v tensorflow_addons/

# to rerun all previous tests, running the ones that failed first
pytest --ff tensorflow_addons/

# You know which function to execute, but you're too 
# lazy to type the file path
pytest -k "test_get_all_shared_objects" ./tensorflow_addons/

# get the 10 slowest functions
pytest --duration=10 tensorflow_addons/
```

#### Testing with Pycharm

Pycharm has a debugger build in the IDE for visual inspection of variables
and step by step executions of Python instructions. It can run your test 
functions from the little green arrows next to it. And you can add 
 breakpoints by just clicking next to a line in the code (a red dot will appear). 
 
 But in order for the debugger to run correctly, you need to specify 
 that you use pytest as your main test runner, not unittest (the default one). 
 
 For that, go in File -> Settings -> search box -> Default test runner -> Select "Pytest".

#### Compiling custom ops

If you need a custom C++/Cuda op for your test, compile your ops with

```bash
python configure.py
pip install tensorflow==2.12.0 -e ./ -r tools/install_deps/pytest.txt
bash tools/install_so_files.sh  # Linux/macos/WSL2
sh tools/install_so_files.sh    # PowerShell
```

Note that you need bazel, a C++ compiler and a NVCC compiler (if you want to test
Cuda ops). For that reason, we recommend you run inside the custom-op docker containers. 
This will avoid you the hassle of installing Bazel, GCC/clang...
See below.


#### Run Manually

Running tests interactively in Docker gives you good flexibility and doesn't require 
to install any additional tools.

CPU Docker: 
```bash
docker run --rm -it -v ${PWD}:/addons -w /addons tfaddons/dev_container:latest-cpu
```

GPU Docker: 
```bash
docker run --gpus all --rm -it -v ${PWD}:/addons -w /addons gcr.io/tensorflow-testing/nosla-cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-multipython
```

Configure:
```bash
python3 -m pip install tensorflow==2.12.0
python3 ./configure.py  # Links project with TensorFlow dependency
```

Install in editable mode
```bash
python3 -m pip install -e .
python3 -m pip install -r tools/install_deps/pytest.txt
```

Compile the custom ops
```bash
export TF_NEED_CUDA=1 # If GPU is to be used
bash tools/install_so_files.sh
```

Run selected tests:
```bash
python3 -m pytest path/to/file/or/directory/to/test
```

Run the gpu only tests with `pytest -m needs_gpu ./tensorflow_addons`.
Run the cpu only tests with `pytest -m 'not needs_gpu' ./tensorflow_addons`.


#### Testing with Bazel

Testing with Bazel is still supported but not recommended unless you have prior experience 
with Bazel, and would like to use it for specific capabilities (Remote execution, etc).
This is because pytest offers many more options to run your test suite and has
better error reports, timings reports, open-source plugins and documentation online 
for Python testing. 

Internally, Google can use Bazel to test many commits 
quickly, as Bazel has great support for caching and distributed testing.

To test with Bazel:

```bash
python3 -m pip install tensorflow==2.12.0
python3 configure.py
python3 -m pip install -r tools/install_deps/pytest.txt
bazel test -c opt -k \
--test_timeout 300,450,1200,3600 \
--test_output=all \
--run_under=$(readlink -f tools/testing/parallel_gpu_execute.sh) \
//tensorflow_addons/...
```

#### Testing docstrings

We use [DocTest](https://docs.python.org/3/library/doctest.html) to test code snippets
in Python docstrings. The snippet must be executable Python code.
To enable testing, prepend the line with `>>>` (three left-angle brackets).
Available namespace include `np` for numpy, `tf` for TensorFlow, and `tfa` for TensorFlow Addons.
See [docs_ref](https://www.tensorflow.org/community/contribute/docs_ref) for more details.

To test docstrings locally, run either
```bash
bash tools/run_cpu_tests.sh
```
on all files, or
```bash
pytest -v -n auto --durations=25 --doctest-modules /path/to/pyfile
```
on specific files.

## About type hints

Ideally, we would like all the functions and classes constructors exposed in 
the public API to be have type hints (adding the return type for class 
constructors is not necessary).

We do so to improve the user experience. Some users might use IDEs or static
type checking, and having types greatly improve productivity with those tools.

If you are not familiar with type hints, you can read 
the [PEP 484](https://www.python.org/dev/peps/pep-0484/).

We also have a runtime type check that we do 
using [typeguard](https://typeguard.readthedocs.io/en/latest/).
For an example, see the [normalizations.py file](tensorflow_addons/layers/normalizations.py).
Please add it if you type a class constructor (Note that the decorator doesn't 
play nice with autograph at the moment, this is why we don't add it to functions. For more
context, see [this pull request](https://github.com/tensorflow/addons/pull/928)).

You can import some common types 
from [tensorflow_addons/utils/types.py](tensorflow_addons/utils/types.py).

We recommend adding types if you add a new class/function to Addons' public API, 
but we don't enforce it.

Since adding type hints can be hard, especially for people who are not
familiar with it, we made a big todo-list of functions/class constructors that 
need typing. If you want to add a feature to the public API and 
don't want to bother adding type hints, please add your feature to the todo-list 
in [tools/testing/source_code_test.py](tools/testing/source_code_test.py).

Help is welcome to make this TODO list smaller!

## Writing tests

If you add a new feature, you should add tests to ensure that new code changes 
doesn't introduce bugs.

If you fix a bug, you should add a tests which fails before your patch and passes 
after your patch.

We use [Pytest](https://docs.pytest.org/en/latest/) to write tests. We encourage you
to read the documentation, but you'll find a quick summary here:

* If you're testing code written in `xxx.py`, your tests should be in `xxx_test.py`.
* In `xxx_test.py`, all functions starting with `test_` are collected and run by Pytest.
* Tests are run with the TF 2.x behavior, meaning eager mode my default, unless you use a `tf.function`.
* Ensure something is working by using `assert`. For example: `assert my_variable in my_list`.
* When comparing numpy arrays, use 
the [testing module of numpy](https://docs.scipy.org/doc/numpy/reference/routines.testing.html).
Note that since TensorFlow ops often run with float32 of float16, you might need to 
increase the default `atol` and `rtol`. You can take a look at [the default values used 
in the TensorFlow repository](https://www.tensorflow.org/api_docs/python/tf/test/TestCase#assertAllClose).
* Prefer using your code's public API when writing tests. It ensures future refactoring is possible
without changing the tests.
* When testing multiple configurations, prefer using
 [parametrize](https://docs.pytest.org/en/latest/parametrize.html) rather than for 
 loops for a clearer error report.
* Running all the tests in a single file should take no more than 5 seconds. You very 
rarely need to do heavy computation to test things. Your tests should be small and 
focused on a specific feature/parameter.
* Don't be afraid to write too many tests. This is fine as long as they're fast.

### Code example
* It is required to contribute a code example in the docstring when adding new features.
* It is strongly suggested to expand or contribute a new [tutorial](https://github.com/tensorflow/addons/blob/master/docs/tutorials/README.md) for more complex features that are hard to be expressed in the docstring only.

### Fixtures and assert functions:
We provide [fixtures](https://docs.pytest.org/en/latest/fixture.html) to help your write 
your tests as well as helper functions. Those can be found in 
[test_utils.py](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/test_utils.py).

#### maybe_run_functions_eagerly

Will run your test function twice, once normally and once with 
`tf.config.run_functions_eagerly(True)`. To use it:

```python
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_something():
    assert ...== ...
```

##### When to use it?

You should use it only if you are using `tf.function` and running some control flow
on Tensors, `if` or `for` for example. Or with `TensorArray`. In short, when the 
 conversion to graph is not trivial. No need to use it on all
your tests. Having fast tests is important.

#### Selecting the devices to run the test

By default, each test is wrapped behind the scenes with a 
```python
with tf.device("CPU:0"):
    ...
```

This is automatic. But it's also possible to ask the test runner to run 
the test twice, on CPU and on GPU, or only on GPU. Here is how to do it.

```python
import pytest
import tensorflow as tf
from tensorflow_addons.utils import test_utils

@pytest.mark.with_device(["cpu", "gpu"])
def test_something():
    # the code here will run twice, once on gpu, once on cpu.
    ...


@pytest.mark.with_device(["cpu", "gpu"])
def test_something2(device):
    # the code here will run twice, once on gpu, once on cpu.
    # device will be "cpu:0" or "gpu:0" or "gpu:1" or "gpu:2" ...   
    if "cpu" in device:
        print("do something.")
    if "gpu" in device:
        print("do something else.")



@pytest.mark.with_device(["cpu", "gpu", tf.distribute.MirroredStrategy])
def test_something3(device):
    # the code here will run three times, once on gpu, once on cpu and once with 
    # a mirror distributed strategy.
    # device will be "cpu:0" or "gpu:0" or the strategy.
    # with the MirroredStrategy, it's equivalent to:
    # strategy = tf.distribute.MirroredStrategy(...)
    # with strategy.scope():
    #     test_function(strategy)
    if "cpu" in device:
        print("do something.")
    if "gpu" in device:
        print("do something else.")
    if isinstance(device, tf.distribute.Strategy):
        device.run(...)


@pytest.mark.with_device(["gpu"])
def test_something_else():
    # This test will be only run on gpu.
    # The test runner will call with tf.device("GPU:0") behind the scenes.  
    ...

@pytest.mark.with_device(["cpu"])
def test_something_more():
    # Don't do that, this is the default behavior. 
    ...


@pytest.mark.with_device(["no_device"])
@pytest.mark.needs_gpu
def test_something_more2():
    # When running the function, there will be no `with tf.device` wrapper.
    # You are free to do whatever you wish with the devices in there.
    # Make sure to use only the cpu, or only gpus available to the current process with
    # test_utils.gpu_for_testing() , otherwise, it might not play nice with 
    # pytest's multiprocessing.
    # If you use a gpu, mark the test with @pytest.mark.needs_gpu , otherwise the 
    # test will fail if no gpu is available on the system.
    # for example
    ...
    strategy = tf.distribute.MirroredStrategy(test_utils.gpus_for_testing())
    with strategy.scope():
        print("I'm doing whatever I want.") 
    ...
```

Note that if a gpu is not detected on the system, the test will be 
skipped and not marked as failed. Only the first gpu of the system is used,
even when running pytest in multiprocessing mode. (`-n` argument). 
Beware of the out of cuda memory errors if the number of pytest workers is too high.

##### When to use it?

When you test custom CUDA code or float16 ops.
We can expect other existing TensorFlow ops to behave the same on CPU and GPU.

#### data_format

Will run your test function twice, once with `data_format` being `channels_first` and 
once with `data_format` being `channels_last`. To use it:

```python
def test_something(data_format):
    assert my_function_to_test(..., data_format=data_format) == ...
```

##### When to use it?

When your function has a `data_format` argument. You'll want to make sure your 
function behaves correctly with both data format.


#### assert_allclose_according_to_type

Is the same as [tf.test.TestCase.assertAllCloseAccordingToType](https://www.tensorflow.org/api_docs/python/tf/test/TestCase#assertAllCloseAccordingToType)
but doesn't require any subclassing to be done. Can be used as a plain function. To use it:

```python
from tensorflow_addons.utils import test_utils

def test_something():
    expected = ...
    computed = my_function_i_just_wrote(...).numpy()
    test_utils.assert_allclose_according_to_type(computed, expected)
```

##### When to use it?

When you want to test your function with multiple dtypes. Different dtypes requires 
different tolerances when comparing values.


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
