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

```
cd .git/hooks && ln -s -f ../../tools/pre-commit.sh pre-commit
```

and you're good to go.

On Windows, in powershell, do:

```
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

```
pip install virtualenv
venv my_dev_environement
source my_dev_environement/bin/activate  # Linux/macos/WSL2
.\my_dev_environement\Scripts\activate   # PowerShell
```

If you want to work in 
a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
```
conda create --name my_dev_environement
conda activate my_dev_environement
```


#### Install TensorFlow Addons in editable mode

Just run from the root:

```
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

```
pip uninstall tensorflow-addons
```

#### Run the tests with pytest

If TensorFlow Addons is installed in editable mode, you can then just run your tests by 
running Pytest. For example:
```bash
pip install -r tools/install_deps/pytest.txt
python -m pytest tensorflow_addons/rnn/cell_test.py
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
python configure.py --no-deps   # if you don't want any dependencies installed with pip
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
```
docker run --rm -it -v ${PWD}:/addons -w /addons tensorflow/tensorflow:2.1.0-custom-op-ubuntu16
```

GPU Docker: 
```
docker run --runtime=nvidia --rm -it -v ${PWD}:/addons -w /addons tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16
```

Configure:
```
python3 ./configure.py  # Links project with TensorFlow dependency
```

Install in editable mode
```
python3 -m pip install -e .
python3 -m pip install pytest pytest-xdist
```

Compile the custom ops
```
bash tools/install_so_files.sh
```

Run selected tests:
```bash
python3 -m pytest path/to/file/or/directory/to/test
```

#### Testing with Bazel

Testing with Bazel is still supported but not recommended unless you have prior experience 
with Bazel, and would like to use it for specific capabilities (Remote execution, etc).
This is because pytest offers many more options to run your test suite and has
better error reports, timings reports, open-source plugins and documentation online 
for Python testing. 

Internally, Google can use Bazel to test many commits 
quickly, as Bazel has great support for caching and distributed testing.

To test with Bazel:

```
python configure.py
pip install pytest
bazel test -c opt -k \
--test_timeout 300,450,1200,3600 \
--test_output=all \
--run_under=$(readlink -f tools/testing/parallel_gpu_execute.sh) \
//tensorflow_addons/...
```

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
in [tools/testing/check_typing_info.py](tools/testing/check_typing_info.py).

Help is welcome to make this TODO list smaller!

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
