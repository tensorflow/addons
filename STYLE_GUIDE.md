#### C++
C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Addons uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
to check your C/C++ changes. Sometimes you have some manually formatted
code that you don’t want clang-format to touch.
You can disable formatting like this:

```cpp
int formatted_code;
// clang-format off
    void    unformatted_code  ;
// clang-format on
void formatted_code_again;
```

Install Clang-format 9 with:

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository -u 'http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main'
sudo apt install clang-format-9
```

format all with:
```bash
clang-format-9 -i --style=google **/*.cc **/*.h
```

#### Python

Addons uses [flake8](http://flake8.pycqa.org/en/latest/) to check pep8 compliance and 
code analysis.

Addons use [Black](https://black.readthedocs.io/en/stable/) to format our code.
The continuous integration check will fail if you do not use it.

Install them with:
```
pip install flake8 black
```

Be sure to run them both before you push your commits, otherwise the CI will fail!

```
python -m black ./
python -m flake8
```

#### TensorFlow Conventions

Follow the guidance in the [TensorFlow Style Guide - Conventions](https://www.tensorflow.org/community/contribute/code_style#tensorflow_conventions_and_special_uses).

Please note that Addons follows the conventions of the TensorFlow library, but formats our code using [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines.
