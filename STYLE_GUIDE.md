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

#### Python

Addons uses [flake8](http://flake8.pycqa.org/en/latest/) to check pep8 compliance and 
code analysis.

We use [Black]() to format our code.

Install both with:

```bash
pip install flake8 black
```

We recommend running them as pre-commit. For that, open (or create) the file
```
.git/hooks/pre-commit
```

and write inside:

```bash
python -m black ./
python -m flake8
```


#### TensorFlow Conventions

Follow the guidance in the [TensorFlow Style Guide - Conventions](https://www.tensorflow.org/community/contribute/code_style#tensorflow_conventions_and_special_uses).

Please note that Addons follows the conventions of the TensorFlow library, but formats our code using [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines.
