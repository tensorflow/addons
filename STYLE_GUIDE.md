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

Addon use [Black](https://black.readthedocs.io/en/stable/) to format our code.
The continuous integration check will fail if you do not use it.


It's very useful to have a pre-commit hook that will run them both before 
each of your commits. To do that:
```bash
pip install pre-commit
pre-commit install
```

When making a commit, black and flake8 will run.

* If Black makes the pre-commit fail, you'll just have to run `git add` and `git commit`
again (black makes the hook fail and modify your files, you need to redo the commit).

* If flake8 makes the pre-commit fail, you need to fix it yourself before committing again.

#### TensorFlow Conventions

Follow the guidance in the [TensorFlow Style Guide - Conventions](https://www.tensorflow.org/community/contribute/code_style#tensorflow_conventions_and_special_uses).

Please note that Addons follows the conventions of the TensorFlow library, but formats our code using [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines.
