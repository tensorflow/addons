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
Python code should conform to [PEP8](https://www.python.org/dev/peps/pep-0008/).

Addons uses [yapf](https://github.com/google/yapf) to format code,
and [pylint](https://www.pylint.org/) for code analysis.
You can disable them locally like this:

```python
# yapf: disable
FOO = {
    # ... some very large, complex data literal.
}

BAR = [
    # ... another large data literal.
]
# yapf: enable
```

```python
# pylint: disable=protected-access
foo._protected_member
# pylint: enable=protected-access
```


##### Python Special Cases Conventions

Follow the guidance in the [TensorFlow Style Guide - Conventions sections](https://www.tensorflow.org/community/contribute/code_style#tensorflow_conventions_and_special_uses).
