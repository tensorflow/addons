# TensorFlow Addons Examples

TensorFlow Addons welcomes and highly encourages example contributions.


## How To Contribute

Addons examples are created using [Google Colab](https://colab.research.google.com/) 
and the jupyter notebooks are saved to this directory in the repository. To do 
this... follow the below steps:

1. Create a docs branch on your fork of TensorFlow Addons
2. Goto [Google Colab](https://colab.research.google.com/) and start a new 
notebook using this template:
[https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb](https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb)
3. Remove the URL box titled "View on TensorFlow.org"
8. Edit the "View source on Github" and "Run in Google Colab" URL boxesso that 
they point to the where the example notebook will exist on the master branch 
after the PR is merged.
4. Install the correct TF versions at the top of the colab session:
    ```
    !pip install tensorflow==2.0.0.a0
    !pip install tensorflow-addons
    ```
5. Follow the guidelines of the template
6. "Save a copy in Github" and select your new branch. The notebook should be 
named `subpackage_submodule`
7. Submit the branch as a PR on the TF-Addons Github
