import yaml
from pathlib import Path


def get_yaml_links():
    with open("docs/tutorials/_toc.yaml", "r") as stream:
        toc = yaml.safe_load(stream)["toc"]

    links = [x["path"].split("/")[-1] for x in toc if "path" in x]
    links.remove("overview")

    links_set = set(links)
    assert len(links) == len(
        links_set
    ), "There are duplicate links in the table of contents"

    return links_set


def get_tutorial_notebooks():
    tutorial_dir = Path("docs/tutorials")
    notebooks = [x.stem for x in tutorial_dir.glob("*.ipynb")]
    notebooks.remove("_template")

    notebook_set = set(notebooks)

    assert len(notebooks) == len(notebook_set), "There are duplicate notebook filenames"

    return notebook_set


def verify_table_of_contents():

    links = get_yaml_links()
    tutorials = get_tutorial_notebooks()
    difference = links ^ tutorials

    assert (
        len(difference) == 0
    ), "{0} tutorials were not found as ipython notebooks and table of contents links".format(
        difference
    )


if __name__ == "__main__":
    verify_table_of_contents()
