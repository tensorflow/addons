import sys
import pkg_resources
from pip._internal.req import parse_requirements
from pkg_resources import DistributionNotFound, VersionConflict


def check_dependencies(requirement_file_name):
    """Checks to see if the python dependencies are fullfilled.

    If check passes return 0. Otherwise print error and return 1
    """
    dependencies = []
    for req in parse_requirements(requirement_file_name, session=False):
        dependencies.append(str(req.req))
    try:
        pkg_resources.working_set.require(dependencies)
    except VersionConflict as e:
        try:
            print("{} was found on your system, "
                  "but {} is required for this build.\n".format(e.dist, e.req))
            sys.exit(1)
        except AttributeError:
            sys.exit(1)
    except DistributionNotFound as e:
        print(e)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    check_dependencies('requirements.txt')
