from tensorflow_addons.utils.test_utils import maybe_run_functions_eagerly  # noqa: F401
from tensorflow_addons.utils.test_utils import cpu_and_gpu  # noqa: F401
from tensorflow_addons.utils.test_utils import data_format  # noqa: F401
from tensorflow_addons.utils.test_utils import set_seeds  # noqa: F401

# fixtures present in this file will be available
# when running tests and can be referenced with strings
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions

from collections import defaultdict


def pytest_terminal_summary(terminalreporter):
    durations = 17
    tr = terminalreporter
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if not dlist:
        return

    # group by file
    durations_by_file = defaultdict(float)
    for test_report in dlist:
        durations_by_file[test_report.fspath] += test_report.duration

    dlist = list(durations_by_file.items())

    dlist.sort(key=lambda x: x[1])
    dlist.reverse()
    if not durations:
        tr.write_sep("=", "slowest file durations")
    else:
        tr.write_sep("=", "slowest %s file durations" % durations)
        dlist = dlist[:durations]

    for filename, test_time in dlist:
        tr.write_line("{:02.2f}s {:<8} {}".format(test_time, " ", filename))
