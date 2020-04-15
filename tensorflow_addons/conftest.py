from tensorflow_addons.utils.test_utils import maybe_run_functions_eagerly  # noqa: F401
from tensorflow_addons.utils.test_utils import cpu_and_gpu  # noqa: F401
from tensorflow_addons.utils.test_utils import data_format  # noqa: F401
from tensorflow_addons.utils.test_utils import set_seeds  # noqa: F401

# fixtures present in this file will be available
# when running tests and can be referenced with strings
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions

from collections import defaultdict


def get_test_reports(terminalreporter):
    dlist = []
    for replist in terminalreporter.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    return dlist


def report_file_durations(terminalreporter):
    durations = 17
    dlist = get_test_reports(terminalreporter)
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
        terminalreporter.write_sep("=", "slowest file durations")
    else:
        terminalreporter.write_sep("=", "slowest %s file durations" % durations)
        dlist = dlist[:durations]

    for filename, test_time in dlist:
        terminalreporter.write_line("{:02.2f}s {}".format(test_time, filename))


def report_funtions_durations(terminalreporter):
    durations = 17
    dlist = get_test_reports(terminalreporter)
    if not dlist:
        return

    # group by file
    durations_by_file = defaultdict(float)
    for test_report in dlist:
        if "[" in test_report.nodeid:
            file_and_function = test_report.nodeid[:test_report.nodeid.index("[")]
        else:
            file_and_function = test_report.nodeid
        durations_by_file[file_and_function] += test_report.duration

    dlist = list(durations_by_file.items())

    dlist.sort(key=lambda x: x[1])
    dlist.reverse()
    if not durations:
        terminalreporter.write_sep("=", "slowest test functions durations")
    else:
        terminalreporter.write_sep("=", "slowest %s test functions" % durations)
        dlist = dlist[:durations]

    for filename, test_time in dlist:
        terminalreporter.write_line("{:02.2f}s {}".format(test_time, filename))


def report_sum_durations(terminalreporter):
    """Print the sum of durations of all the tests."""
    dlist = get_test_reports(terminalreporter)
    if not dlist:
        return

    terminalreporter.write_sep("=", "Sum of all tests durations")
    terminalreporter.write_line("{:02.2f}s".format(sum(x.duration for x in dlist)))

def pytest_terminal_summary(terminalreporter):
    report_file_durations(terminalreporter)
    report_funtions_durations(terminalreporter)
    report_sum_durations(terminalreporter)