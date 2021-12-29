import github
import click

from pathlib import Path
from typing import List, Tuple
import re
import fnmatch
import glob
import os
import json

# Github already take care
# of notifying users with write access
WRITE_ACCESS_LIST = [
    "tensorflow/sig-addons-maintainers",
    "facaiy",
    "seanpmorgan",
    "squadrick",
    "shun-lin",
    "windqaq",
    "qlzh727",
    "guillaumekln",
]


def get_github_client():
    return github.Github(os.environ["BOT_TOKEN"])


CLIENT = get_github_client()

# faster checks
valid_users_cache = set()


def check_user(user: str, line_idx: int):
    print(f"Checking that {user} actually exists")
    if user[0] != "@":
        raise ValueError(
            f"User '{user}' at line {line_idx} of CODEOWNERS "
            f"doesn't start with '@' "
        )
    user = user[1:]
    user = user.lower()  # in github, user names are case insensitive
    if user in WRITE_ACCESS_LIST:
        return None
    try:
        if user in valid_users_cache:
            return user
        CLIENT.get_user(user)
        valid_users_cache.add(user)
    except github.UnknownObjectException:
        raise KeyError(
            f"User '{user}' line {line_idx} does not exist. Did you make a typo?"
        )
    return user


def check_pattern(pattern: str, line_idx: int):
    if pattern[0] == "/":
        pattern = pattern[1:]

    pattern = Pattern(pattern)

    if not pattern.match_in_dir("."):
        raise FileNotFoundError(
            f"'{pattern.string}' present in CODEOWNERS line"
            f" {line_idx} does not match any file in the repository. "
            f"Did you make a typo?"
        )

    return pattern


class Pattern:
    def __init__(self, pattern: str):
        self.string = pattern
        if "*" in self.string:
            self.regex = re.compile(fnmatch.translate(pattern))
        else:
            self.regex = None

    def match(self, file):
        if self.regex:
            return re.match(self.regex, file) is not None
        else:
            return file.startswith(self.string)

    def match_in_dir(self, directory):
        list_files = glob.glob(f"{directory}/**/*", recursive=True)
        list_files = [os.path.relpath(x, directory) for x in list_files]
        matching_files = list(filter(self.match, list_files))
        return bool(matching_files)


CodeOwners = List[Tuple[Pattern, List[str]]]


def parse_codeowners(text: str) -> CodeOwners:
    result = []

    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if line == "":
            continue
        if line[0] == "#":  # comment
            continue
        elements = list(filter(lambda x: x != "", line.split(" ")))

        pattern = check_pattern(elements[0], i)
        users = [check_user(user, i) for user in elements[1:]]
        users = [user for user in users if user is not None]
        if users:
            result.append((pattern, users))

    return result


nice_message = """

You are owner{} of some files modified in this pull request.
Would you kindly review the changes whenever you have the time to?
Thank you very much.
"""


def craft_message(codeowners: CodeOwners, pull_request):

    owners = set()
    for file in pull_request.get_files():
        for pattern, users in codeowners:
            if not pattern.match(file.filename):
                continue
            owners.update(users)

    author = pull_request.user.login.lower()
    try:
        owners.remove(author)  # no need to notify the author
    except KeyError:
        pass

    owners = [f"@{owner}" for owner in owners]
    if not owners:
        return None
    if len(owners) >= 2:
        plural = "s"
    else:
        plural = ""
    return " ".join(owners + [nice_message.format(plural)])


def get_pull_request_id_from_gh_actions():
    actions_file = Path(os.environ["GITHUB_EVENT_PATH"])
    return json.loads(actions_file.read_text())["number"]


@click.command()
@click.option("--pull-request-id")
@click.option("--no-dry-run", is_flag=True)
def notify_codeowners(pull_request_id, no_dry_run):
    text = (Path(__file__).parents[1] / "CODEOWNERS").read_text()
    codeowners = parse_codeowners(text)

    if pull_request_id is not None:
        if pull_request_id == "auto":
            pull_request_id = get_pull_request_id_from_gh_actions()
        pull_request_id = int(pull_request_id)
        pull_request = CLIENT.get_repo("tensorflow/addons").get_pull(pull_request_id)
        msg = craft_message(codeowners, pull_request)
        print(msg)
        if no_dry_run and msg is not None:
            pull_request.create_issue_comment(msg)


if __name__ == "__main__":
    notify_codeowners()
