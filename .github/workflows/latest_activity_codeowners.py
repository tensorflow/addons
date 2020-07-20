import datetime
from github import Github
import pickle
import re
import requests


def events_github_api(g, owners):
    excluded_events = ["mentioned", "subscribed", "referenced"]
    repo = g.get_repo("tensorflow/addons")
    events = repo.get_issues_events().reversed
    users_stats = dict.fromkeys(owners, datetime.datetime.min)
    for event in events:
        if event.event not in excluded_events:
            if hasattr(event.issue, "user"):
                login = event.issue.user.login.lower()
            elif hasattr(event.pull_request, "user"):
                login = event.pull_resquest.user.login.lower()
            elif hasattr(event.actor, "login"):
                login = event.actor.login.lower()
            else:
                continue
            if login in owners:
                print(
                    "User: %s Event date: %s Event type: %s"
                    % (login, event.created_at, event.event)
                )
                users_stats[login] = event.created_at
    users_stats = {
        k: v for k, v in sorted(users_stats.items(), key=lambda item: item[1])
    }
    return users_stats


def get_owners():
    result = requests.get(
        "https://raw.githubusercontent.com/tensorflow/addons/master/.github/CODEOWNERS"
    )
    owners = re.findall(r"@(\S+)", result.text)
    owners = [owner.lower() for owner in owners]
    owners = sorted(set(owners))
    return owners


def main():
    owners = get_owners()
    print("=" * 80)
    print("OWNERS")
    print("=" * 80)
    for owner in owners:
        print(owner)

    # First create a Github instance:

    # using username and password
    # g = Github("user", "password")

    # or using an access token
    g = Github("your_access_token")

    users_stats = {}

    # Use cached pickle or retrive from Github API
    from_file = False
    # I really don't know if we want to serialize something
    # TODO if we maintain this we could use secure tmp files
    pickle_file = "/tmp/tfa_issues_events.pickle"
    if not from_file:
        users_stats = events_github_api(g,owners)
        with open(pickle_file, "wb") as fp:
            pickle.dump(users_stats, fp)
    else:
        with open(pickle_file, "rb") as fp:
            users_stats = pickle.load(fp)
    print("=" * 80)
    print("Owners latest activity in TFA repo:")
    print("=" * 80)
    for user, last_event in users_stats.items():
        print("User: %s Event date: %s" % (user, last_event))


if __name__ == "__main__":
    main()
