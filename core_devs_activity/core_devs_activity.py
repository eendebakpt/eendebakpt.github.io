#!/usr/bin/env python3
"""Plot CPython core-dev counts over time, with metrics:

  1. Listed core devs  — from the devguide team-log page (Name / GitHub login /
                          Joined / Left columns; historical members back to
                          1989-12-25, Guido).
                          Cross-checked against `python/voters/python-core.toml`
                          git log (current members only; voters started ~2020).
  2. Active core devs  — of those listed at a given month, how many had >=1 PR
                          or issue authored in `python/cpython` within the
                          trailing activity window (default: two years).
  3. Active contributors — distinct accounts (any author) with >=1 PR/issue
                          opened in python/cpython within the same window.
                          Active core devs are a subset of this.

Comments are not counted: GitHub's search/GraphQL API has no per-user-per-repo
comment query.

USAGE:
    GITHUB_TOKEN=ghp_xxx python3 core_devs_activity.py

The script runs four stages in order, each cached to disk so re-runs are
cheap:

    1. fetch-roster       ->  data/roster.json  (per-month: set of active logins)
    2. fetch-activity     ->  data/activity/<login>.json
    3. fetch-repo-activity->  data/all_repo_activity.json (all PRs/issues)
    4. plot               ->  out/core_devs_over_time.png

Run with --only <stage> to do just one. Run with --refresh-activity to
re-fetch any login whose cache is older than --max-cache-age-days.

Requires: requests, matplotlib. Optional but strongly recommended:
GITHUB_TOKEN env var (raises GraphQL rate limit from 60/hr to 5000/hr).
"""

import argparse
import bisect
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
# matplotlib backend is chosen after argv parsing (see main()): Agg when
# --no-show / no display, otherwise whatever GUI backend the env provides.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


REPO = "python/cpython"
# Right-axis "total contributions" line — a dark green that stays legible dashed.
TOTAL_COLOR = "#0b6e34"
TEAM_LOG_URL = "https://devguide.python.org/core-team/team-log/index.html"
VOTERS_REPO = "python/voters"
VOTERS_TOML = "python-core.toml"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ACTIVITY_DIR = DATA_DIR / "activity"
OUT_DIR = SCRIPT_DIR / "out"
PYODIDE_DIR = SCRIPT_DIR / "pyodide_web"


def gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def gh_request(method, url, **kwargs):
    """GitHub REST with simple rate-limit handling."""
    for attempt in range(5):
        r = requests.request(method, url, headers=gh_headers(), timeout=30, **kwargs)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = int(r.headers.get("X-RateLimit-Reset", "0"))
            wait = max(1, reset - int(time.time()) + 2)
            print(f"  [rate-limit] sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code in (502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"failed after retries: {method} {url}")


# ---------------------------------------------------------------------------
# Stage 1: fetch core-dev roster from devguide team-log (primary source)
#          and cross-check against python/voters TOML (current members only).
# ---------------------------------------------------------------------------

from html.parser import HTMLParser


class TeamLogTableParser(HTMLParser):
    """Extract the rows of the team-log HTML table.

    Schema (as of 2026-05): Name | GitHub username | Joined | Left | Notes
    """

    def __init__(self):
        super().__init__()
        self._in_table = False
        self._in_cell = False
        self._cell_text: list[str] = []
        self._row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._row = []
        elif tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._cell_text = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._row.append("".join(self._cell_text).strip())
        elif tag == "tr" and self._in_table and self._row:
            self.rows.append(self._row)
            self._row = []
        elif tag == "table" and self._in_table:
            self._in_table = False

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text.append(data)


def fetch_team_log() -> list[dict]:
    """Return list of {name, login, joined, left} (left may be None)."""
    r = requests.get(TEAM_LOG_URL, timeout=30)
    r.raise_for_status()
    parser = TeamLogTableParser()
    parser.feed(r.text)
    if not parser.rows:
        raise RuntimeError(f"no table found at {TEAM_LOG_URL}")
    header = [c.lower() for c in parser.rows[0]]
    # Expect: ['name', 'github username', 'joined', 'left', 'notes']
    if "github username" not in header or "joined" not in header:
        raise RuntimeError(f"unexpected team-log header: {header}")
    name_i  = header.index("name")
    login_i = header.index("github username")
    joined_i = header.index("joined")
    left_i  = header.index("left") if "left" in header else None

    out = []
    for row in parser.rows[1:]:
        if len(row) <= max(name_i, login_i, joined_i,
                           left_i if left_i is not None else 0):
            continue
        name   = row[name_i].strip()
        login  = row[login_i].strip().lower()
        joined = row[joined_i].strip()
        left   = row[left_i].strip() if left_i is not None else ""
        if not name or not joined:
            continue
        try:
            dt.date.fromisoformat(joined)
        except ValueError:
            print(f"  skipping unparsable joined date for {name}: {joined!r}",
                  file=sys.stderr)
            continue
        if left:
            try:
                dt.date.fromisoformat(left)
            except ValueError:
                print(f"  skipping unparsable left date for {name}: {left!r}",
                      file=sys.stderr)
                left = ""
        out.append({
            "name": name,
            "login": login or None,   # None for pre-GitHub members who never made the transition
            "joined": joined,
            "left": left or None,
        })
    return out


def parse_python_core_toml(text: str) -> set[str]:
    """Return the set of GitHub logins in a python-core.toml snapshot."""
    logins = set()
    for line in text.splitlines():
        line = line.strip()
        for prefix in ('github = "', 'github="', 'mailmap = "', 'mailmap="'):
            if line.startswith(prefix):
                val = line[len(prefix):].split('"', 1)[0].strip()
                if val and "@" not in val:
                    logins.add(val.lower())
                break
    return logins


def fetch_current_voters() -> set[str] | None:
    """Best-effort fetch of the current python-core.toml. Used as a sanity
    cross-check against the team-log snapshot. Returns None on failure."""
    url = (f"https://raw.githubusercontent.com/{VOTERS_REPO}/main/{VOTERS_TOML}")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return parse_python_core_toml(r.text)
    except Exception as e:
        print(f"  [voters cross-check] skipped: {e}", file=sys.stderr)
        return None


def stage_fetch_roster(force: bool = False) -> Path:
    out = DATA_DIR / "roster.json"
    if out.exists() and not force:
        print(f"[roster] cached at {out}", file=sys.stderr)
        return out
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[roster] fetching {TEAM_LOG_URL} ...", file=sys.stderr)
    members = fetch_team_log()
    print(f"[roster] team-log: {len(members)} members "
          f"({sum(1 for m in members if m['left'] is None)} currently active)",
          file=sys.stderr)

    # Cross-check current set against voters TOML
    voters = fetch_current_voters()
    if voters is not None:
        current_team = {m["login"] for m in members if m["left"] is None}
        only_team_log = current_team - voters
        only_voters = voters - current_team
        print(f"[roster] voters TOML has {len(voters)} entries", file=sys.stderr)
        if only_team_log:
            print(f"  in team-log but not voters: {sorted(only_team_log)}",
                  file=sys.stderr)
        if only_voters:
            print(f"  in voters but not team-log: {sorted(only_voters)}",
                  file=sys.stderr)

    payload = {
        "source": TEAM_LOG_URL,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "members": members,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[roster] saved -> {out}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Stage 2: per-login activity on python/cpython
# ---------------------------------------------------------------------------

GRAPHQL_QUERY = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      __typename
      ... on PullRequest { createdAt author { login } }
      ... on Issue       { createdAt author { login } }
    }
  }
  rateLimit { remaining resetAt }
}
"""

# GitHub's search API (REST and GraphQL alike) hard-caps result *retrieval* at
# 1000 items per query — paging past that errors out. `issueCount`, however,
# reports the true total. So to collect every match we recursively split the
# date range until each sub-range has <=1000 results, then page through it.
GH_SEARCH_CAP = 1000
# Earliest item the GitHub search returns for python/cpython is 2000-06-06:
# bpo issues migrated to GitHub in 2022 kept their original (backdated) creation
# dates, so the searchable history reaches back well before the Feb-2017 repo
# migration. Start a little earlier than that and let the range-splitter skip
# the empty leading years cheaply (one issueCount probe each).
GH_ERA_START = dt.date(2000, 1, 1)


def gh_graphql(query: str, variables: dict) -> dict:
    for attempt in range(5):
        r = requests.post(
            "https://api.github.com/graphql",
            headers=gh_headers(),
            json={"query": query, "variables": variables},
            timeout=60,
        )
        if r.status_code in (502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 401:
            # A persistent 401 means a missing/bad token, but a long run can hit
            # a transient one — retry a couple times before giving up so a single
            # blip doesn't discard minutes of fetching.
            if attempt < 3:
                print("  [graphql] 401 — retrying", file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            raise SystemExit("GraphQL: unauthorized — set GITHUB_TOKEN")
        if r.status_code == 200:
            data = r.json()
            if "errors" in data:
                # Common: secondary rate limit -> sleep and retry
                msg = json.dumps(data["errors"])
                if "rate limit" in msg.lower() or "abuse" in msg.lower():
                    print("  [graphql rate-limit] sleeping 60s", file=sys.stderr)
                    time.sleep(60)
                    continue
                raise RuntimeError(f"GraphQL error: {msg}")
            return data["data"]
        time.sleep(2 ** attempt)
    raise RuntimeError("GraphQL: failed after retries")


def _log_rate_limit(rl: dict) -> None:
    if rl.get("remaining", 100) < 50:
        print(f"  [graphql] remaining={rl['remaining']} reset={rl.get('resetAt')}",
              file=sys.stderr)


def iter_search_nodes(base_query: str,
                      start: dt.date = GH_ERA_START,
                      end: dt.date | None = None,
                      progress: bool = False):
    """Yield every issue/PR node matching `base_query` across [start, end].

    Works around GitHub search's 1000-result cap by recursively halving any
    date range whose `issueCount` exceeds the cap until each sub-range fits,
    then paging that sub-range to completion.
    """
    if end is None:
        end = dt.date.today()
    seen = 0
    # LIFO stack of date ranges still to process; we push the later half first
    # so ranges are visited oldest-first (nicer progress output).
    stack: list[tuple[dt.date, dt.date]] = [(start, end)]
    while stack:
        lo, hi = stack.pop()
        q = f"{base_query} created:{lo.isoformat()}..{hi.isoformat()}"
        # The first page doubles as the count probe — no wasted request.
        data = gh_graphql(GRAPHQL_QUERY, {"q": q, "cursor": None})
        search = data["search"]
        count = search["issueCount"]

        if count > GH_SEARCH_CAP and lo < hi:
            mid = lo + (hi - lo) // 2
            # push later half first -> earlier half processed next (oldest-first)
            stack.append((mid + dt.timedelta(days=1), hi))
            stack.append((lo, mid))
            continue
        if count > GH_SEARCH_CAP:
            # Single day with >1000 items: unreachable for this repo in
            # practice, but flag it rather than silently truncating.
            print(f"  [search] WARNING: {q} has {count} > {GH_SEARCH_CAP} "
                  f"results on a single day; collecting only the first "
                  f"{GH_SEARCH_CAP}.", file=sys.stderr)

        page = search
        while True:
            for node in page["nodes"]:
                yield node
                seen += 1
            if not page["pageInfo"]["hasNextPage"]:
                break
            data = gh_graphql(GRAPHQL_QUERY,
                              {"q": q, "cursor": page["pageInfo"]["endCursor"]})
            _log_rate_limit(data.get("rateLimit", {}))
            page = data["search"]
        if progress and count:
            print(f"  [search] {lo.isoformat()}..{hi.isoformat()}: "
                  f"{count} (total so far: {seen})", file=sys.stderr)
        _log_rate_limit(data.get("rateLimit", {}))


def fetch_search_dates(base_query: str) -> list[str]:
    """Sorted createdAt day (YYYY-MM-DD) of every issue/PR matching the query."""
    return sorted(n["createdAt"][:10] for n in iter_search_nodes(base_query)
                  if n.get("createdAt"))


def fetch_user_activity(login: str) -> list[str]:
    """Return list of ISO date strings — one per PR/issue created by `login`
    in python/cpython (across all time)."""
    return fetch_search_dates(f"author:{login} repo:{REPO}")


def fetch_repo_contributors() -> dict[str, list[str]]:
    """Map each GitHub login -> sorted list of distinct YYYY-MM-DD days on
    which they opened a PR or issue in python/cpython (across all time).

    Per-author days (not raw events) are what the "active contributors"
    metric needs: a contributor counts once per window regardless of volume.
    Uses adaptive date-range splitting to get past the 1000-result cap."""
    by_login: dict[str, set] = {}
    for node in iter_search_nodes(f"repo:{REPO}", progress=True):
        ca = node.get("createdAt")
        author = node.get("author")  # null for deleted/ghost accounts
        login = author.get("login") if author else None
        if ca and login:
            by_login.setdefault(login.lower(), set()).add(ca[:10])
    return {lg: sorted(days) for lg, days in sorted(by_login.items())}


def stage_fetch_activity(roster_path: Path, max_cache_age_days: int = 7,
                         refresh: bool = False) -> Path:
    ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
    with open(roster_path) as f:
        roster = json.load(f)
    # Pre-GitHub members (login is None) are never queryable. Skip them.
    all_logins = sorted({m["login"] for m in roster["members"] if m["login"]})
    n_no_login = sum(1 for m in roster["members"] if not m["login"])
    print(f"[activity] {len(all_logins)} unique logins to fetch "
          f"({n_no_login} historical members have no GitHub login — skipped)",
          file=sys.stderr)

    now = dt.datetime.now(dt.timezone.utc)
    fetched = skipped = 0
    for i, login in enumerate(all_logins, 1):
        cache = ACTIVITY_DIR / f"{login}.json"
        if cache.exists() and not refresh:
            age = now - dt.datetime.fromtimestamp(cache.stat().st_mtime, dt.timezone.utc)
            if age.days <= max_cache_age_days:
                skipped += 1
                continue
        try:
            dates = fetch_user_activity(login)
        except Exception as e:
            print(f"  [{i}/{len(all_logins)}] {login}: error {e}", file=sys.stderr)
            continue
        with open(cache, "w") as f:
            json.dump({"login": login, "dates": dates,
                       "fetched_at": now.isoformat()}, f)
        fetched += 1
        print(f"  [{i}/{len(all_logins)}] {login}: {len(dates)} contributions",
              file=sys.stderr)
    print(f"[activity] done. fetched={fetched} skipped={skipped}", file=sys.stderr)
    return ACTIVITY_DIR


def stage_fetch_repo_activity(force: bool = False) -> Path:
    """Fetch per-author PR/issue activity days for python/cpython (any author)."""
    out = DATA_DIR / "all_repo_activity.json"
    if out.exists() and not force:
        print(f"[repo-activity] cached at {out}", file=sys.stderr)
        return out

    print(f"[repo-activity] fetching all PR/issue authors from {REPO} ...",
          file=sys.stderr)
    contributors = fetch_repo_contributors()
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "repo": REPO,
        "fetched_at": now.isoformat(),
        "note": "Per-login list of YYYY-MM-DD days with >=1 PR/issue opened in the repo",
        "contributors": contributors,
    }
    with open(out, "w") as f:
        json.dump(payload, f)
    n_days = sum(len(v) for v in contributors.values())
    print(f"[repo-activity] saved -> {out} "
          f"({len(contributors)} contributors, {n_days} active-days)",
          file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Stage 3: aggregate + plot
# ---------------------------------------------------------------------------

def month_range(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = dt.date(start.year, start.month, 1)
    end = dt.date(end.year, end.month, 1)
    while cur <= end:
        yield cur
        # next month
        y, m = cur.year + (cur.month // 12), cur.month % 12 + 1
        cur = dt.date(y, m, 1)


def roster_at(members: list[dict], when: dt.date) -> list[dict]:
    """Return the list of members on the core team at `when`, i.e.
    joined<=when and (left is None or left>when). Includes pre-GitHub
    members whose `login` is None."""
    out = []
    for m in members:
        joined = dt.date.fromisoformat(m["joined"])
        if joined > when:
            continue
        if m["left"]:
            left = dt.date.fromisoformat(m["left"])
            if left <= when:
                continue
        out.append(m)
    return out


def load_activity_dates(login: str) -> list[dt.date]:
    p = ACTIVITY_DIR / f"{login}.json"
    if not p.exists():
        return []
    with open(p) as f:
        return [dt.date.fromisoformat(d) for d in json.load(f)["dates"]]


def active_contributors_over_months(contributors: dict, months: list,
                                    window_days: int) -> list[int]:
    """For each month m, the number of distinct logins with >=1 active day in
    the window [m - window_days, m]. One sweep with two pointers over events
    sorted by day, keeping a per-login in-window day-count — O(events)."""
    events = sorted((day, lg)
                    for lg, days in contributors.items() for day in days)
    n = len(events)
    counts: dict[str, int] = {}
    active = lo = hi = 0
    series = []
    for m in months:
        m_iso = m.isoformat()
        win_start = (m - dt.timedelta(days=window_days)).isoformat()
        while hi < n and events[hi][0] <= m_iso:          # entering the window
            lg = events[hi][1]
            counts[lg] = counts.get(lg, 0) + 1
            if counts[lg] == 1:
                active += 1
            hi += 1
        while lo < hi and events[lo][0] < win_start:       # leaving the window
            lg = events[lo][1]
            counts[lg] -= 1
            if counts[lg] == 0:
                active -= 1
            lo += 1
        series.append(active)
    return series


def stage_plot(roster_path: Path, activity_window_days: int = 365,
               start_year: int | None = None, show: bool = False):
    with open(roster_path) as f:
        roster = json.load(f)
    members = roster["members"]
    if not members:
        raise SystemExit("empty roster")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    first_joined = min(dt.date.fromisoformat(m["joined"]) for m in members)
    if start_year is not None:
        first_joined = max(first_joined, dt.date(start_year, 1, 1))
    # Plot only complete months: the current month is still in progress, so its
    # partial data would understate every series. End at the last full month
    # (the day before the 1st of the current month).
    today = dt.date.today()
    last = dt.date(today.year, today.month, 1) - dt.timedelta(days=1)
    months = list(month_range(first_joined, last))

    all_logins = sorted({m["login"] for m in members if m["login"]})
    activity = {lg: load_activity_dates(lg) for lg in all_logins}

    # Load per-author repo activity (any author) for the "active contributors"
    # series: distinct logins with >=1 PR/issue in the trailing window.
    repo_activity_path = DATA_DIR / "all_repo_activity.json"
    if repo_activity_path.exists():
        with open(repo_activity_path) as f:
            contributors = json.load(f).get("contributors", {})
    else:
        contributors = {}
    active_contrib_series = (
        active_contributors_over_months(contributors, months, activity_window_days)
        if contributors else [])

    listed_series = []
    active_series = []
    no_login_in_team_series = []

    for m in months:
        listed = roster_at(members, m)
        window_start = m - dt.timedelta(days=activity_window_days)
        active = sum(
            1 for mem in listed
            if mem["login"] and any(window_start <= d <= m
                                    for d in activity.get(mem["login"], []))
        )
        listed_series.append(len(listed))
        active_series.append(active)
        no_login_in_team_series.append(sum(1 for mem in listed if not mem["login"]))

    # num=str pins the figure window: re-running the script reuses the same
    # window instead of stacking a new one. clear=True wipes prior axes.
    fig, ax = plt.subplots(num="core_devs_over_time", figsize=(11, 6), clear=True)
    ax.plot(months, listed_series,
            label="Listed core team (devguide team-log)",
            linewidth=2)
    ax.plot(months, active_series,
            label=f"Active (>=1 PR/issue in past {activity_window_days} days on {REPO})",
            linewidth=2)
    ax.fill_between(months, active_series, listed_series, alpha=0.15,
                    label="Listed but not active in window")
    # Active contributors (whole community) are ~1-2 orders of magnitude larger
    # than the core team, so they get their own right-hand axis — on a shared
    # axis they'd flatten the team lines onto the x-axis. (Active core devs are
    # a subset of this line.)
    if active_contrib_series:
        ax_t = ax.twinx()
        ax_t.plot(months, active_contrib_series, color=TOTAL_COLOR,
                  label=f"Active contributors (any author, past {activity_window_days} days)",
                  linewidth=2, linestyle="--", alpha=0.9)
        ax_t.set_ylabel("Active contributors (any author)", color=TOTAL_COLOR)
        ax_t.tick_params(axis="y", labelcolor=TOTAL_COLOR)
        ax_t.set_ylim(bottom=0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Core team members (count)")
    ax.set_ylim(bottom=0)
    ax.set_title("CPython core developer team over time\n"
                 "(sources: github.com/python/cpython and https://devguide.python.org/core-team/team-log/)")
    ax.grid(alpha=0.3)
    # Merge legends from both axes into one box.
    handles, labels = ax.get_legend_handles_labels()
    if active_contrib_series:
        h2, l2 = ax_t.get_legend_handles_labels()
        handles += h2; labels += l2
    ax.legend(handles, labels, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5 if (last.year-first_joined.year)>20 else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    max_no_login = max(no_login_in_team_series) if no_login_in_team_series else 0
    ax.text(0.01, -0.18,
            f"Roster: devguide team-log ({TEAM_LOG_URL.replace('https://', '')})\n"
            f"Activity: GitHub search API (PR + issue authorship in {REPO}). "
            f"Comments not counted (no per-user-per-repo comment search).\n"
            f"Note: {max_no_login} listed members (peak) had no GitHub account "
            f"(pre-GitHub era); they count as listed but never as active.",
            transform=ax.transAxes, fontsize=8, color="grey")

    out = OUT_DIR / "core_devs_over_time.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}", file=sys.stderr)

    # Modern subset (post-2010, the GitHub-era) for a more useful view
    if first_joined < dt.date(2010, 1, 1):
        fig2, ax2 = plt.subplots(num="core_devs_since_2010",
                                 figsize=(11, 6), clear=True)
        mask = [m >= dt.date(2010, 1, 1) for m in months]
        months_m = [m for m, k in zip(months, mask) if k]
        listed_m = [v for v, k in zip(listed_series, mask) if k]
        active_m = [v for v, k in zip(active_series, mask) if k]
        contrib_m = [v for v, k in zip(active_contrib_series, mask) if k]
        ax2.plot(months_m, listed_m, label="Listed core team", linewidth=2)
        ax2.plot(months_m, active_m,
                 label=f"Active (>=1 PR/issue in past {activity_window_days}d)",
                 linewidth=2)
        ax2.fill_between(months_m, active_m, listed_m, alpha=0.15)
        ax2.set_xlabel("Date"); ax2.set_ylabel("Core team members (count)")
        ax2.set_ylim(bottom=0)
        if active_contrib_series:
            ax2_t = ax2.twinx()
            ax2_t.plot(months_m, contrib_m, color=TOTAL_COLOR,
                       label=f"Active contributors (any author, past {activity_window_days}d)",
                       linewidth=2, linestyle="--", alpha=0.9)
            ax2_t.set_ylabel("Active contributors (any author)", color=TOTAL_COLOR)
            ax2_t.tick_params(axis="y", labelcolor=TOTAL_COLOR)
            ax2_t.set_ylim(bottom=0)
        ax2.set_title("CPython core team since 2010 (GitHub era)")
        ax2.grid(alpha=0.3)
        h, l = ax2.get_legend_handles_labels()
        if active_contrib_series:
            h2, l2 = ax2_t.get_legend_handles_labels()
            h += h2; l += l2
        ax2.legend(h, l, loc="upper left")
        ax2.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig2.autofmt_xdate()
        out2 = OUT_DIR / "core_devs_since_2010.png"
        fig2.tight_layout()
        fig2.savefig(out2, dpi=130)
        print(f"[plot] wrote {out2}", file=sys.stderr)

    with open(OUT_DIR / "series.csv", "w") as f:
        f.write("month,listed,active,listed_no_github_account,active_contributors\n")
        contrib = active_contrib_series or [0] * len(months)
        for m, lst, act, nol, con in zip(months, listed_series, active_series,
                                    no_login_in_team_series, contrib):
            f.write(f"{m.isoformat()},{lst},{act},{nol},{con}\n")
    print("[plot] wrote series.csv", file=sys.stderr)

    if show:
        print("[plot] showing interactively (close windows to exit) ...",
              file=sys.stderr)
        plt.show()


# ---------------------------------------------------------------------------
# Stage 4 (optional): export pre-baked JSON for the static GitHub Pages site
# ---------------------------------------------------------------------------

def stage_export_web(roster_path: Path) -> Path:
    """Write pyodide_web/data.js — an inline script-tag bundle of both the
    roster and the per-login activity dates, plus total repo activity. 
    The page reads it via a plain <script src="data.js"> tag, so it works 
    from file:// as well as via HTTP."""
    PYODIDE_DIR.mkdir(parents=True, exist_ok=True)
    with open(roster_path) as f:
        roster = json.load(f)

    # Roster: keep only the columns the page actually uses.
    members_out = [
        {"name": m["name"], "login": m["login"],
         "joined": m["joined"], "left": m["left"]}
        for m in roster["members"]
    ]
    web_roster = {
        "source": roster.get("source", TEAM_LOG_URL),
        "fetched_at": roster.get("fetched_at"),
        "members": members_out,
    }

    # Activity: full ISO dates so the client uses the same day-precision
    # window as the desktop script. Deduplicate (devs often make several
    # contribs on the same day).
    activity_compact: dict[str, list[str]] = {}
    for m in roster["members"]:
        login = m.get("login")
        if not login:
            continue
        dates = load_activity_dates(login)
        if not dates:
            continue
        activity_compact[login] = sorted({d.isoformat() for d in dates})
    web_activity = {
        "fetched_at": roster.get("fetched_at"),
        "note": "Per-login list of YYYY-MM-DD days with >=1 PR/issue authored in python/cpython.",
        "logins": activity_compact,
    }

    # Per-author repo activity (any author) for the "active contributors"
    # series. Per-login days let the client recompute for any chosen window.
    repo_activity_path = DATA_DIR / "all_repo_activity.json"
    web_contributors = None
    if repo_activity_path.exists():
        with open(repo_activity_path) as f:
            repo_data = json.load(f)
        web_contributors = {
            "fetched_at": repo_data.get("fetched_at"),
            "note": "Per-login list of YYYY-MM-DD days with >=1 PR/issue opened in python/cpython (any author).",
            "logins": repo_data.get("contributors", {}),
        }

    out_bundle = PYODIDE_DIR / "data.js"
    with open(out_bundle, "w") as f:
        f.write("// Auto-generated by core_devs_activity.py --only export-web.\n")
        f.write("// Loaded via <script src=\"data.js\">. Script tags are not blocked\n")
        f.write("// by the same-origin policy, so the page works from file:// too.\n")
        f.write("window.__DATA__ = {roster: ")
        json.dump(web_roster, f, separators=(",", ":"))
        f.write(", activity: ")
        json.dump(web_activity, f, separators=(",", ":"))
        if web_contributors:
            f.write(", contributors: ")
            json.dump(web_contributors, f, separators=(",", ":"))
        f.write("};\n")
    size_kb = out_bundle.stat().st_size / 1024
    n_contrib = len(web_contributors["logins"]) if web_contributors else 0
    print(f"[web] wrote {out_bundle} ({size_kb:.1f} KB, "
          f"{len(members_out)} members / {len(activity_compact)} logins / "
          f"{n_contrib} contributors)", file=sys.stderr)
    return PYODIDE_DIR


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["roster", "activity", "repo-activity", "plot", "export-web"],
                    help="run a single stage")
    ap.add_argument("--refresh-roster", action="store_true",
                    help="ignore cached roster and re-fetch")
    ap.add_argument("--refresh-activity", action="store_true",
                    help="ignore activity cache and re-fetch all logins")
    ap.add_argument("--refresh-repo-activity", action="store_true",
                    help="ignore cached repo activity and re-fetch")
    ap.add_argument("--max-cache-age-days", type=int, default=7,
                    help="re-fetch activity caches older than this (default 7)")
    ap.add_argument("--activity-window-days", type=int, default=2*365,
                    help="window for 'active' metric (default is two years)")
    ap.add_argument("--start-year", type=int, default=None,
                    help="clip plot x-axis start (default: first joined date)")
    ap.add_argument("--show", dest="show", action="store_true", default=None,
                    help="open interactive plot windows (default: auto-detect "
                         "from $DISPLAY + stdout.isatty())")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="force PNG-only output (no GUI windows)")
    args = ap.parse_args()

    # Auto-detect interactive mode if user did not set --show / --no-show.
    if args.show is None:
        args.show = bool(os.environ.get("DISPLAY") or
                         os.environ.get("WAYLAND_DISPLAY")) and sys.stdout.isatty()
    if not args.show:
        # Lock in the non-interactive backend before any pyplot call.
        plt.switch_backend("Agg")

    if not os.environ.get("GITHUB_TOKEN"):
        print("WARNING: GITHUB_TOKEN not set — you will hit the 60/hr "
              "unauthenticated rate limit fast. Create a fine-grained "
              "token at https://github.com/settings/tokens with public_repo "
              "read scope.", file=sys.stderr)

    stages = (["roster", "activity", "repo-activity", "plot"] if args.only is None
              else [args.only])
    roster_path = DATA_DIR / "roster.json"
    for s in stages:
        if s == "roster":
            roster_path = stage_fetch_roster(force=args.refresh_roster)
        elif s == "activity":
            stage_fetch_activity(roster_path,
                                 max_cache_age_days=args.max_cache_age_days,
                                 refresh=args.refresh_activity)
        elif s == "repo-activity":
            stage_fetch_repo_activity(force=args.refresh_repo_activity)
        elif s == "plot":
            stage_plot(roster_path,
                       activity_window_days=args.activity_window_days,
                       start_year=args.start_year,
                       show=args.show)
        elif s == "export-web":
            stage_export_web(roster_path)


if __name__ == "__main__":
    main()
