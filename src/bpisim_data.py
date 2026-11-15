"""Calibrate BPISim from recorded incomplete application handling."""
from __future__ import annotations

from collections import Counter
import gzip
import xml.etree.ElementTree as ET

from src.bpi_queue import work_segments
from src.cycle_data import CLOSURES
from src.domain_loop_data import Attempt, Case
from src.loop_feasibility import attributes, local, timestamp


WORK_NAMES = {"W_Call incomplete files", "W_Validate application"}
OUTCOME_BY_MARKER = {"A_Incomplete": "repeat", "A_Pending": "resolved",
                     "A_Cancelled": "negative", "A_Denied": "negative"}


def extract_bpi_case(key, events, source_end, config, audit):
    markers = [(timestamp(row["time:timestamp"]), row["concept:name"])
               for row in events if row["concept:name"] in CLOSURES | {"A_Validating"}]
    markers.sort(key=lambda row: row[0])
    first = next((index for index, (_, name) in enumerate(markers) if name == "A_Incomplete"), None)
    if first is None:
        return None
    audit["cases_with_incomplete"] += 1
    work, work_audit = work_segments([row for row in events if row["concept:name"] in WORK_NAMES], config)
    if work_audit["cases_invalid"]:
        audit["cases_with_invalid_work_lifecycle"] += 1
        return None

    start = markers[first][0]
    opened, milestone = start, None
    windows = []
    for index, (time, name) in enumerate(markers[first + 1:], first + 1):
        if name == "A_Validating" and milestone is None:
            milestone = time
        if name not in CLOSURES:
            continue
        windows.append((opened, time, OUTCOME_BY_MARKER[name], milestone))
        if name != "A_Incomplete":
            audit["incomplete_markers_after_exit"] += sum(activity == "A_Incomplete" for _, activity in markers[index + 1:])
            break
        opened, milestone = time, None
    else:
        windows.append((opened, source_end, "unobserved", milestone))

    attempts = []
    for begin, end, outcome, milestone in windows:
        # Work already active at the routing decision belongs to the prior attempt.
        selected = [(row["start"], min(row["end"], end)) for row in work if begin <= row["start"] < end]
        if not selected:
            audit["attempts_without_post_marker_work"] += 1
            continue
        first_work = selected[0][0]
        latest, service = first_work, 0.
        for left, right in selected:
            service += max(0., right - max(left, latest))
            latest = max(latest, right)
        attempts.append(Attempt(outcome, (end - begin) / 60, service / 60,
                                None if milestone is None else (milestone - begin) / 60,
                                (first_work - begin) / 60))
    if len(attempts) != len(windows):
        audit["cases_missing_work_in_an_attempt"] += 1
        return None
    audit.update({"eligible_attempts": len(attempts),
                  "attempts_with_observed_milestone": sum(item.milestone is not None for item in attempts),
                  "attempts_with_pre_service_delay": sum(item.before_service > 0 for item in attempts)})
    audit.update(f"terminal_{item.outcome}" for item in attempts if item.outcome != "repeat")
    return Case(key, start / 60, source_end / 60, tuple(attempts))


def load_bpi_cases(path, spec):
    cutoff = spec["protected_start_epoch_seconds"]
    audit, cases = Counter(), []
    with gzip.open(path, "rb") as stream:
        parser = ET.iterparse(stream, events=("start", "end"))
        _, root = next(parser)
        for kind, trace in parser:
            if kind != "end" or local(trace.tag) != "trace":
                continue
            audit["source_cases"] += 1
            elements = [node for node in trace if local(node.tag) == "event"]
            audit["source_events"] += len(elements)
            times = [timestamp(value.get("value")) for node in elements for value in node
                     if value.get("key") == "time:timestamp"]
            if not times or min(times) >= cutoff:
                audit["cases_at_or_after_protected_start"] += 1
            elif max(times) >= cutoff:
                audit["cases_crossing_protected_start"] += 1
            else:
                key = attributes(trace)["concept:name"]
                rows = [attributes(node) for node in elements]
                case = extract_bpi_case(key, rows, max(times), spec, audit)
                if case is not None:
                    cases.append(case)
            trace.clear()
            root.clear()
    audit["eligible_cases"] = len(cases)
    return cases, dict(audit)
