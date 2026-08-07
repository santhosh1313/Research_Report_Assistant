# workflow_validator.py
"""
Validates that a pipeline run actually exhibited collaborative multi-agent
behavior: every agent received a handoff, did its work, and passed output
to the next agent in the expected order. Produces a demonstrable trace
artifact (comm_log + state snapshot) for Milestone 3 — Agent Coordination
& Memory Systems — rather than relying on eyeballing console prints.
"""
import json
from datetime import datetime

# The order in which agents are expected to hand off work, regardless of
# mode — Harvester's three mode-specific functions all log as "Harvester".
EXPECTED_HANDOFF_SEQUENCE = ["Atlas", "Pathfinder", "Harvester", "Synthesizer", "Scribe"]


def validate_workflow(vault):
    """Returns (passed: bool, report: dict) describing whether the
    collaborative workflow executed as expected for this run."""
    issues = []

    actual_sequence = [entry["from"] for entry in vault.comm_log]

    if actual_sequence != EXPECTED_HANDOFF_SEQUENCE:
        issues.append(
            f"Handoff sequence mismatch — expected {EXPECTED_HANDOFF_SEQUENCE}, got {actual_sequence}"
        )

    if not vault.subtasks:
        issues.append("Pathfinder produced no subtasks")

    if not vault.synthesis or not vault.synthesis.strip():
        issues.append("Synthesizer produced no synthesis")

    if not vault.final_report or not vault.final_report.strip():
        issues.append("Scribe produced no final report")

    passed = not issues

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": vault.mode,
        "passed": passed,
        "issues": issues,
        "handoff_sequence": actual_sequence,
        "communication_log": vault.comm_log,
        "state_snapshot": {
            "subtasks_count": len(vault.subtasks),
            "facts_count": len(vault.facts),
            "synthesis_length": len(vault.synthesis) if vault.synthesis else 0,
            "final_report_length": len(vault.final_report) if vault.final_report else 0,
            "long_term_memory_hits": len(vault.retrieved_memory),
        }
    }
    return passed, report


def save_trace(report, path="workflow_trace.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def print_trace_summary(report):
    status = "PASS" if report["passed"] else "FAIL"
    print(f"\n===== WORKFLOW VALIDATION: {status} =====")
    print(f"Mode: {report['mode']}")
    print("Agent handoffs:")
    for entry in report["communication_log"]:
        print(f"  {entry['from']} -> {entry['to']}: {entry['action']} ({entry['detail']})")
    if report["issues"]:
        print("Issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
    snap = report["state_snapshot"]
    print(
        f"State: {snap['subtasks_count']} subtasks, {snap['facts_count']} facts, "
        f"{snap['synthesis_length']} char synthesis, {snap['final_report_length']} char report, "
        f"{snap['long_term_memory_hits']} long-term memory hits"
    )