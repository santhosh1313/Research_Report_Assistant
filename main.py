import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from vault import Vault
from agents.atlas import atlas_route
from agents.pathfinder import pathfinder_plan, PathfinderError
from agents.harvester import (
    harvester_web_search,
    harvester_parse_single,
    harvester_embed_multi,
    HarvesterError,
)
from agents.synthesizer import synthesizer_run, SynthesizerError
from agents.scribe import scribe_write, ScribeError
from workflow_validator import validate_workflow, save_trace, print_trace_summary


class GraphState(TypedDict):
    vault: Any  # holds your existing Vault object, unchanged


# ---- Node wrappers: each just calls your existing agent function ----
def atlas_node(state: GraphState) -> GraphState:
    atlas_route(state["vault"])
    return state


def pathfinder_node(state: GraphState) -> GraphState:
    pathfinder_plan(state["vault"])
    return state


def harvester_web_node(state: GraphState) -> GraphState:
    harvester_web_search(state["vault"])
    return state


def harvester_single_node(state: GraphState) -> GraphState:
    harvester_parse_single(state["vault"])
    return state


def harvester_multi_node(state: GraphState) -> GraphState:
    harvester_embed_multi(state["vault"])
    return state


def synthesizer_node(state: GraphState) -> GraphState:
    synthesizer_run(state["vault"])
    return state


def scribe_node(state: GraphState) -> GraphState:
    scribe_write(state["vault"])
    return state


# ---- Routing logic: decide which Harvester variant to call ----
def route_after_pathfinder(state: GraphState) -> str:
    mode = state["vault"].mode
    if mode == "topic":
        return "harvester_web"
    elif mode == "single_doc":
        return "harvester_single"
    elif mode == "multi_doc":
        return "harvester_multi"
    raise ValueError(f"Unknown mode: {mode}")


# GROQ_API_KEY added: Pathfinder and Scribe both depend on it, so a missing
# key should fail fast at startup instead of deep inside a graph node.
required_keys = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "GROQ_API_KEY"]

missing = [k for k in required_keys if not os.getenv(k)]

if missing:
    print(f"Missing required environment variables: {missing}")
    print("Check your .env file.")
    sys.exit(1)

# ---- Build the graph ----
graph = StateGraph(GraphState)
graph.add_node("atlas", atlas_node)
graph.add_node("pathfinder", pathfinder_node)
graph.add_node("harvester_web", harvester_web_node)
graph.add_node("harvester_single", harvester_single_node)
graph.add_node("harvester_multi", harvester_multi_node)
graph.add_node("synthesizer", synthesizer_node)
graph.add_node("scribe", scribe_node)

graph.set_entry_point("atlas")
graph.add_edge("atlas", "pathfinder")
graph.add_conditional_edges(
    "pathfinder",
    route_after_pathfinder,
    {
        "harvester_web": "harvester_web",
        "harvester_single": "harvester_single",
        "harvester_multi": "harvester_multi",
    },
)
graph.add_edge("harvester_web", "synthesizer")
graph.add_edge("harvester_single", "synthesizer")
graph.add_edge("harvester_multi", "synthesizer")
graph.add_edge("synthesizer", "scribe")
graph.add_edge("scribe", END)

app = graph.compile()

# ---- Run it ----
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-AI Agent Research & Report Assistant")
    parser.add_argument("--topic", type=str, help="Research topic (topic mode)")
    parser.add_argument("--docs", nargs="+", help="One or more PDF paths (document mode)")
    parser.add_argument("--out", type=str, default="report.txt", help="Output file for the final report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.topic and not args.docs:
        print("Error: provide either --topic \"your topic\" or --docs path1.pdf path2.pdf")
        sys.exit(1)

    vault = Vault()
    vault.input_data = args.topic if args.topic else args.docs

    try:
        result = app.invoke({"vault": vault})
    except (PathfinderError, HarvesterError, SynthesizerError, ScribeError) as e:
        # Known, named failure points in the pipeline — surfaced with the
        # specific agent that failed so it's actionable instead of a raw traceback.
        print(f"Pipeline failed at a known step: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Input file error: {e}")
        sys.exit(1)
    except Exception as e:
        # Anything unexpected — still caught so the CLI exits cleanly
        # instead of dumping a raw stack trace to the user.
        print(f"Pipeline failed with an unexpected error: {e}")
        sys.exit(1)

    final_report = result["vault"].final_report
    print("\n===== FINAL REPORT =====\n")
    print(final_report)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"\nReport saved to {args.out}")

    # Milestone 3 — validate and record that the agents actually
    # collaborated as expected, and save the evidence alongside the report.
    passed, trace_report = validate_workflow(result["vault"])
    print_trace_summary(trace_report)
    trace_path = save_trace(trace_report, path=os.path.splitext(args.out)[0] + "_trace.json")
    print(f"Workflow trace saved to {trace_path}")