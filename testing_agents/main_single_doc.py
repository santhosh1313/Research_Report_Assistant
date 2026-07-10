import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from vault import Vault
from agents.atlas import atlas_route
from agents.pathfinder import pathfinder_plan
from agents.harvester import (
    harvester_web_search,
    harvester_parse_single,
    harvester_embed_multi,
)
from agents.synthesizer import synthesizer_run
from agents.scribe import scribe_write


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

if __name__ == "__main__":
    vault = Vault()

    # --- Pick ONE input style to test ---
    # vault.input_data = "Impact of Artificial Intelligence in Healthcare"   # topic mode
    vault.input_data = ["docs/English_paper.pdf"]                        # single_doc mode
    # vault.input_data = ["docs/appreciation.pdf", "docs/satisfaction.pdf"]# multi_doc mode

    result = app.invoke({"vault": vault})

    print("\n===== FINAL REPORT =====\n")
    print(result["vault"].final_report)