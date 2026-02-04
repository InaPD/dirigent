"""Graph wiring.

    START -> router
    router -> plan | research | review | write | END      (conditional, on route())
    plan | research | review | write -> router

Four work nodes and one router. The router is a node so that later phases can record a
budget step from it, but the dispatch itself is route(), a pure function of the state.

Deliberately built from add_node, add_edge and add_conditional_edges only. The Command
goto API would express the same thing, but these three are the oldest and most stable
surface LangGraph has, and the point of this file is that it does not move.
"""

from langgraph.graph import END, START, StateGraph

from ra.deps import Deps
from ra.nodes.base import NodeFn, as_graph_node
from ra.nodes.canned import CANNED_NODES
from ra.routing import route
from ra.schemas import RunState

WORK_NODES = ("plan", "research", "review", "write")

# A run visits the router once per work node, so the step count is roughly twice the node
# count. Four sub-questions with one revision each is about 40 hops.
RECURSION_LIMIT = 100


async def router(state: RunState) -> dict:
    """Pass through in Phase 1. Phase 2 adds the budget check and its step record."""
    return {}


def select_nodes(deps: Deps) -> dict[str, NodeFn]:
    """Which implementation of each node to use. Phase 1 has only the canned set."""
    return dict(CANNED_NODES)


def build_graph(deps: Deps):
    """Compile the graph. One per process; it holds no run state."""
    builder = StateGraph(RunState)
    builder.add_node("router", router)
    builder.add_edge(START, "router")

    nodes = select_nodes(deps)
    for name in WORK_NODES:
        builder.add_node(name, as_graph_node(name, nodes[name], deps))
        builder.add_edge(name, "router")

    builder.add_conditional_edges(
        "router",
        route,
        {name: name for name in WORK_NODES} | {END: END},
    )
    return builder.compile()
