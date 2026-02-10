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

from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from ra.budgets import apply_budget_stop, check_budgets
from ra.deps import Deps
from ra.nodes.canned import CANNED_NODES
from ra.nodes.plan import plan
from ra.nodes.research import research
from ra.nodes.review import review
from ra.nodes.stubs import STUBS
from ra.nodes.write import write
from ra.progress import apply_stall_stop, is_stalled
from ra.routing import route
from ra.schemas import RunState
from ra.trace import NodeFn, as_graph_node

WORK_NODES = ("plan", "research", "review", "write")

# A run visits the router once per work node, so the step count is roughly twice the node
# count. Four sub-questions with one revision each is about 40 hops.
RECURSION_LIMIT = 100


def make_router(deps: Deps) -> Callable[[RunState], Awaitable[dict]]:
    """The router checks the caps, then the conditional edge asks route() where to go.

    It is a node rather than a bare edge function so that a tripped cap can be written into
    the trace as a step, with the names of the caps that tripped.
    """

    async def router(state: RunState) -> dict:
        if state.is_terminal or state.budget_stopped:
            return {}

        if is_stalled(state):
            stopped = apply_stall_stop(state)
        else:
            verdict = check_budgets(state, next_node=route(state))
            if verdict.ok:
                return {}
            stopped = apply_budget_stop(state, verdict)
        await deps.store.save(stopped)
        if stopped.worker_id:
            await deps.store.refresh_lease(stopped.run_id, stopped.worker_id)
        return stopped.model_dump()

    return router


def select_nodes(deps: Deps) -> dict[str, NodeFn]:
    """Which implementation of each node to use.

    A node falls back to its canned stand-in when its dependencies are missing, which is what
    lets the whole pipeline run in tests and in CI with no credentials.
    """
    nodes = dict(CANNED_NODES)
    if deps.llm is not None:
        nodes["plan"] = plan
        nodes["review"] = review
        nodes["write"] = write
    if deps.llm is not None and deps.search is not None:
        nodes["research"] = research
    if deps.settings.stub:
        # Test scaffolding, selected with RA_STUB. Applied last so it wins.
        nodes.update(STUBS[deps.settings.stub])
    return nodes


def build_graph(deps: Deps):
    """Compile the graph. One per process; it holds no run state."""
    builder = StateGraph(RunState)
    builder.add_node("router", make_router(deps))
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
