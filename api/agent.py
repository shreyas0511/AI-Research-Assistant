from langgraph.graph import StateGraph, END
from utils.state import AgentState
from utils.nodes import *

# graph construction
graph = StateGraph(AgentState)

# add the planner node and set it as the start node
graph.add_node("planner", planner)
graph.set_entry_point("planner")

# add the router node
# add planner -> router edge
graph.add_node("router", passthrough)
graph.add_edge("planner", "router")

# add the retrieval node
graph.add_node("retrieval", retrieve)

# add the arxiv search node
# conditional edge from router -> arxiv
# edge from arxiv -> router
graph.add_node("search_arxiv", search_arxiv)
graph.add_conditional_edges(
    "router",
    router,
    {
        "tool call": "search_arxiv",
        "relevance": "retrieval"
    }
)
graph.add_edge("search_arxiv", "router")

# add reflection and summarize node
# conditional edge from reflection -> summarize
# conditional edge from reflection -> planner
graph.add_node("reflection", reflection)

# add an edge from retrieval to reflection
graph.add_edge("retrieval", "reflection")

graph.add_node("summarize", summarize)
graph.add_node("reflection_router", passthrough)
graph.add_edge("reflection", "reflection_router")

graph.add_conditional_edges(
    "reflection_router",
    reflection_router,
    {
        "summarize":"summarize",
        "plan": "planner"
    }
)

# mark summarize node as the end node
graph.add_edge("summarize", END)

agent = graph.compile()