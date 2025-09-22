from langchain.tools import tool
import json

@tool
def tavily_tool(messages: list) -> str:
    """
    Uses TavilySearch to perform a web search based on the latest user message.
    Returns the top results as a JSON string with result, source, and max_results.
    """
    from langchain_tavily import TavilySearch

    max_results = 2
    tavily_search = TavilySearch(
        max_results=max_results,
        topic="general",
    )

    if not messages:
        return json.dumps({
            "result": "No messages provided for search.",
            "source": "Tavily",
            "max_results": max_results
        })
    last_message = messages[-1]
    query = getattr(last_message, "content", str(last_message))
    results = tavily_search.invoke(query)

    if isinstance(results, list):
        result_str = "\n".join(str(r) for r in results)
    else:
        result_str = str(results)

    return json.dumps({
        "result": result_str,
        "source": "Tavily",
        "max_results": max_results
    })


