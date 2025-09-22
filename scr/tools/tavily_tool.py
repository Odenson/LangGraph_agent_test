from langchain.tools import tool
#from langchain_tavily import TavilySearch

# Instantiate the TavilySearch tool once (recommended for efficiency)
#tavily_search = TavilySearch(
#    max_results=2,
#    topic="general",
#    # Additional options can be set here if needed
#)

@tool
def tavily_tool(messages: list) -> str:
    """
    Uses TavilySearch to perform a web search based on the latest user message.
    Returns the top results as a string.
    """
    from langchain_tavily import TavilySearch

    # Instantiate the TavilySearch tool once (recommended for efficiency)
    tavily_search = TavilySearch(
        max_results=2,
        topic="general",
        # Additional options can be set here if needed
    )

    # Extract the latest human message (assumes last message is the query)
    if not messages:
        return "No messages provided for search."
    last_message = messages[-1]
    query = getattr(last_message, "content", str(last_message))
    results = tavily_search.invoke(query)
    if isinstance(results, list):
        return "\n".join(str(r) for r in results)
    
    return str(results)


