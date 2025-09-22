from langchain.tools import tool

# Register this function as a LangChain tool (so it can be called in agents or chains)
@tool
def google_search(query: str) -> str:
    """
    Uses Google to perform a web search based on the latest user message.
    """

    #from langchain_core.tools import Tool
    from langchain_google_community import GoogleSearchAPIWrapper

    search = GoogleSearchAPIWrapper(k=5)

    response = search.run(query)
    return response

if __name__ == "__main__":
    # Prompt the user for a question in the terminal
    user_input = input("Enter your question for Google Search: ")
    response = google_search(user_input)
    print("Google Search Response:")
    print(response)