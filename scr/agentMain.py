import os
from flask.cli import load_dotenv
from langchain.chat_models import init_chat_model

from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from rich.console import Console
from rich.panel import Panel

from tools.tavily_tool import tavily_tool
from tools.count_tokens import count_tokens
from tools.google_search import google_search
from tools.answerQuality import answer_similarity

# Load environment variables from .env file
load_dotenv()  

# Load API keys from environment variable
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Initialize the OpenAI chat model
llm = init_chat_model("openai:gpt-4.1")

# Define the state for the graph, using TypedDict for type safety
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create a new state graph for the conversation
graph_builder = StateGraph(State)

# List of tools to be used by the model
agent_tools = [tavily_tool, google_search]

# Bind the tools to the language model that the LLM can use to answer user queries
llm_with_tools = llm.bind_tools(agent_tools)

# Define the chatbot node logic
def chatbot(state: State):
    # The chatbot node invokes the model with the current messages
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Create and add the tool node to the graph
tool_node = ToolNode(tools=agent_tools)
graph_builder.add_node("tools", tool_node)
#graph_builder.add_node("tavily_search", ToolNode(tools=[tavily_tool]))
#graph_builder.add_node("google_search", ToolNode(tools=[google_search]))

graph_builder.add_node("count_tokens", ToolNode(tools=[count_tokens])) # Add a separate node for the count_tokens tool as it must run last
graph_builder.add_node("answer_similarity", ToolNode(tools=[answer_similarity])) # Add a separate node for the answer_similarity tool

# Add conditional edges: if a tool is needed, go to the tool node
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Start the graph at the chatbot node
graph_builder.add_edge(START, "chatbot")
# After using a tool, return to the chatbot node
graph_builder.add_edge("tools", "chatbot")
#graph_builder.add_edge("chatbot", "tavily_search")
#graph_builder.add_edge("chatbot", "google_search")

graph_builder.add_edge("chatbot", "count_tokens")
graph_builder.add_edge("chatbot", "answer_similarity")
# End the graph after the count_tokens node
graph_builder.add_edge("count_tokens", END)
# Compile the graph for execution
graph = graph_builder.compile()

# system prompt
system_prompt = SystemMessage(
    content=(
        "Answer the user question and be concise. Respond with only the essential information."
        "Use both the Tavily and Google Search tools to find the answer. "
        "In your answer, cite both results separately and show their source. "
        "Also include a details of token usage and the total number of characters used at the end of your answer" 
        " Your reply should follow the format below:"
        " <Answer to the user question> "
        " "
        " <Answer from Tavily> : source <URL> "
        " <Answer from Google Search> : source <URL> "
        " "
        " Agent Tool (Tokens used: <estimated_token_count>, Characters used: <character_count>). These values must be obtained from the count_tokens tool without alteration."
        " LLM Details (Tokens used: <total_tokens>). This information must be obtained from the LLM response looking for the response metadata for the total_tokens values."
        " The similarity of the two answers is <similarity_percentage>% (similarity score: <similarity_score>). This information must be obtained from the answer_similarity tool without alteration."
    )
)

if __name__ == "__main__":
    # Initialize the Rich console for pretty output
    console = Console()
    os.system('cls')

    #testing
    # Inspect all messages in the state
    from langchain.globals import set_debug
    set_debug(False)

    # Prompt the user for a question in the terminal
    print("-" *20+ " Chat with LangGraph " + "-" * 20)

    user_input = input("Enter your question for the chatbot, or press enter for default: \n\n").strip()
    if not user_input:
        user_input = "What is the worlds largest dog breed?"

    print("\n" + "-" *61)

    question = HumanMessage(content=user_input)

    # Prepend the system prompt to the messages
    messages = [system_prompt, question]

    # Invoke the graph with the initial messages
    answer = graph.invoke({"messages": messages})
    
    # Get the formatted answer from the response
    formatted_answer = answer["messages"][-1].content
    # Print the answer in a styled panel
    console.print(Panel(formatted_answer, title="Answer", border_style="blue"))

    print("Done")