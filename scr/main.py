import os
from flask.cli import load_dotenv
from langchain.chat_models import init_chat_model

from typing import Annotated
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from rich.console import Console
from rich.panel import Panel

# Load environment variables from .env file
load_dotenv()  

# Load API keys from environment variable
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize the OpenAI chat model
llm = init_chat_model("openai:gpt-4.1")

# Define the state for the graph, using TypedDict for type safety
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create a new state graph for the conversation
graph_builder = StateGraph(State)

# Initialize the Tavily search tool
tool = TavilySearch(
    max_results=2,
    topic="general",
    # Additional options can be set here if needed
)

# List of tools to be used by the model
tools = [tool]

# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node logic
def chatbot(state: State):
    # The chatbot node invokes the model with the current messages
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Create and add the tool node to the graph
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Add conditional edges: if a tool is needed, go to the tool node
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# After using a tool, return to the chatbot node
graph_builder.add_edge("tools", "chatbot")
# Start the graph at the chatbot node
graph_builder.add_edge(START, "chatbot")
# Compile the graph for execution
graph = graph_builder.compile()

if __name__ == "__main__":
    # Example system prompt to reduce verbosity
    system_prompt = SystemMessage(content="Be concise. Respond with only the essential information.")

    # Prompt the user for a question in the terminal
    user_input = input("Enter your question for the chatbot: ").strip()
    if not user_input:
        user_input = "What was the largest stock market crash and why?"
    question = HumanMessage(content=user_input)

    # Prepend the system prompt to the messages
    messages = [system_prompt, question]

    # Invoke the graph with the initial messages
    answer = graph.invoke({"messages": messages})

    # Initialize the Rich console for pretty output
    console = Console()
    # Get the formatted answer from the response
    formatted_answer = answer["messages"][-1].content
    # Print the answer in a styled panel
    console.print(Panel(formatted_answer, title="Answer", border_style="blue"))

print("Done")