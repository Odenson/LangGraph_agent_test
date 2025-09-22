import os
from flask.cli import load_dotenv
from langchain.chat_models import init_chat_model

from typing import Annotated
#from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from rich.console import Console
from rich.panel import Panel

from tavily_tool import tavily_tool
from count_tokens import count_tokens
from google_search import google_search

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
agent_tools = [tavily_tool, count_tokens, google_search]

# Bind the tools to the language model
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

# system prompt
system_prompt = SystemMessage(
    content=(
        "Answer the user question and be concise. Respond with only the essential information."
        "Use both the Tavily and Google Search tools to find the answer. "
        "In your answer, cite both results separately and show the source. "
        "Also include a details of token usage and the total number of characters used at the end of your answer, in the following format"
        " (Tokens used: <estimated_token_count>, Characters used: <character_count>)."   
    )
)

if __name__ == "__main__":
    # Initialize the Rich console for pretty output
    console = Console()
    os.system('cls')

    # Prompt the user for a question in the terminal
    print("-" *20+ " Chat with LangGraph " + "-" * 20)

    user_input = input("Enter your question for the chatbot, or press enter for default: \n\n").strip()
    if not user_input:
        user_input = "What was the largest stock market crash and why?"

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