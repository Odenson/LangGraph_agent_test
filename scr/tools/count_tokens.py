from langchain.tools import tool
import json

# Register this function as a LangChain tool
@tool
def count_tokens(messages: list) -> str:
    """
    Counts the number of characters in the messages, estimates the token count,
    and counts the number characters in the messages.
    Returns a json with the estimated token count and character count.
    """

    text = ""
    response_count = 0 

    # Iterate through all messages in the list
    for msg in messages:
        # If the message has a 'content' attribute (e.g., an AIMessage or HumanMessage object in LangChain)
        if hasattr(msg, "content"):
            text += str(msg.content)
            response_count += 1 
        else:
            # If it's a plain string or another type without 'content'
            text += str(msg)
            response_count += 1

    # Count how many characters are in the combined text
    char_count = len(text)

    # Estimate token count: roughly 1 token per 4 characters (a common heuristic)
    token_estimate = char_count // 4
    
    # Build a JSON object with interaction details
    result = {
        "character_count": char_count,
        "estimated_token_count": token_estimate,
        "response_count": response_count,
    }

    # Return as a JSON string (so LangChain can easily handle it)
    return json.dumps(result, indent=2)
