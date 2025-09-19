from langchain.tools import tool

@tool
def count_tokens(messages: list) -> str:
    """
    Counts the number of characters in the messages, estimates the token count,
    and counts the number of responses in the interaction.
    Returns a string with the estimated token count and response count.
    """
    text = ""
    response_count = 0
    for msg in messages:
        if hasattr(msg, "content"):
            text += str(msg.content)
            response_count += 1
        else:
            text += str(msg)
            response_count += 1
    char_count = len(text)
    token_estimate = char_count // 4
    return (
        f"Estimated token count for this interaction: {token_estimate} tokens "
        f"(based on {char_count} characters). "
        f"Number of responses in this interaction: {response_count}."
    )