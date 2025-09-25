from langchain.tools import tool
import json

# Register this function as a LangChain tool
@tool
def answer_similarity(messages: list) -> str:
    """
    given two lists of messages, compare the similarity of the answers in the messages.
    Returns a json with the similarity score and the two answers.
    """

    text1 = ""
    text2 = ""

    # Iterate through the mssages in the list and separate into two texts
    for i, msg in enumerate(messages):
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if content is not None:
            if i % 2 == 0:
                text1 += str(content) + " "
            else:
                text2 += str(content) + " "
        else:
            if i % 2 == 0:
                text1 += str(msg) + " "
            else:
                text2 += str(msg) + " "

    # Cosine similarity can be calculated using various libraries, here we use a simple approach
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics.pairwise import cosine_similarity

    pipeline = Pipeline([
        ("count", CountVectorizer()),
        ("tfidf", TfidfTransformer())
    ])
    # Fit and transform both texts
    vectors = pipeline.fit_transform([text1, text2]).toarray()
    cosine_sim = cosine_similarity(vectors)
    similarity_score = float(cosine_sim[0][1])

    result = {
        "text1": text1.strip(),
        "text2": text2.strip(),
        "similarity_score": similarity_score,
        "similarity_percentage": round(similarity_score * 100, 2)
    }

    # Return as a JSON string (so LangChain can easily handle it)
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage
    # Example usage
    messages = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    ]

    #test function
    result = answer_similarity.invoke({"messages": messages})
    print(result)