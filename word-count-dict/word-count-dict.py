def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    if len(sentences) == 0:
        return {}
    else:
        docs = sentences[0]
        for i in range(1, len(sentences)):
            docs += sentences[i]
        words = set(docs)
        count = {word: docs.count(word) for word in words}
        return count 