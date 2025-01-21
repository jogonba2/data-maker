import tiktoken


def get_openai_tokenizer(model_name: str = "gpt-4o") -> tiktoken.core.Encoding:
    """
    Get the tokenizer for an OpenAI LLM.

    Args:
        model_name (str): the LLM name

    Returns:
        tiktoken.core.Encoding: the associated tokenizer
    """
    return tiktoken.encoding_for_model(model_name)


def truncate(text: str, max_length: int, model: str = "gpt-4o") -> str:
    """
    Truncates a text to have `max_length` token ids.

    Args:
        text (str): a text
        max_length (int): max length in tokens
        model (str): name of the LLM

    Returns:
        str: the detokenized text, having at most `max_length` tokens.
    """
    tokenizer = get_openai_tokenizer(model)
    truncated_tokens = tokenizer.encode(text)[:max_length]
    truncated_text = tokenizer.decode(truncated_tokens)

    return truncated_text
