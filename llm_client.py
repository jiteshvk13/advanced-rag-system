from functools  import lru_cache
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import config


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGroq:
    """
    Create and return the Groq chat model.

    Returns:
        ChatGroq: Configured Groq chat model.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
    """
    config.check_groq_api_key()

    return ChatGroq(
        model = config.GROQ_CHAT_MODEL,
        temperature = config.LLM_TEMPERATURE,
        api_key = config.GROQ_API_KEY,
    )

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create (or reuse) the Hugging Face embedding model used to build
    the vector index.

    An embedding turns a piece of text into a list of numbers (a vector).
    Texts with similar meaning get similar vectors, which makes
    semantic search possible.

    Returns:
        A configured LangChain HuggingFaceEmbeddings model.
    """

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )
