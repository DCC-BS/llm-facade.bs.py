from pydantic import BaseModel


class LLMConfig(BaseModel):
    """
    Configuration for the LLM.

    Attributes:
        model_name (str): The name of the LLM model to use.
        api_key (str): The API key for accessing the LLM service.
        base_url (str): The base URL for the LLM service.
    """

    openai_api_key: str
    openai_api_base_url: str
    llm_model: str
