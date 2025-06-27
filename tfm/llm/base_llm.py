from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for LLMs."""

    def __init__(self, model_name: str):
        """Empty initializer for BaseLLM.

        Args:
            model_name (str): Name of the LLM.
        """
        self.model_name = model_name

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0) -> str:
        """Generate text based on the system prompt and user prompt.

        Args:
            system_prompt (str): The system's instruction or context.
            user_prompt (str): The user's query or message.
            temperature (float, optional): _description_. Defaults to 0.0.

        Returns:
            str: The generated text response.
        """
        pass
