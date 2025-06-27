import tiktoken

from tfm.llm.base_llm import BaseLLM
from tfm.settings import settings
import requests


class VLLMLLM(BaseLLM):
    def __init__(self, model_name: str):
        """Initialize the vLLM LLM with the name of a given LLM.

        Args:
            model_name (str): Name of the OpenAI LLM.
        """
        super().__init__(model_name)
        self.url = "http://localhost:8000/v1/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",  # vLLM no valida el token
        }

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        frequency_penalty: float = 0.0, 
        presence_penalty: float = 0.0
    ) -> str:
        """Generate text based on the system prompt and user prompt.

        Args:
            system_prompt (str): The system's instruction or context.
            user_prompt (str): The user's query or message.
            temperature (float, optional): _description_. Defaults to 0.0.
            past_messages (List[Dict[str, str]]): list of dictionaries, where each dictionary represents a message, with keys 'role' and 'content'. Defaults to [].

        Returns:
            str: The generated text response.
        """
        prompt = self.prepare_prompt(system_prompt, user_prompt) # vLLM only receives a single prompt, not a list of messages
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 2048, 
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        resp = requests.post(self.url, headers=self.headers, json=data)
        return resp.json()["choices"][0]['text']
    
    def prepare_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Prepare the prompt by combining the system prompt and user prompt.

        Args:
            system_prompt (str): The system's instruction or context.
            user_prompt (str): The user's query or message.

        Returns:
            str: The combined prompt.
        """
        return f"{system_prompt}\n\n{user_prompt}"