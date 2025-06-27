import tiktoken
from openai import OpenAI

from tfm.llm.base_llm import BaseLLM
from tfm.settings import settings


class OpenAILLM(BaseLLM):
    """Class to interact with OpenAI's large language models, including generating text and streaming responses based on given prompts.

    The OpenAILLM class facilitates interaction with OpenAI's language models by providing
    methods to generate text or stream responses. The class initializes with the model name
    and sets up token encoding and context window sizes based on the model type. It includes
    methods such as generate_text for generating complete responses based on system and user
    prompts, and generate_text_stream for streaming text responses in chunks. The class also
    ensures that responses are deterministic by controlling temperature, top_p, and seed values.
    """

    def __init__(self, model_name: str):
        """Initialize the OpenAI LLM with the name of a given OpenAI LLM.

        Args:
            model_name (str): Name of the OpenAI LLM.
        """
        super().__init__(model_name)
        self.client = OpenAI(
            api_key=settings.openai_apikey
            )

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
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            store=True,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content
