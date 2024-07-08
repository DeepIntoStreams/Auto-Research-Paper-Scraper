# Class that helps query an LLM by wrapping the client in a common interface used throughout the code of this repository.
import os
from abc import ABC, abstractmethod
from typing import Optional

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI

from config import ENABLE_LLM_FILTER, PROMPT_LLM_DEEPINTOMLF, MODEL_LLM


class Llm_wrapper(ABC):
    # It is important that all child classes have a parameter-free constructor, to be used in the factory pattern.
    def __init__(self, client: Optional[object]):
        self.client = client

    @abstractmethod
    def query(self, prompt: str) -> Optional[str]:
        """
        Query the LLM with a given prompt.
        Args:
            prompt ():

        Returns:

        """
        pass


class llm_wrapper_mistral(Llm_wrapper):
    def __init__(self):
        if "MISTRAL_API_KEY" not in os.environ or not ENABLE_LLM_FILTER:
            print("No API key for the Mistral LLM found, returning None.")
            client = None
        else:
            client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
        super().__init__(client)

    def query(self, prompt: str) -> Optional[str]:
        try:
            answer = self.client.chat(
                model=MODEL_LLM,
                messages=[ChatMessage(role="user", content=prompt)],
                # Set 100 to get explanation from LLM in answer
                max_tokens=1,
                temperature=0.03,
            )

            res = answer.choices[0].message.content

            if answer.choices[0].finish_reason == 'error':
                # Error 401 means the API key is invalid.
                print(f"Error with the LLM for the prompt: \n                     {prompt[:50]}.")
                return None
            return res
        except Exception as e:
            print(
                f"Error with the LLM for the prompt: \n                     {prompt[:50]}. "
                f"\n                     Error: {e}"
            )
            return None


class llm_wrapper_openai(Llm_wrapper):
    def __init__(self):
        if "OPENAI_API_KEY" not in os.environ or not ENABLE_LLM_FILTER:
            print("No API key for the OpenAI LLM found, using None for client.")
            client = None
        else:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        super().__init__(client)

    def query(self, prompt: str) -> Optional[str]:
        try:
            prompt = PROMPT_LLM_DEEPINTOMLF + prompt

            answer = self.client.chat.completions.create(
                model=MODEL_LLM,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                # Set 100 to get explanation from LLM in answer
                max_tokens=1,
                temperature=0.03,
            )

            res = answer.choices[0].message.content
            return res
        except Exception as e:
            print(
                f"Error with the LLM for the prompt: \n                     {prompt[:50]}. "
                f"\n                     Error: {e}"
            )
            return None


class llm_wrapper_llama(Llm_wrapper):
    def __init__(self):
        from llama import Llama

        client = Llama.build(
            ckpt_dir="../../llama/llama-2-13b-chat",
            tokenizer_path="../../llama/tokenizer.model",
            max_seq_len=1024,
            max_batch_size=1,
            model_parallel_size=2,
        )
        super().__init__(client)

    def query(self, prompt: str) -> Optional[str]:
        try:
            prompt = PROMPT_LLM_DEEPINTOMLF + prompt
            answer = self.client.chat_completion(
                [
                    [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                ],
                max_gen_len=2,
                temperature=0.03,
            )
            res = answer[0]['generation']['content']
        except Exception as e:
            print(
                f"Error with the LLM for the prompt: \n                     {prompt[:50]}. "
                f"\n                     Error: {e}"
            )
            return None


class llm_wrapper_local_mixstral_8x7B(Llm_wrapper):
    pass


class llm_wrapper_local_command_r(Llm_wrapper):
    pass


###### Register the wrapper here after implementing it ######
# This is a simple factory-pattern implementation. We put a cache on top of it.
REGISTERED_LLM_WRAPPERS = {
    'Mistral': llm_wrapper_mistral,
    'OpenAI': llm_wrapper_openai,
    'LocalMixstral8x7B': llm_wrapper_local_mixstral_8x7B,
    'LocalCommandR': llm_wrapper_local_command_r,
    'Llama': llm_wrapper_llama,
}
##############################################################
