from enum import Enum


# As of 02/05/2024:
# gpt-4,
# gpt-4-turbo-preview,
# gpt-3.5-turbo


class OpenAIModelName(str, Enum):
    # GPT4 = "gpt-4"  # more expensive than GPT-4-turbo as of 05/05/2024.
    GPT4_TURBO = "gpt-4-turbo"
    GPT3_5_TURBO = "gpt-3.5-turbo"
