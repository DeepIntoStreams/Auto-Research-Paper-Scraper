from enum import Enum


# As of 02/03/2024:
# open-mistral-7b (aka mistral-tiny-2312)
# open-mixtral-8x7b (aka mistral-small-2312)
# mistral-small-latest (aka mistral-small-2402)
# mistral-medium-latest (aka mistral-medium-2312)
# mistral-large-latest (aka mistral-large-2402)


class MistralModelName(str, Enum):
    TINY = "mistral-tiny-latest"
    SMALL = "mistral-small-latest"
    MEDIUM = "mistral-medium-latest"
    LARGE = "mistral-large-latest"
