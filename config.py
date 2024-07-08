import os
from datetime import datetime, timedelta

from src.mistralmodels import MistralModelName
from src.openaimodels import OpenAIModelName

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

today_date = datetime.now()
tomorrow_date = today_date + timedelta(days=1)
rmv_one_year = lambda date: date.replace(year=date.year - 1)


MAX_PAPERS_PER_QUERY = 1_000  # safe bound on how many papers we fetch per query.


ENABLE_LLM_FILTER = False
# This string needs to match one of the keys of REGISTERED_LLM_WRAPPERS.
LLM_WRAPPER_CHOICE = 'Llama'
MODEL_LLM = None
### or use this
# LLM_WRAPPER_CHOICE = 'Mistral'
# MODEL_LLM = MistralModelName.LARGE
# LLM_WRAPPER_CHOICE = 'OpenAI'
# MODEL_LLM = OpenAIModelName.GPT3_5_TURBO


# Query details: https://info.arxiv.org/help/api/user-manual.html#query_details
# categories : https://arxiv.org/category_taxonomy
QUERY_ARXIV_DEEPINTOMLF = (
    '(all:"Variational" '
    'OR all:GAN '
    'OR all:CGAN '
    'OR all:GANs '
    'OR all:Generative '
    'OR all:"Generative Adversarial Network" '
    'OR all:"sequence modelling"'
    ') '
    'AND (all:"time series"  '
    'OR all:"time-series" '
    'OR all:"point processes" '
    'OR all:"sequences" '
    'OR all:"sequential data" '
    'OR all:"streams" '
    'OR all:RNN '
    'OR all:LSTM '
    'OR all:GRU '
    'OR all:"market generator" '
    'OR all:"simulator" '
    ') '
    'AND '
    '(cat:cs.LG '
    # No paper seem to be relevant to us there.
    # 'OR cat:cs.CE '
    'OR cat:cs.AI '
    'OR cat:q-fin.CP '
    'OR cat:q-fin.MF '
    'OR cat:stat.ML'
    ')'
)

PROMPT_LLM_DEEPINTOMLF = """After the placeholder -START-, I will share the title and abstract of an academic paper, and I want you to determine whether it pertains to time series generation. Specifically, assess whether the core objective and methodology revolve around generating temporal sequences or time series or surveys on that topic. We are only interested in papers explicitly developing new or studying existing generative models for temporal data including studies and surveys reviewing existing methods. This does not include forecasting tasks that are not generative-based, classification problems or generation of images. Your reply must be: TRUE or FALSE. Be mindful and don't guess, give me a confident and correct answer in one word, thank you and be as accurate as possible, I will tip you if you do well. \n -START- \n"""
