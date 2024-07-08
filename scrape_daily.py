import logging.config
import os

import arxiv
import pandas as pd
from dotenv import load_dotenv

from config import QUERY_ARXIV_DEEPINTOMLF, today_date, tomorrow_date, ROOT_DIR
from src.scraping_methods import get_data, scrape_arxiv, llm_filter_on_relevance
from src.utils import factory_fct_linked_path, get_date_range, get_last_date_scraped, rmv_one_day

load_dotenv()  # Load environment variables from .env file

logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "simple": {
                "format": "%(levelname)-8s ~ [%(asctime)s] {%(pathname)s:%(lineno)d} --- %(message)s",
                "datefmt": "%H:%M:%S",
                "style": "%",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    }
)

logger = logging.getLogger(__name__)

path_to_data_scraped = factory_fct_linked_path(ROOT_DIR, "data_scraped")([""])

############ Testing Search ############
## Use the arxiv library to perform the search
NUM_RESULTS = 200

last_scraped_date = get_last_date_scraped(path_to_data_scraped)
QUERY_ARXIV_DEEPINTOMLF += f'AND submittedDate:[{get_date_range((last_scraped_date, tomorrow_date))}]'
print(f"Run the daily scraper on the interval of time: {last_scraped_date} to {tomorrow_date}")

df_data: pd.DataFrame = llm_filter_on_relevance(
    get_data(scrape_arxiv(QUERY_ARXIV_DEEPINTOMLF, arxiv.SortOrder.Descending, NUM_RESULTS))
)

if len(df_data) == 0:
    logger.warning("We have not have found any papers passing the filters, we do not save the data.")
else:
    df_data.to_csv(
        path_to_data_scraped + f"tsgen_papers_{today_date.strftime('%Y_%m_%d')}.csv",
        index=False,
    )

try:
    # We can only print 2 columns by default.
    print(
        df_data[
            [
                "Title",
                "Publication Date",
            ]
        ]
    )
    logger.info(f"Found {len(df_data)} papers")
except Exception as e:
    logger.error(e)
    print(df_data)
