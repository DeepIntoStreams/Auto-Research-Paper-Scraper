"""
Example of how to scrape arxiv and get the data from the papers.
 The script defines a search query incorporating a dynamic date range, retrieves and filters results using a
 relevance-based language model, and saves the filtered data to a CSV file. Additionally, it handles exceptions and
 provides a summary of the retrieved papers.
"""

import logging.config
import os
from datetime import datetime

import arxiv
from dotenv import load_dotenv

from config import QUERY_ARXIV_DEEPINTOMLF, tomorrow_date, ROOT_DIR
from src.scraping_methods import get_data, scrape_arxiv, llm_filter_on_relevance
from src.utils import factory_fct_linked_path, get_date_range

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
NUM_RESULTS = 1000
# last_query_date: datetime = datetime(2000, 12, 20)
# current_query_date: datetime = datetime(2022, 12, 31)
# current_query_date = tomorrow_date
# query += f"AND submittedDate:[{get_date_range((last_query_date, current_query_date))}]"

last_query_date: datetime = datetime(2023, 11, 10)
current_query_date = datetime(2023, 11, 25)
QUERY_ARXIV_DEEPINTOMLF += f'AND submittedDate:[{get_date_range((last_query_date, current_query_date))}]'

df_data = llm_filter_on_relevance(
    get_data(scrape_arxiv(QUERY_ARXIV_DEEPINTOMLF, arxiv.SortOrder.Descending, NUM_RESULTS))
)


df_data.to_csv(
    path_to_data_scraped + f"tsgen_papers_test.csv",
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
    print(f"Found {len(df_data)} papers")
except Exception as e:
    logger.error(e)
    print(df_data)


# Do a script that takes the whole data set and fills a column for the model.
# Use the results to plot a confusion matrix.
