"""
This Python script automates the collection and filtration of papers from arXiv over the last year.
It dynamically queries for papers, applies a relevance filter, and outputs the findings into a CSV file named
papers_{start_date}_to_{end_date}.csv, indicating the collection period.

Variables to Modify:

    ARXIV_QUERY: The base query for fetching papers from arXiv, which should be tailored to your specific research area.
    MAX_PAPERS_PER_QUERY: The maximum number of papers to fetch per query to arXiv.
    Environment variables in the .env file for API keys or any other sensitive/configurable information.
"""

import os
from datetime import datetime

import arxiv
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import QUERY_ARXIV_DEEPINTOMLF, today_date, MAX_PAPERS_PER_QUERY, ROOT_DIR
from config import tomorrow_date
from src.scraping_methods import get_data, scrape_arxiv, llm_filter_on_relevance
from src.utils import get_date_range, factory_fct_linked_path, rmv_one_month
import logging.config

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

total_query_results = []
next_iter_date: datetime = tomorrow_date
# Over the last 10 years, collect the papers.
for _ in tqdm(range(0, 12 * 1), desc="Collecting papers.", unit="month"):
    loop_current_date_max: datetime = next_iter_date
    next_iter_date: datetime = rmv_one_month(loop_current_date_max)
    query_year = (
        QUERY_ARXIV_DEEPINTOMLF + f'AND submittedDate:[{get_date_range((next_iter_date, loop_current_date_max))}]'
    )

    query_results: list = scrape_arxiv(query_year, arxiv.SortOrder.Descending, MAX_PAPERS_PER_QUERY)

    if len(query_results) == 0:
        logger.warning("No results found.")
    elif len(query_results) == MAX_PAPERS_PER_QUERY:
        logger.warning(f"Max number of paper reached for the query: {query_year}, consider changing the query.")
    total_query_results += query_results
    ### End loop

df_data: pd.DataFrame = get_data(total_query_results)
df_data: pd.DataFrame = llm_filter_on_relevance(df_data)

df_data.to_csv(
    path_to_data_scraped
    + f"tsgen_papers_{next_iter_date.strftime('%Y_%m_%d')}_TO_{today_date.strftime('%Y_%m_%d')}.csv",
    index=False,
)
logger.info(
    f"~~~~~~Dataframe saved at tsgen_papers_{next_iter_date.strftime('%Y_%m_%d')}_TO_{today_date.strftime('%Y_%m_%d')}.csv.~~~~~~"
)
