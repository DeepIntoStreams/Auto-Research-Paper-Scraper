import os
import unittest

import arxiv  # Assuming this is the module you use for sorting order
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import MistralClient

from config import QUERY_ARXIV_DEEPINTOMLF
from src.scraping_methods import scrape_arxiv, get_data, scrape_llm


class TestArxivScraping(unittest.TestCase):

    def test_query_results_for_matching_papers(self):
        """Test that our query does not filter out at the arxiv level the desired papers."""
        df = pd.read_csv('../data_tests/matching_papers.csv', dtype=str)

        # Filter rows where 'Arxiv Code' exists
        df_filtered = df.dropna(subset=['Arxiv Code'])

        # To store errors
        errors = []

        # Test each row
        for index, row in df_filtered.iterrows():
            arxiv_code = row['Arxiv Code']
            title = row['Title']
            query = QUERY_ARXIV_DEEPINTOMLF + f'AND id:{arxiv_code}'

            try:
                df_data = get_data(scrape_arxiv(query, arxiv.SortOrder.Descending, 1), False)

                # Check if data length is 1 and title matches
                if len(df_data) != 1 or df_data.iloc[0]['Title'].upper() != title.upper():
                    # Remove expected errors:
                    if arxiv_code in []:
                        continue
                    errors.append(
                        f"Row {index + 1} - Title: \"{title}\" with code {arxiv_code} did not match expected result: {df_data.iloc[0]['Title'] if len(df_data) > 0 else 'No data found'}"
                    )
            except Exception as e:
                errors.append(
                    f"Row {index + 1} - Title: \"{title}\" with code {arxiv_code} failed to query or process: {e}"
                )

        if len(errors) > 0:
            error_message = (
                f"In total, there was {len(errors)} papers with issues. Errors occurred with the following papers:\n"
                + "\n".join(errors)
            )
            self.assertTrue(len(errors) == 0, error_message)

    def test_query_results_for_not_matching_papers(self):
        """Check that a paper does not appear as accepted. This can happen when the pipeline is wrong or has a typo."""
        df = pd.read_csv('../data_tests/not_matching_papers.csv', dtype=str)
        arxiv_codes = df['Arxiv Code']
        for arxiv_code in arxiv_codes:
            query = QUERY_ARXIV_DEEPINTOMLF + f'AND id:{arxiv_code}'

            df_data = get_data(scrape_arxiv(query, arxiv.SortOrder.Descending, 1), False)
            self.assertTrue(len(df_data) == 0, f"Paper with code {arxiv_code} should not appear in the query results.")

    def test_llm_filter_correctly_case_not_matching(self):
        ####### USES LLM API KEY and induces a cost.

        load_dotenv()  # Load environment variables from .env file

        df = pd.read_csv('../data_tests/not_matching_papers.csv', dtype=str)
        titles = df['Title']
        abstracts = df['Abstract']

        errors = []
        for title, abstract in zip(titles, abstracts):
            #### To investigate one specific example
            # if title != "":
            #     continue

            is_relevant = scrape_llm(f"{title}. {abstract}")
            # We expect all the papers to NOT be relevant.
            if is_relevant:
                errors.append(f"Paper with title {title} should not appear in the query results.")

        if len(errors) > 0:
            error_message = (
                f"In total, there was {len(errors)} papers with issues. Errors occurred with the following papers:\n"
                + "\n".join(errors)
            )
            self.assertTrue(len(errors) == 0, error_message)

    def test_llm_filter_correctly_case_matching(self):
        ####### USES LLM API KEY and induces a cost.

        load_dotenv()  # Load environment variables from .env file

        df = pd.read_csv('../data_tests/matching_papers.csv', dtype=str)
        titles = df['Title']
        abstracts = df['Abstract']

        errors = []
        for i, (title, abstract) in enumerate(zip(titles, abstracts)):
            is_relevant = scrape_llm(f"{title}. {abstract}")
            # We expect all the papers to NOT be relevant.
            if not is_relevant:
                errors.append(f"Paper with title {title} should not appear in the query results.")

        if len(errors) > 0:
            error_message = (
                f"In total, there was {len(errors)} papers with issues. Errors occurred with the following papers:\n"
                + "\n".join(errors)
            )
            self.assertTrue(len(errors) == 0, error_message)

    def test_mistral_key_works(self):
        load_dotenv()  # Load environment variables from .env file
        self.assertTrue("MISTRAL_API_KEY" in os.environ)

        MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
