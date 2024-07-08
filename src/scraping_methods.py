import functools
import typing

import arxiv
import pandas as pd
from tqdm import tqdm

from config import PROMPT_LLM_DEEPINTOMLF, ENABLE_LLM_FILTER, LLM_WRAPPER_CHOICE
from src.llm_wrapper import REGISTERED_LLM_WRAPPERS


def get_data(query_results: typing.List, verbose: bool = True) -> pd.DataFrame:
    """
    Extracts data from query results and returns it as a pandas DataFrame.

    This function iterates over a list of query results, extracting relevant information from each result: the title,
    publication date, authors, abstract, link to the PDF, and the DOI. It then compiles this information into a pandas
    DataFrame for easy manipulation and analysis.

    Parameters:
    - query_results (typing.List): A list of query result objects. Each query result object is expected to have the
    following attributes: title, published, authors, summary, pdf_url, and doi. This parameter should ideally come from
    `scrape_arxiv`.
    - verbose (bool, optional): If True (default), shows a progress bar indicating the progress of data extraction using
     tqdm. If False, the progress bar is suppressed.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the extracted data, with columns for "Title", "Publication Date",
     "Author(s)", "Abstract", "Link", and "DOI".
    """
    data = []
    for result in tqdm(query_results, disable=not verbose):
        title = result.title
        publication_date = result.published.strftime("%d/%m/%Y")
        authors = ", ".join([author.name for author in result.authors])
        abstract = result.summary
        link = result.pdf_url
        doi = result.doi
        data.append([title, publication_date, authors, abstract, link, doi])
    df = pd.DataFrame(data, columns=["Title", "Publication Date", "Author(s)", "Abstract", "Link", "DOI"])
    if len(df) == 0:
        print("No papers found with arxiv filter!")
    return df


def scrape_arxiv(
    query: str,
    sort_order: arxiv.SortOrder,
    max_results: int,
):
    """
    Searches arXiv for papers matching a given query and returns the results.

    This function performs a search on the arXiv platform using specified criteria,
    including a search query, the sort order for results, and the maximum number of
    results to return. It utilizes the arxiv library to facilitate the search.

    Parameters:
    - query (str): The search query string.
    - sort_order (arxiv.SortOrder): The order in which to sort the results, e.g., ascending or descending.
    - max_results (int): The maximum number of search results to return.

    Returns:
    - list: A list of search result objects, each representing a paper found on arXiv
      matching the search criteria.

    Examples:
    >>> query_results = scrape_arxiv("quantum computing", arxiv.SortOrder.Descending, 10)
    """
    search = arxiv.Search(
        query=query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=sort_order,
        max_results=max_results,
    )

    query_results: list = list(arxiv.Client().results(search))
    return query_results


@functools.lru_cache(maxsize=None)
def get_client(llm_wrapper_choice):
    return REGISTERED_LLM_WRAPPERS[llm_wrapper_choice]()


def scrape_llm(
    paper_sum_as_str: str,
) -> bool:
    """
    This function takes a string summarizing a paper and uses a pre-defined prompt
    to query a language model. The function expects a specific environment variable
    to be set for the API key required to interact with the LLM. It returns a boolean
    value indicating the LLM's assessment of the paper's significance.

    Parameters:
    - paper_sum_as_str (str): A string summarizing the paper to be evaluated.

    Returns:
    - bool: True if the LLM considers the paper significant, False otherwise. Returns
      False also if there's an error in the process or the API key is missing.
    """
    try:
        # We use a cache to avoid reintializing the client at each call.
        client = get_client(LLM_WRAPPER_CHOICE)

        prompt = PROMPT_LLM_DEEPINTOMLF + paper_sum_as_str

        # Some cleaning of the answer
        answer = client.query(prompt).lstrip().upper()

        # We exclude the case when both appear.
        if 'TRUE' in answer and not ('FALSE' in answer):
            return True
        elif 'FALSE' in answer and not ('TRUE' in answer):
            return False
        else:
            print(
                f"Unexpected response from the LLM: '{answer}' for the paper: \n                     {paper_sum_as_str[:50]}"
            )
            return False
    except Exception as e:
        print(f"Error with the LLM for the paper {paper_sum_as_str[:50]}: {e}")
        return False


def llm_filter_on_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters articles in the DataFrame based on their relevance as determined by the `scrape_llm` function.

    Parameters:
    - df (pd.DataFrame): DataFrame containing articles with columns "Title" and "Abstract".

    Returns:
    - pd.DataFrame: Filtered DataFrame containing only relevant articles.
    """
    # Add a new column to mark relevance, initialize as False
    df['Relevant'] = False

    num_papers_before_filter_llm = len(df)

    if not ENABLE_LLM_FILTER:
        print("LLM filter is disabled (check parameter ENABLE_LLM_FILTER), returning the original DataFrame.")
        return df

    for index, row in df.iterrows():
        title_abstract = f"{row['Title']}. {row['Abstract']}"
        if scrape_llm(title_abstract):
            df.at[index, 'Relevant'] = True

    # Extract relevant articles
    relevant_articles = df[df['Relevant']]

    # Extract and print discarded articles
    discarded_articles = df[~df['Relevant']]
    if len(discarded_articles) > 0:
        print(
            f"Reduced with an LLM from {num_papers_before_filter_llm} to {len(relevant_articles)} papers. Discarded: "
        )
        print(discarded_articles[['Title', 'Publication Date']])
    else:
        print("No papers were discarded by the LLM.")

    # Return the DataFrame without the 'Relevant' column for further processing
    return relevant_articles.drop(columns=['Relevant'])
