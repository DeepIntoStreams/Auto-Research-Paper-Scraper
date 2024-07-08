import os
import typing
from datetime import datetime, timedelta

from config import today_date


def get_date_range(date_range: typing.Tuple[datetime, datetime]) -> str:
    return f"{date_range[0].strftime('%Y%m%d')} TO {date_range[1].strftime('%Y%m%d')}"


def get_last_date_scraped(path_to_files: str) -> datetime:
    # List all files in the directory
    files = os.listdir(path_to_files)
    print("Analysing previous scraped data files to deduce when was the last process ran. ")
    # Process each file
    dates = []
    for file in files:
        # Check if the file follows the naming convention
        if file.startswith('tsgen_papers_') and file.endswith('.csv'):
            try:
                # Extract the date part of the filename
                # Assumes the format is 'tsgen_papers_YYYY_MM_DD.csv'
                date_str = file[len('tsgen_papers_') :].split('.')[0]

                # Convert the date string to a datetime object
                datetime_instance = datetime.strptime(date_str, '%Y_%m_%d')

                # Add the datetime object to the list of dates
                dates.append(datetime_instance)
            except (IndexError, ValueError):
                # Skip files that do not conform to the expected date format
                continue

    # Find the most recent date
    most_recent_date = max(dates) if dates else None

    if most_recent_date:
        print(f"The most recent date is: {most_recent_date.strftime('%Y-%m-%d')}")
        return most_recent_date
    else:
        print("No valid files found, return yesterday.")
        return rmv_one_day(today_date)


def factory_fct_linked_path(ROOT_DIR, path_to_folder):
    """
    Semantics:

    Args:
        ROOT_DIR: path to the root of the project.
        path_to_folder: a path written in the format you want because we use the function os.path.join to link it.

    Returns:
        The linker
    Examples:
              linked_path = factory_fct_linked_path(ROOT_DIR, "path/a"):
              path_save_history = linked_path(['plots', f"best_score_{nb}.pth"])
              #and ROOT_DIR should be imported from a script at the root where it is written:

              import os
              ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    """
    # example:

    PATH_TO_ROOT = os.path.join(ROOT_DIR, path_to_folder)

    def linked_path(path):
        # a list of folders like: ['C','users','name'...]
        # when adding a '' at the end like
        #       path_to_directory = linker_path_to_result_file([path, ''])
        # one adds a \ at the end of the path. This is necessary in order to continue writing the path.
        return os.path.join(PATH_TO_ROOT, *path)

    return linked_path


rmv_one_day = lambda date: date - timedelta(days=1)


def rmv_one_month(date):
    # Function to find the last day of a month for a given year and month
    def last_day_of_month(year, month):
        if month == 12:  # December to January
            next_month_first_day = datetime(year + 1, 1, 1)
        else:
            next_month_first_day = datetime(year, month + 1, 1)
        return (next_month_first_day - timedelta(days=1)).day

    new_year = date.year - (1 if date.month == 1 else 0)  # Decrement year if January
    new_month = 12 if date.month == 1 else date.month - 1  # Wrap around to December if January
    new_day = min(date.day, last_day_of_month(new_year, new_month))  # Adjust day if necessary

    return datetime(new_year, new_month, new_day)
