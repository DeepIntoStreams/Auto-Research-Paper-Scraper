import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from config import ROOT_DIR
from src.scraping_methods import scrape_llm
from src.utils import factory_fct_linked_path
from tests.drawconfusionmatrix import draw_confusion_matrix

load_dotenv()  # Load environment variables from .env file

# The prompt used for benchmark of GPT-3.5 and GPT-4:
# PROMPT_LLM_DEEPINTOMLF = """After the placeholder -START-, I will share the title and abstract of an academic paper, and I want you to determine whether it
# pertains to time series generation. Specifically, assess whether the core objective and methodology revolve around generating temporal sequences or time
# series or surveys on that topic. We are only interested in papers explicitly developing new or studying existing generative models for temporal data including
# studies and surveys reviewing existing methods. This does not include forecasting tasks that are not generative-based, classification problems or generation
# of images. Your reply must be: TRUE or FALSE. Be mindful and don't guess, give me a confident and correct answer in one word, thank you and be as accurate as
# possible, I will tip you if you do well. \n -START- \n"""

####### By deactivating this, we prevent changing any data.
# Configuration to annotate the benchmark data or not
ANNOTATE_BENCHMARK_DATA = False
# Name of column in the CSV file where the LLM results are saved
COLUMN_CSV_WHERE_SAVED = "Label Llama"


path_data_benchmark = factory_fct_linked_path(ROOT_DIR, "data_tests")(["benchmark_papers.csv"])

try:
    papers_benchmark = pd.read_csv(path_data_benchmark)
except FileNotFoundError:
    raise Exception("The specified file path does not exist.")

if ANNOTATE_BENCHMARK_DATA:
    # Update DataFrame with results from LLM scraping
    def annotate_row(row):
        """Scrape using LLM and annotate the DataFrame row."""
        result = scrape_llm(f"{row['Title']}. {row['Abstract']}")
        return 'Y' if result else 'N'

    tqdm.pandas(desc="Annotating Data")
    ### TO CHANGE MODEL AND WHERE DATA IS SAVED, CHANGE HERE ###
    papers_benchmark[COLUMN_CSV_WHERE_SAVED] = papers_benchmark.progress_apply(annotate_row, axis=1)
    papers_benchmark.to_csv(path_data_benchmark, index=False)

### TO CHANGE MODEL AND WHERE DATA IS SAVED, CHANGE HERE ###
rows_false_filtering = papers_benchmark[papers_benchmark['True Label'] != papers_benchmark[COLUMN_CSV_WHERE_SAVED]]

# Print statistics
print("\nStatistics:")
print(f"Total number of rows: {papers_benchmark.shape[0]}")
print(f"Counts of True Label categories:\n{papers_benchmark['True Label'].value_counts()}")
print(f"Counts of Label GPT-3.5 categories:\n{papers_benchmark['Label GPT-3.5'].value_counts()}")
print(f"Counts of Label GPT-4 categories:\n{papers_benchmark['Label GPT-4'].value_counts()}")


# Print titles of papers where labels disagree, if any.
if not rows_false_filtering.empty:
    ### TO CHANGE MODEL AND WHERE DATA IS SAVED, CHANGE HERE ###
    print("Titles of papers with label disagreements (true vs predicted) for model choice:")
    for title in rows_false_filtering['Title']:
        print("      " + title)

# Generate and display the confusion matrix
LABELS_FILTER = ['Filtered-Out', 'Keep the Paper']  # the order of the labels is given by the first paper's label in DB.
assessed_models = ['GPT-3.5', 'GPT-4', 'Llama']
for assessed_model in assessed_models:
    print(f"{'-' * 30}")
    print(f"\nAssessing {assessed_model} model")
    print(f"{'-' * 30}")
    draw_confusion_matrix(
        papers_benchmark['True Label'],
        papers_benchmark['Label ' + assessed_model],
        LABELS_FILTER,
        assessed_model + " predictions",
    )

    binary_true_labels = (papers_benchmark['True Label'] == 'Y').astype(int)
    binary_predicted_labels = (papers_benchmark['Label ' + assessed_model] == 'Y').astype(int)

    accuracy = accuracy_score(binary_true_labels, binary_predicted_labels)
    precision = precision_score(binary_true_labels, binary_predicted_labels)
    recall = recall_score(binary_true_labels, binary_predicted_labels)
    f1 = f1_score(binary_true_labels, binary_predicted_labels)
    roc_auc = roc_auc_score(binary_true_labels, binary_predicted_labels)

    # Print the scores
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

plt.show()
