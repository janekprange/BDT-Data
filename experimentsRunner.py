import os
import pandas as pd
from typing import List, Literal
from experiment.errorDetection import (
    ErrorDetection,
    Flights,
    Food,
    Hospital,
    CustomDataSet,
    DataSet as EDDataset,
)
from experiment.duplicateDetection import (
    DuplicateDetection,
    Affiliation,
    DataSet as DDDataset,
)


def test_errorDetection_customDataset(
    logging_path: str,
    n_iterations=5,
    n_samples=100,
    n_examples=3,
):
    """Excecutes an experiment to test the performance of the error detection on a custom dataset.

    Args:
        logging_path (str): The path where the log files are saved. Skips the experiment if the folder already exists.
        n_iterations (int, optional): The number of times the experiment should be repeated. Defaults to 5.
        n_samples (int, optional): The number of rows that are used in the experiment. Defaults to 100.
        n_examples (int, optional): The number of rows that are used as examples in few shot experiments. Defaults to 3.
    """
    if os.path.exists(logging_path):
        print(
            f"The directory '{logging_path}' already exists, skipping test_prompts_errordetection."
        )
        return
    clean_dataframe = pd.read_csv("data/error_detection/custom/clean_dataframe.csv")
    experiment_datasets: List[dict[Literal["name", "dirty_path"], str]] = [
        {
            "name": "Syntactic Errors",
            "dirty_path": "data/error_detection/custom/dirty_dataframe_typos.csv",
        },
        {
            "name": "Semantic Errors",
            "dirty_path": "data/error_detection/custom/dirty_data_wrong_cities.csv",
        },
        {
            "name": "All Errors",
            "dirty_path": "data/error_detection/custom/dirty_dataframe_all_errors.csv",
        },
    ]

    for _ in range(n_iterations):
        for exp in experiment_datasets:
            dataset = CustomDataSet(
                dirty_data=pd.read_csv(exp["dirty_path"]),
                clean_data=clean_dataframe,
                name="Custom Typos",
            )
            ed = ErrorDetection(dataset=dataset, logging_path=logging_path)
            ed.zero_shot(
                n_samples=n_samples,
                experiment_name=exp["name"],
                experiment_namespace="ErrorDetection.ZeroShot.CustomDataset",
            )
            ed.few_shot(
                n_samples=n_samples,
                experiment_name=exp["name"],
                experiment_namespace="ErrorDetection.FewShot.CustomDataset",
                example_count=n_examples,
                custom_examples="Country: Vietnam, City: Paris, Population: 5253000? Yes\n"
                + "Country: New Zealand, City: Welington, Population: 212700? Yes\n"
                + "Country: Saint Lucia, City: Castries, Population: 200000? Yes\n"
                + "Country: Estonia, City: Montevideo, Population: 1? Yes\n"
                + "Country: Australia, City: Canberra, Population: 395790? No\n",
            )


def test_duplicateDetection_different_grammar(
    logging_path: str,
    n_iterations=3,
    n_rows=50,
    n_duplicates=20,
    chance_multiple_duplicates=0.3,
    dataset: DDDataset = Affiliation(),
):
    """Excecutes an experiment to test the performance duplicate detection with and without grammar.

    Args:
        logging_path (str): The path where the log files are saved. Skips the experiment if the folder already exists.
        n_iterations (int, optional): The number of times the experiment should be repeated. Defaults to 5.
        n_rows (int, optional): The number of rows that are used in the experiment. . Defaults to 50.
        n_duplicates (int, optional): The number of duplicate rows in the data. Defaults to 20.
        chance_multiple_duplicates (float, optional): The chance that a row has multiple duplicates. Defaults to 0.3.
        dataset (DDDataset, optional): A duplicate detection dataset. Defaults to Affiliation().
    """
    if os.path.exists(logging_path):
        print(
            f"The directory '{logging_path}' already exists, skipping test_prompts_errordetection."
        )
        return
    dd = DuplicateDetection(dataset=dataset, logging_path=logging_path)
    for _ in range(n_iterations):
        dd.zero_shot(
            n_samples=n_rows,
            rows_with_duplicates=n_duplicates,
            multiple_duplicate_chance=chance_multiple_duplicates,
            grammar=dd.GRAMMAR_YES_OR_NO,
            experiment_name=f"With Grammar",
            experiment_namespace="DuplicateDetection.ZeroShot.Grammar",
        )
        dd.zero_shot(
            n_samples=n_rows,
            rows_with_duplicates=n_duplicates,
            multiple_duplicate_chance=chance_multiple_duplicates,
            experiment_name=f"Without Grammar",
            experiment_namespace="DuplicateDetection.ZeroShot.Grammar",
        )


def test_errorDetection_prompts(
    logging_path: str,
    n_iterations=5,
    n_samples=100,
    n_examples=3,
    dataset: EDDataset = Flights(),
):
    """Excecute an experiment to test the perfomance of different prompts for error detection.

    Args:
        logging_path (str): The path where the log files are saved. Skips the experiment if the folder already exists.
        n_iterations (int, optional): The number of times the experiment should be repeated. Defaults to 5.
        n_samples (int, optional): The number of rows that are used in the experiment. Defaults to 100.
        n_examples (int, optional): The number of rows that are used as examples in few shot experiments. Defaults to 3.
    """
    if os.path.exists(logging_path):
        print(
            f"The directory '{logging_path}' already exists, skipping test_prompts_errordetection."
        )
        return

    experiment_prompts: List[dict[Literal["namespace", "name", "prompt"], str]] = [
        {
            "name": "Answer either yes or no",
            "prompt": "You are a helpful assistant who is great in finding errors in tabular data. You always answer in a single word, either yes or no.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
        },
        {
            "name": "Answer in a single word",
            "prompt": "You are a helpful assistant who is great in finding errors in tabular data. You always answer in a single word.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
        },
        {
            "name": "Precise and short",
            "prompt": "You are a helpful assistant who is great in finding errors in tabular data. You answer as precise and short as possible.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
        },
        {
            "name": "Deep breath",
            "prompt": "You are a helpful assistant who is great in finding errors in tabular data. Take a deep breath and than answer the following question.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
        },
        {
            "name": "No prompt introduction",
            "prompt": "Q: Is there an error in {attr}?\n{context}\n\nA:",
        },
    ]

    ed = ErrorDetection(dataset=dataset, logging_path=logging_path)
    for _ in range(n_iterations):
        for prompt in experiment_prompts:
            ed.zero_shot(
                n_samples=n_samples,
                prompt_template=prompt["prompt"],
                experiment_name=prompt["name"],
                experiment_namespace="ErrorDetection.ZeroShot.CustomPrompt",
            )
    for _ in range(n_iterations):
        for prompt in experiment_prompts:
            ed.few_shot(
                n_samples=n_samples,
                prompt_template=prompt["prompt"],
                experiment_name=prompt["name"],
                example_count=n_examples,
                experiment_namespace="ErrorDetection.FewShot.CustomPrompt",
            )


if __name__ == "__main__":
    test_errorDetection_customDataset(
        logging_path="logs/ed_customDataset", n_iterations=1, n_samples=5
    )
    # test_duplicateDetection_different_grammar(logging_path="keep_logs/dd_grammar")
    test_errorDetection_prompts(
        logging_path="logs/ed_prompts", n_iterations=1, n_samples=5
    )
