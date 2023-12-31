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

############################################################################################################################################################
    ## Error Detection

# Experiment 5: Different Meta Promts
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

    # TODO: few_shot prompts!
    experiment_prompts: List[
        dict[Literal["namespace", "name", "zeroshot_prompt", "fewshot_prompt"], str]
    ] = [
        {
            "name": "Answer either yes or no",
            "zeroshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. You always answer in a single word, either yes or no.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
            "fewshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. You always answer in a single word, either yes or no.\n\nIs there an error in {attr}?\n{example}\nQ: {context}  A:",
        },
        {
            "name": "Answer in a single word",
            "zeroshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. You always answer in a single word.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
            "fewshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. You always answer in a single word.\n\nIs there an error in {attr}?\n{example}\nQ: {context}  A:",
        },
        {
            "name": "Precise and short",
            "zeroshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. You answer as precise and short as possible.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
            "fewshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. You answer as precise and short as possible.\n\nIs there an error in {attr}?\n{example}\nQ: {context}  A:",
        },
        {
            "name": "Deep breath",
            "zeroshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. Take a deep breath and than answer the following question.\n\nQ: Is there an error in {attr}?\n{context}\n\nA:",
            "fewshot_prompt": "You are a helpful assistant who is great in finding errors in tabular data. Take a deep breath and than answer the following question.\n\nIs there an error in {attr}?\n{example}\nQ: {context}  A:",
        },
        {
            "name": "No prompt introduction",
            "zeroshot_prompt": "Q: Is there an error in {attr}?\n{context}\n\nA:",
            "fewshot_prompt": "Is there an error in {attr}?\n{example}\nQ: {context}  A:",
        },
    ]

    ed = ErrorDetection(dataset=dataset, logging_path=logging_path)
    for _ in range(n_iterations):
        for prompt in experiment_prompts:
            ed.zero_shot(
                n_samples=n_samples,
                prompt_template=prompt["zeroshot_prompt"],
                experiment_name=prompt["name"],
                experiment_namespace="ErrorDetection.ZeroShot.CustomPrompt",
            )
    for _ in range(n_iterations):
        for prompt in experiment_prompts:
            ed.few_shot(
                n_samples=n_samples,
                prompt_template=prompt["fewshot_prompt"],
                experiment_name=prompt["name"],
                example_count=n_examples,
                experiment_namespace="ErrorDetection.FewShot.CustomPrompt",
                q_and_A=True
            )

# Experiment 4: Meta vs. Bloke
def test_errorDetection_Meta_vs_Bloke(
    logging_path: str,
    n_iterations=3,
    n_samples=20,
    n_examples=5,
    skip_meta: bool = False,
    skip_bloke: bool = False
):
    """Excecutes an experiment to test the performance of both the Meta model and the theBloke model.

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
    examples_clean_data = pd.read_csv(
        "data/error_detection/custom/examples_clean_dataframe.csv"
    )
    experiment_datasets: List[
        dict[Literal["name", "dirty_path", "example_path"], str]
    ] = [
        {
            "name": "Typos",
            "dirty_path": "data/error_detection/custom/dirty_dataframe_typos.csv",
            "example_path": "data/error_detection/custom/examples_dataframe_typos.csv",
        },
        {
            "name": "Wrong Cities",
            "dirty_path": "data/error_detection/custom/dirty_data_wrong_cities.csv",
            "example_path": "data/error_detection/custom/examples_dirty_dataframe_wrong_cities.csv",
        },
        {
            "name": "Wrong Population",
            "dirty_path": "data/error_detection/custom/dirty_data_wrong_population.csv",
            "example_path": "data/error_detection/custom/examples_dirty_dataframe_wrong_population.csv",
        },
        {
            "name": "All Errors",
            "dirty_path": "data/error_detection/custom/dirty_dataframe_all_errors.csv",
            "example_path": "data/error_detection/custom/examples_dirty_dataframe_all_errors.csv",
        },
    ]

    # first execute meta entirely
    if(not skip_meta):
        for _ in range(n_iterations):
            for exp in experiment_datasets:
                dataset = CustomDataSet(
                    dirty_data=pd.read_csv(exp["dirty_path"]),
                    clean_data=clean_dataframe,
                    name=exp["name"],
                )
                ed_meta = ErrorDetection(dataset=dataset, logging_path=logging_path)
                ed_meta.zero_shot(
                    n_samples=n_samples,
                    experiment_name=exp["name"],
                    experiment_namespace="Meta.ErrorDetection.ZeroShot.CustomDataset",
                )
                ed_meta.few_shot(
                    n_samples=n_samples,
                    experiment_name=exp["name"],
                    experiment_namespace="Meta.ErrorDetection.FewShot.CustomDataset",
                    example_count=n_examples,
                    custom_examples_dataset=CustomDataSet(
                        dirty_data=pd.read_csv(exp["example_path"]),
                        clean_data=examples_clean_data,
                        name=exp["name"],
                    ),
                    q_and_A = False,
                )
            
    # then execute theBloke entirely
    # NOTICE: the other way around apparently causes errors
    if(not skip_bloke):
        for _ in range(n_iterations):
            for exp in experiment_datasets:
                dataset = CustomDataSet(
                    dirty_data=pd.read_csv(exp["dirty_path"]),
                    clean_data=clean_dataframe,
                    name=exp["name"],
                )
                ed_theBloke = ErrorDetection(dataset=dataset, logging_path=logging_path, model_source="bloke")
                ed_theBloke.zero_shot(
                    n_samples=n_samples,
                    experiment_name=exp["name"],
                    experiment_namespace="Bloke.ErrorDetection.ZeroShot.CustomDataset",
                    grammar=ErrorDetection.GRAMMAR_YES_OR_NO
                )
                ed_theBloke.few_shot(
                    n_samples=n_samples,
                    experiment_name=exp["name"],
                    experiment_namespace="Bloke.ErrorDetection.FewShot.CustomDataset",
                    example_count=n_examples,
                    custom_examples_dataset=CustomDataSet(
                        dirty_data=pd.read_csv(exp["example_path"]),
                        clean_data=examples_clean_data,
                        name=exp["name"],
                    ),
                    q_and_A = False,
                    grammar=ErrorDetection.GRAMMAR_YES_OR_NO
                )

# Experiment 3: Custom Dataset
def test_errorDetection_customDataset(
    logging_path: str,
    n_iterations=3,
    n_samples=20,
    n_examples=5,
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
    examples_clean_data = pd.read_csv(
        "data/error_detection/custom/examples_clean_dataframe.csv"
    )
    experiment_datasets: List[
        dict[Literal["name", "dirty_path", "example_path"], str]
    ] = [
        {
            "name": "Typos",
            "dirty_path": "data/error_detection/custom/dirty_dataframe_typos.csv",
            "example_path": "data/error_detection/custom/examples_dataframe_typos.csv",
        },
        {
            "name": "Wrong Cities",
            "dirty_path": "data/error_detection/custom/dirty_data_wrong_cities.csv",
            "example_path": "data/error_detection/custom/examples_dirty_dataframe_wrong_cities.csv",
        },
        {
            "name": "Wrong Population",
            "dirty_path": "data/error_detection/custom/dirty_data_wrong_population.csv",
            "example_path": "data/error_detection/custom/examples_dirty_dataframe_wrong_population.csv",
        },
        {
            "name": "All Errors",
            "dirty_path": "data/error_detection/custom/dirty_dataframe_all_errors.csv",
            "example_path": "data/error_detection/custom/examples_dirty_dataframe_all_errors.csv",
        },
    ]

    for _ in range(n_iterations):
        for exp in experiment_datasets:
            dataset = CustomDataSet(
                dirty_data=pd.read_csv(exp["dirty_path"]),
                clean_data=clean_dataframe,
                name=exp["name"],
            )
            ed_theBloke = ErrorDetection(dataset=dataset, logging_path=logging_path, model_source="bloke")
            ed_theBloke.zero_shot(
                n_samples=n_samples,
                experiment_name=exp["name"],
                experiment_namespace="Bloke.ErrorDetection.ZeroShot.CustomDataset",
                grammar=ErrorDetection.GRAMMAR_YES_OR_NO
            )
            ed_theBloke.few_shot(
                n_samples=n_samples,
                experiment_name=exp["name"],
                experiment_namespace="Bloke.ErrorDetection.FewShot.CustomDataset",
                example_count=n_examples,
                custom_examples_dataset=CustomDataSet(
                    dirty_data=pd.read_csv(exp["example_path"]),
                    clean_data=examples_clean_data,
                    name=exp["name"],
                ),
                q_and_A = False,
                grammar=ErrorDetection.GRAMMAR_YES_OR_NO
            )


############################################################################################################################################################
    ## Duplicate Detection

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


if __name__ == "__main__":
    test_errorDetection_Meta_vs_Bloke(logging_path="keep_logs/test_ed_Meta_vs_Bloke_only_Bloke", skip_meta=True)
    # test_errorDetection_customDataset(logging_path="keep_logs/test_ed_customDataset")
    # test_duplicateDetection_different_grammar(logging_path="keep_logs/dd_grammar")
    # test_errorDetection_prompts(logging_path="keep_logs/ed_prompts")
