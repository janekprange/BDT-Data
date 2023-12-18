from typing import Tuple, List, Union, Literal
import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display
from datetime import datetime
from ..setupExperiment import SetupExperiment, serialize_row
from .dataset import DataSet
from ..experimentLogger import Logger
from sklearn.metrics import f1_score
import time
from llama_cpp.llama_grammar import LlamaGrammar


class ErrorDetection(SetupExperiment):
    """A class that wraps the error detection experiments."""

    GRAMMAR_YES_OR_NO = SetupExperiment.GRAMMAR_YES_OR_NO

    def __init__(
        self,
        dataset: DataSet,
        skip_prompting: bool = False,
        logging_path: str = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        model_size: Literal["small", "medium", "large"] = "small",
    ) -> None:
        """
        A class that wraps the error detection experiments.

        Args:
            dataset (DataSet): The dataset that is used for the experiments,
            skip_prompting (bool, optional): If True, Llama is neither initialized nor prompted. Defaults to False.
            logging_path (str, optional): The path the logging files are written to. Defaults to f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}".
            model_size (Literal[&quot;small&quot;, &quot;medium&quot;, &quot;large&quot;], optional): The model is available in different sizes/complexities. With a larger model comes a better quality and a worse runtime. Defaults to "large".
        """
        super().__init__(skip_prompting, model_size=model_size)
        self.dataset = dataset
        self.logger = Logger(name="ErrorDetection", path=logging_path)

    def _execute(
        self,
        dirty_data: pd.DataFrame,
        clean_data: pd.DataFrame,
        prompt_template: str,
        experiment_name: str,
        experiment_namespace: str,
        dataset_name: str,
        generated_example_count: int = 0,
        custom_examples_dataset: Union[DataSet, None] = None,
        log_id: Union[int, str] = "",
        grammar: Union[LlamaGrammar, None] = None,
    ) -> Tuple[float, float]:
        n_samples = len(dirty_data)
        progressBar = IntProgress(
            min=0,
            max=min(dirty_data.shape[0], n_samples) * dirty_data.shape[1],
            description=f"Attributes Prompted {experiment_name}",
        )
        display(progressBar)

        result = {
            "true_pos": 0,
            "false_pos": 0,
            "true_neg": 0,
            "false_neg": 0,
        }
        y_true: List[int] = []
        y_pred: List[int] = []
        start_time = time.time()

        for row_index, row in dirty_data.iterrows():
            serialized_row = serialize_row(row)
            for column, (attribute, value) in enumerate(row.items()):
                # create the promt with custom examples
                if custom_examples_dataset is not None:
                    # we want to generate the examples with all the row of the example dataset
                    all_rows_count = custom_examples_dataset.clean_set.shape[0]
                    examples = custom_examples_dataset.generate_examples(
                        column_id=column, amount=all_rows_count
                    )
                    prompt = prompt_template.format(
                        attr=attribute, context=serialized_row, example=examples
                    )
                # otherwise create promt with examples if needed
                elif generated_example_count > 0:
                    examples = self.dataset.generate_examples(
                        column_id=column, amount=generated_example_count
                    )
                    prompt = prompt_template.format(
                        attr=attribute, context=serialized_row, example=examples
                    )
                
                # no examples
                else:
                    prompt = prompt_template.format(
                        attr=attribute, val=value, context=serialized_row
                    )

                correct_value: bool = (
                    clean_data.loc[[row_index]][attribute].values[0] != value
                )
                y_true.append(int(correct_value))
                timestamp = int(time.time_ns() / 10**6)
                response = self._prompt(
                    prompt,
                    id=f"{log_id}-{timestamp}",
                    logger=self.logger,
                    correct_answer=correct_value,
                    grammar=grammar,
                )

                # evaluate response
                if "Yes" in response or "yes" in response:
                    self.logger.log_prompting_result(
                        id=f"{log_id}-{timestamp}",
                        predicted=1,
                        correct=int(correct_value),
                    )
                    y_pred.append(1)
                    if correct_value:
                        result["true_pos"] += 1
                    else:
                        result["false_pos"] += 1
                else:
                    self.logger.log_prompting_result(
                        id=f"{log_id}-{timestamp}",
                        predicted=0,
                        correct=int(correct_value),
                    )
                    y_pred.append(0)
                    if correct_value:
                        result["false_neg"] += 1
                    else:
                        result["true_neg"] += 1

                progressBar.value += 1
        runtime = time.time() - start_time
        runtimeString = time.strftime("%H:%M:%S", time.gmtime(runtime))
        f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1))
        self.logger.log_experiment_result(
            name=experiment_name,
            namespace=experiment_namespace,
            runtime=runtimeString,
            n_rows=n_samples,
            n_examples=generated_example_count,
            dataset=dataset_name,
            f1=f1,
            true_pos=result["true_pos"],
            true_neg=result["true_neg"],
            false_pos=result["false_pos"],
            false_neg=result["false_neg"],
            prompt=prompt_template,
        )
        self.logger.info(
            f"Finished {experiment_name} in {time.strftime('%H:%M:%S', time.gmtime(runtime))}"
        )
        return runtime, f1

    def zero_shot(
        self,
        data_indices: List[int] | None = None,
        prompt_template: str = "Is there an error in {attr}?\n{context}?",
        n_samples: int = 100,
        log_id: Union[int, str] = "",
        grammar: Union[LlamaGrammar, None] = None,
        experiment_name="",
        experiment_namespace="ErrorDetection.ZeroShot",
    ) -> Tuple[float, float]:
        """Execute a zero shot experiment on the dataset the class was initialized with.

        Args:
            data_indices (List[int] | None, optional): A list of indeces of rows in the dataset that will be used for the experiment. Defaults to None.
            prompt_template (str, optional): This template will be used to prompt the model. It will be formatted with `attr` (the name of the column that is checked for an error) and `context` (the row that is currently evaluated). Defaults to "Is there an error in {attr}?\\n{context}?".
            n_samples (int, optional): The number of rows to use for the experiment. Defaults to 100.
            id (Union[int, str], optional): An id that is used to identify the logging files belonging to this experiment. Defaults to "".

        Returns:
            Tuple[float, float]: runtime and f1 of the experiment
        """

        self.logger.info(f"Started zero shot for {n_samples} rows")
        dirty_data: pd.DataFrame
        clean_data: pd.DataFrame
        if data_indices is None:
            dirty_data, clean_data = self.dataset.random_sample(n_samples)
        else:
            n_samples = len(data_indices)
            dirty_data = self.dataset.get(dirty=True).iloc(data_indices)
            clean_data = self.dataset.get(dirty=False).iloc(data_indices)

        return self._execute(
            dirty_data=dirty_data,
            clean_data=clean_data,
            prompt_template=prompt_template,
            experiment_name=experiment_name,
            experiment_namespace=experiment_namespace,
            dataset_name=self.dataset.name,
            generated_example_count=0,
            log_id=f"ed_zs{log_id}",
            grammar=grammar,
        )

    def few_shot(
        self,
        data_indices: List[int] | None = None,
        prompt_template: str = "Is there an error in {attr}?\n\n{example}\n\n{context}?",
        n_samples: int = 100,
        example_count: int = 2,
        log_id: Union[int, str] = "",
        grammar: Union[LlamaGrammar, None] = None,
        custom_examples_dataset: Union[DataSet, None] = None,
        experiment_name="",
        experiment_namespace="ErrorDetection.FewShot",
    ) -> Tuple[float, float]:
        """Execute a few shot experiment on the dataset the class was initialized with.

        Args:
            data_indices (List[int] | None, optional): A list of indeces of rows in the dataset that will be used for the experiment. Defaults to None.
            prompt_template (str, optional): This template will be used to prompt the model. It will be formatted with `attr` (the name of the column that is checked for an error), `example` (example rows with the expected answer) and `context` (the row that is currently evaluated). Defaults to "Is there an error in {attr}?\\n\\n{example}\\n\\n{context}?".
            n_samples (int, optional): The number of rows to use for the experiment. Defaults to 100.
            example_count (int, optional): The number of rows (with the expected answer) the model is given. Defaults to 2.
            id (Union[int, str], optional): An id that is used to identify the logging files belonging to this experiment. Defaults to "".

        Returns:
            Tuple[float, float]: runtime and f1 of the experiment
        """
        self.logger.info(
            f"Started few shot for {n_samples} rows with {example_count} examples"
        )
        dirty_data: pd.DataFrame
        clean_data: pd.DataFrame
        if data_indices is None:
            dirty_data, clean_data = self.dataset.random_sample(n_samples)
        else:
            n_samples = len(data_indices)
            dirty_data = self.dataset.get(dirty=True).iloc(data_indices)
            clean_data = self.dataset.get(dirty=False).iloc(data_indices)

        return self._execute(
            dirty_data=dirty_data,
            clean_data=clean_data,
            prompt_template=prompt_template,
            experiment_name=experiment_name,
            experiment_namespace=experiment_namespace,
            dataset_name=self.dataset.name,
            generated_example_count=example_count,
            log_id=f"ed_fs{log_id}",
            grammar=grammar,
            custom_examples_dataset=custom_examples_dataset,
        )
