from typing import Tuple, List
import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display
from datetime import datetime
from .setupExperiment import SetupExperiment
from DataSet import DataSet, serialize_row
from .experimentLogger import Logger
from sklearn.metrics import f1_score
import time


class ErrorDetection(SetupExperiment):
    def __init__(
        self,
        dataset: DataSet,
        skip_prompting: bool = False,
        logging_path: str = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    ) -> None:
        super().__init__(skip_prompting)
        self.dataset = dataset
        self.logger = Logger(name="ErrorDetection", path=logging_path)

    def zero_shot(
        self,
        data_indeces: List[int] | None = None,
        prompt_template: str = "Is there an error in {attr}?\n{context}?",
        n_samples: int = 100,
    ) -> Tuple[float, float]:
        self.logger.info("Started zero shot for {n_samples} rows")
        dirty_data: pd.DataFrame
        clean_data: pd.DataFrame
        if data_indeces is None:
            dirty_data, clean_data = self.dataset.random_sample(n_samples)
        else:
            n_samples = len(data_indeces)
            dirty_data = self.dataset.get(dirty=True).iloc(data_indeces)
            clean_data = self.dataset.get(dirty=False).iloc(data_indeces)

        progress_bar = IntProgress(
            min=0,
            max=min(dirty_data.shape[0], n_samples) * dirty_data.shape[1],
            description="Attributes Prompted Error Detection Zero Shot",
        )
        display(progress_bar)

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
            for _, (attribute, value) in enumerate(row.items()):
                # create prompt
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
                    id=f"ed_zs-{timestamp}",
                    logger=self.logger,
                )

                # evaluate response
                if "Yes" in response or "yes" in response:
                    self.logger.log_prompting_result(
                        id=f"ed_zs-{timestamp}",
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
                        id=f"ed_zs-{timestamp}",
                        predicted=0,
                        correct=int(correct_value),
                    )
                    y_pred.append(0)
                    if correct_value:
                        result["false_neg"] += 1
                    else:
                        result["true_neg"] += 1

                progress_bar.value += 1
        runtime = time.time() - start_time
        runtimeString = time.strftime("%H:%M:%S", time.gmtime(runtime))
        f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1))
        self.logger.log_experiment_result(
            name="Error Detection zero shot",
            runtime=runtimeString,
            n_rows=n_samples,
            dataset=self.dataset.name,
            f1=f1,
            true_pos=result["true_pos"],
            true_neg=result["true_neg"],
            false_pos=result["false_pos"],
            false_neg=result["false_neg"],
        )
        self.logger.info(
            f"Finished zero shot in {time.strftime('%H:%M:%S', time.gmtime(runtime))}"
        )
        return runtime, f1

    def few_shot(
        self,
        data_indeces: List[int] | None = None,
        promt_template: str = "Is there an error in {attr}?\n\n{example}\n\n{context}?",
        n_samples: int = 100,
        example_count: int = 2,
    ) -> Tuple[float, float]:
        self.logger.info(
            f"Started few shot for {n_samples} rows with {example_count} examples"
        )
        dirty_data: pd.DataFrame
        clean_data: pd.DataFrame
        if data_indeces is None:
            dirty_data, clean_data = self.dataset.random_sample(n_samples)
        else:
            n_samples = len(data_indeces)
            dirty_data = self.dataset.get(dirty=True).iloc(data_indeces)
            clean_data = self.dataset.get(dirty=False).iloc(data_indeces)

        progressBar = IntProgress(
            min=0,
            max=min(dirty_data.shape[0], n_samples) * dirty_data.shape[1],
            description="Attributes Prompted Error Detection Few Shot",
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
                # get examples
                examples = self.dataset.generate_examples(
                    column_id=column, amount=example_count
                )

                # create prompt
                prompt = promt_template.format(
                    attr=attribute, context=serialized_row, example=examples
                )
                correct_value: bool = (
                    clean_data.loc[[row_index]][attribute].values[0] != value
                )
                y_true.append(int(correct_value))
                timestamp = int(time.time_ns() / 10**6)
                response = self._prompt(
                    prompt, id=f"ed_fs-{timestamp}", logger=self.logger
                )

                # evaluate response
                if "Yes" in response or "yes" in response:
                    self.logger.log_prompting_result(
                        id=f"ed_fs-{timestamp}", predicted=1, correct=int(correct_value)
                    )
                    y_pred.append(1)
                    if correct_value:
                        result["true_pos"] += 1
                    else:
                        result["false_pos"] += 1
                else:
                    self.logger.log_prompting_result(
                        id=f"ed_fs-{timestamp}", predicted=0, correct=int(correct_value)
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
            name="Error Detection few shot",
            runtime=runtimeString,
            n_rows=n_samples,
            dataset=self.dataset.name,
            f1=f1,
            true_pos=result["true_pos"],
            true_neg=result["true_neg"],
            false_pos=result["false_pos"],
            false_neg=result["false_neg"],
        )
        self.logger.info(
            f"Finished few shot in {time.strftime('%H:%M:%S', time.gmtime(runtime))}"
        )
        return runtime, f1
