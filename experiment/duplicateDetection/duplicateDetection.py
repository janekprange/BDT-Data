from datetime import datetime
import time
from typing import List, Literal, Tuple, Union

from llama_cpp import LlamaGrammar
from sklearn.metrics import f1_score

from .dataset import DataSet
from ..setupExperiment import SetupExperiment, serialize_row
from ..experimentLogger import Logger


class DuplicateDetection(SetupExperiment):
    def __init__(
        self,
        dataset: DataSet,
        skip_prompting: bool = False,
        logging_path: str = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        model_size: Literal["small", "medium", "large"] = "large",
    ) -> None:
        super().__init__(skip_prompting, model_size)
        self.dataset = dataset
        self.logger = Logger(name="DuplicateDetection", path=logging_path)

    def compare_rows(
        self,
        n_samples: int,
        rows_with_duplicates: int,
        multiple_duplicate_chance: float = 0,
        grammar: Union[LlamaGrammar, None] = None,
        prompt_template: str = "Are these two rows duplicates?\n{row1}\n{row2}",
        log_id: Union[int, str] = "",
        experiment_name="Duplicate Detection",
    ) -> Tuple[float, float]:
        self.logger.info(f"Started duplicate detection for {n_samples} rows")

        result = {
            "true_pos": 0,
            "false_pos": 0,
            "true_neg": 0,
            "false_neg": 0,
        }
        y_true: List[int] = []
        y_pred: List[int] = []
        start_time = time.time()

        sample_data = self.dataset.random_sample(
            amount=n_samples,
            rows_with_duplicates=rows_with_duplicates,
            multiple_duplicate_chance=multiple_duplicate_chance,
        )

        for index1, row1 in sample_data.iterrows():
            for index2, row2 in sample_data.iterrows():
                if int(index1) >= int(index2):  # type: ignore
                    continue
                correct_value = self.dataset.isduplicate(index1, index2)
                y_true.append(int(correct_value))

                prompt = prompt_template.format(
                    row1=serialize_row(row1), row2=serialize_row(row2)
                )

                timestamp = int(time.time_ns() / 10**6)
                response = self._prompt(
                    prompt,
                    id=f"dd{log_id}-{timestamp}",
                    logger=self.logger,
                    correct_answer=correct_value,
                    grammar=grammar,
                )
                # evaluate response
                if "Yes" in response or "yes" in response:
                    self.logger.log_prompting_result(
                        id=f"dd{log_id}-{timestamp}",
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
                        id=f"dd{log_id}-{timestamp}",
                        predicted=0,
                        correct=int(correct_value),
                    )
                    y_pred.append(0)
                    if correct_value:
                        result["false_neg"] += 1
                    else:
                        result["true_neg"] += 1

        runtime = time.time() - start_time
        runtimeString = time.strftime("%H:%M:%S", time.gmtime(runtime))
        f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1))

        self.logger.log_experiment_result(
            name=experiment_name,
            runtime=runtimeString,
            n_rows=n_samples,
            n_examples=0,
            dataset=self.dataset.name,
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
