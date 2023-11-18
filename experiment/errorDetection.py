from typing import Tuple, List, Union, Literal
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
        model_size: Literal["small", "medium", "large"] = "large",
    ) -> None:
        super().__init__(skip_prompting, model_size=model_size)
        self.dataset = dataset
        self.logger = Logger(name="ErrorDetection", path=logging_path)

    def _execute(self, 
                 dirty_data: pd.DataFrame, 
                 clean_data: pd.DataFrame, 
                 prompt_template: str, 
                 experiment_name: str, 
                 dataset_name: str, 
                 example_count:int = 0,
                 id: Union[int, str] = "",
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
                
                # create promt with examples if needed
                if(example_count > 0):
                    examples = self.dataset.generate_examples(
                        column_id=column, amount=example_count
                    )
                    prompt = prompt_template.format(
                        attr=attribute, context=serialized_row, example=examples
                    )
                # otherwise create the promt without the examples
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
                    prompt, id=f"ed_fs{id}-{timestamp}", logger=self.logger
                )

                # evaluate response
                if "Yes" in response or "yes" in response:
                    self.logger.log_prompting_result(
                        id=f"ed_fs{id}-{timestamp}",
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
                        id=f"ed_fs{id}-{timestamp}",
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
            runtime=runtimeString,
            n_rows=n_samples,
            n_examples=example_count,
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
        id: Union[int, str] = "",
    ) -> Tuple[float, float]:
        self.logger.info(f"Started zero shot for {n_samples} rows")
        dirty_data: pd.DataFrame
        clean_data: pd.DataFrame
        if data_indices is None:
            dirty_data, clean_data = self.dataset.random_sample(n_samples)
        else:
            n_samples = len(data_indices)
            dirty_data = self.dataset.get(dirty=True).iloc(data_indices)
            clean_data = self.dataset.get(dirty=False).iloc(data_indices)

        return self._execute(dirty_data, clean_data, prompt_template, "Error Detection Zero Shot", self.dataset.name)

    def few_shot(
        self,
        data_indices: List[int] | None = None,
        prompt_template: str = "Is there an error in {attr}?\n\n{example}\n\n{context}?",
        n_samples: int = 100,
        example_count: int = 2,
        id: Union[int, str] = "",
    ) -> Tuple[float, float]:
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

        return self._execute(dirty_data, clean_data, prompt_template, "Error Detection Few Shot", self.dataset.name, example_count)
    
class CustomErrorDetection(ErrorDetection):
    def __init__(
        self,
        skip_prompting: bool = False,
        logging_path: str = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        model_size: Literal["small", "medium", "large"] = "large",
    ) -> None:
        super().__init__(None, skip_prompting=skip_prompting, logging_path=logging_path, model_size=model_size)
        
        # todo use given dataframes for zero_shot and few_shot
        
    

