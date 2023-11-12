from typing import Literal, List
import pandas as pd
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display
from datetime import datetime
from .setupExperiment import SetupExperiment
from DataSet import DataSet, serialize_row
from .experimentLogger import Logger


class ErrorDetection(SetupExperiment):
    def __init__(
        self,
        dataset: DataSet,
        skip_prompting: bool = False,
        n_rows: int = 10,
        logging_path: str = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    ) -> None:
        super().__init__(skip_prompting)
        self.max_row_count = n_rows
        self.dataset = dataset
        self.logger = Logger(name="ErrorDetection", path=logging_path)

    def zero_shot(
        self,
        dirty_data: pd.DataFrame | None = None,
        prompt_template: str = "Is there an error in {attr}?\n{context}?",
        random_samples: int = 100,  # TODO: why is self.max_row_count not used here?
    ) -> List[int]:
        self.logger.debug("Started zero shot")
        if dirty_data is None:
            dirty_data, _ = self.dataset.random_sample(random_samples)

        progress_bar = IntProgress(
            min=0,
            max=min(dirty_data.shape[0], self.max_row_count) * dirty_data.shape[1],
            description="Attributes Prompted",
        )
        display(progress_bar)

        # generate table
        classifications: List[int] = []
        for row_index, row in dirty_data.iterrows():
            serialized_row = serialize_row(row)
            for cell_index, (attribute, value) in enumerate(row.items()):
                # create prompt
                prompt = prompt_template.format(
                    attr=attribute, val=value, context=serialized_row
                )
                response = self._prompt(
                    prompt, id=f"ed_zs-{row_index}-{cell_index}", logger=self.logger
                )

                # evaluate response
                if "Yes" in response or "yes" in response:
                    classifications.append(1)
                else:
                    classifications.append(0)

                progress_bar.value += 1
        self.logger.debug("Finished zero shot")
        return classifications

    def few_shot(
        self,
        dirty_data: pd.DataFrame | None = None,
        promt_template: str = "Is there an error in {attr}?\n\n{example}\n\n{context}?",
        random_samples: int = 100,  # TODO: why is self.max_row_count not used here?
        example_count: int = 2,
    ) -> List[int]:
        self.logger.debug("Started few shot")
        if dirty_data is None:
            dirty_data, _ = self.dataset.random_sample(random_samples)

        progressBar = IntProgress(
            min=0,
            max=min(dirty_data.shape[0], self.max_row_count) * dirty_data.shape[1],
            description="Attributes Prompted",
        )
        display(progressBar)
        # generate table
        classifications: List[int] = []
        for row_index, row in dirty_data.iterrows():
            serialized_row = serialize_row(row)
            for column, (attribute, _) in enumerate(row.items()):
                # get examples
                examples = self.dataset.generate_examples(
                    column_id=column, amount=example_count
                )

                # create prompt
                prompt = promt_template.format(
                    attr=attribute, context=serialized_row, example=examples
                )
                response = self._prompt(
                    prompt, id=f"ed_zs-{row_index}-{column}", logger=self.logger
                )

                # evaluate response
                if "Yes" in response or "yes" in response:
                    classifications.append(1)
                    self.logger.log_prompting_result(
                        id=f"ed_zs-{row_index}-{column}", predicted=1, correct=-1
                    )
                else:
                    classifications.append(0)
                    self.logger.log_prompting_result(
                        id=f"ed_zs-{row_index}-{column}", predicted=0, correct=-1
                    )

                progressBar.value += 1

        self.logger.debug("Finished few shot")
        return classifications
