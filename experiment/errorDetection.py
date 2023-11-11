from typing import Literal, List
import pandas as pd
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display
from .setupExperiment import SetupExperiment


class ErrorDetection(SetupExperiment):
    def __init__(
        self,
        dataset: Literal["flights", "food", "hospital"],
        skip_prompting: bool,
        n_rows: int,
        random_sample: bool = False,
        logging_path: str | None = None,
    ) -> None:
        super().__init__(skip_prompting)
        self.max_row_count = n_rows
        # self.logger = Logger(logging_path)
        clean_data = pd.read_csv(f"prepared/{dataset}_clean.csv")
        dirty_data = pd.read_csv(f"prepared/{dataset}_dirty.csv")
        if random_sample:
            seed = np.random.randint(99999999)
            self.clean_data = clean_data.sample(n=n_rows, random_state=seed)
            self.dirty_data = dirty_data.sample(n=n_rows, random_state=seed)
        else:
            self.clean_data = clean_data.head(n_rows)
            self.dirty_data = dirty_data.head(n_rows)

    def zero_shot(
        self, prompt_template: str = "Is there an error in {attr}:{val}?\n{context}?"
    ) -> List[int]:
        progress_bar = IntProgress(
            min=0,
            max=min(self.dirty_data.shape[0], self.max_row_count)
            * self.dirty_data.shape[1],
            description="Attributes Prompted",
        )
        display(progress_bar)

        # generate table
        classifications: List[int] = []
        for _, row in self.dirty_data.iterrows():
            serialized_row = self.serialize_row(row)
            for _, (attribute, value) in enumerate(row.items()):
                # create prompt
                prompt = prompt_template.format(
                    attr=attribute, val=value, context=serialized_row
                )
                response = self._prompt(prompt)
                # evaluate response
                if "Yes" in response or "yes" in response:
                    classifications.append(1)
                else:
                    classifications.append(0)

                progress_bar.value += 1

        return classifications

    # TODO: why "attr", not "attr: val"?
    def few_shot(
        self,
        promt_template: str = "Is there an error in {attr}?\n\n{example}\n{context}?",
    ) -> List[int]:
        progressBar = IntProgress(
            min=0,
            max=min(self.dirty_data.shape[0], self.max_row_count)
            * self.dirty_data.shape[1],
            description="Attributes Prompted",
        )
        display(progressBar)
        # generate table
        classifications: List[int] = []
        for _, row in self.dirty_data.iterrows():
            serialized_row = self.serialize_row(row)
            for _, (attribute, value) in enumerate(row.items()):
                # get examples
                examples = []
                for _ in range(self.max_row_count):
                    row_str = self.sample_example()
                    examples.append(row_str)

                # create prompt
                prompt = promt_template.format(
                    attr=attribute, context=serialized_row, example="\n".join(examples)
                )
                response = self._prompt(prompt)

                # evaluate response
                if "Yes" in response or "yes" in response:
                    classifications.append(1)
                else:
                    classifications.append(0)

                progressBar.value += 1

        return classifications

    def sample_example(self) -> str:
        rand_row = np.random.randint(0, self.dirty_data.shape[0])
        rand_col = np.random.randint(0, self.dirty_data.shape[1])
        # print(f"row: {rand_row}, col: {rand_col}")
        error_string = " No"
        # if comparison of ground truth and dirty is "False" it means there is an error
        if self.clean_data.iloc[rand_row, rand_col] is False:
            error_string = " Yes"
        row = self.dirty_data.iloc[rand_row]
        result_str = self.serialize_row(row)
        return result_str + "?" + error_string
