import pandas as pd
import numpy as np
import random
from typing import Tuple

from ..setupExperiment import serialize_row


class DataSet:
    """This class encasulates all the functionality related to the error detection datasets."""

    def __init__(self, path_dirty: str, path_clean: str, name: str, end_token:str = "?"):
        self.dirty_set: pd.DataFrame = pd.read_csv(path_dirty)
        self.clean_set: pd.DataFrame = pd.read_csv(path_clean)
        self.name = name
        self.end_token = end_token

    def get(self, dirty: bool) -> pd.DataFrame:
        """Returns a dataset as a Dataframe based on the calling object.
        The "dirty" argument decides whether the dirty dataset or the clean dataset will be returned.
        """
        return self.dirty_set if dirty else self.clean_set

    def random_sample(self, amount: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns two DataFrames with `amount` random sample rows. There are no duplicate rows."""
        # Generate a random permutation of row indices
        rng = np.random.default_rng()  # Create a random number generator
        permutation = rng.permutation(len(self.dirty_set))

        # Shuffle both DataFrames using the same permutation
        dirty_shuffled = self.dirty_set.iloc[permutation[:amount]]
        clean_shuffled = self.clean_set.iloc[permutation[:amount]]

        return (dirty_shuffled, clean_shuffled)

    def random_sample_with_quota(
        self, amount, dirty_amount
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns two DataFrames with `amount` randomly sampled rows, of which `dirty_amount` contain errors while the rest are clean."""
        equal_indices = []
        different_indices = []
        for i, (index, row) in enumerate(self.dirty_set.iterrows()):
            if (self.clean_set.iloc[i] == row).all():
                equal_indices.append(i)
            else:
                different_indices.append(i)

        if dirty_amount > len(different_indices):
            print(
                f"Dataset does not contain enough errors! proceeding with maximum amount of {len(different_indices)} errors"
            )
            dirty_amount = len(different_indices)
        equal_indices = np.random.choice(
            equal_indices, size=amount - dirty_amount, replace=False
        )
        different_indices = np.random.choice(
            different_indices, size=dirty_amount, replace=False
        )

        dirty_rows = self.dirty_set.iloc[different_indices]
        clean_rows = self.dirty_set.iloc[equal_indices]

        dirty_sample = pd.concat([dirty_rows, clean_rows])
        ground_truth = self.clean_set.iloc[different_indices]
        clean_sample = pd.concat([ground_truth, clean_rows])

        rng = np.random.default_rng()  # Create a random number generator
        permutation = rng.permutation(len(dirty_sample))

        dirty_sample = dirty_sample.iloc[permutation]
        clean_sample = clean_sample.iloc[permutation]
        return (dirty_sample, clean_sample)

    def generate_examples(self, column_id: int, amount: int = 1, q_and_A: bool = False) -> str:
        """Returns a string with `amount` corrected random sample rows. There are no duplicate rows."""
        # this call already generates the sample
        sampleDataDirty, sampleDataClean = self.random_sample(amount)

        # it is now a matter of converting the sample into a string
        result_str = ""
        for i in range(amount):
            if(q_and_A): result_str += "Q: "
            result_str += serialize_row(sampleDataDirty.iloc[i])
            if(q_and_A): result_str += "  A:"
            else: result_str += self.end_token

            # the example string also contains corrections of the sample rows
            error_string = " No"
            if sampleDataDirty.iloc[i, column_id] != sampleDataClean.iloc[i, column_id]:
                error_string = " Yes"
            result_str += error_string + "\n"

        return result_str[:-1]


class Flights(DataSet):
    def __init__(self):
        super().__init__(
            "./data/error_detection/prepared/flights_dirty.csv",
            "./data/error_detection/prepared/flights_clean.csv",
            "Flights",
        )


class Food(DataSet):
    def __init__(self):
        super().__init__(
            "./data/error_detection/prepared/food_dirty.csv",
            "./data/error_detection/prepared/food_clean.csv",
            "Food",
        )


class Hospital(DataSet):
    def __init__(self):
        super().__init__(
            "./data/error_detection/prepared/hospital_dirty.csv",
            "./data/error_detection/prepared/hospital_clean.csv",
            "Hospital",
        )


class CustomDataSet(DataSet):
    def __init__(self, dirty_data: pd.DataFrame, clean_data: pd.DataFrame, name: str, end_token:str = "?"):
        self.dirty_set: pd.DataFrame = dirty_data
        self.clean_set: pd.DataFrame = clean_data
        self.name = f"{name}"
        self.end_token = end_token
