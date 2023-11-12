import pandas as pd
import numpy as np
from typing import Tuple


def serialize_row(row: pd.Series) -> str:
    result = ""
    for index, value in row.items():
        result += f"{index}: {value}, "
    return result[:-2] + "?"


class DataSet:
    """This class encasulates all the functionality related to the datasets."""

    def __init__(self, path_dirty: str, path_clean: str):
        self.dirty_set: pd.DataFrame = pd.read_csv(path_dirty)
        self.clean_set: pd.DataFrame = pd.read_csv(path_clean)

    def get(self, dirty: bool) -> pd.DataFrame:
        """Returns a dataset as an Dataframes based on the calling object.
        The "dirty" argument decides whether the dirty dataset or the clean dataset will be returned.
        """
        return self.dirty_set if dirty else self.clean_set

    def random_sample(self, amount: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns two DataFrames with "amount" random sample rows. There are no duplicate rows."""
        # Generate a random permutation of row indices
        rng = np.random.default_rng()  # Create a random number generator
        permutation = rng.permutation(len(self.dirty_set))

        # Shuffle both DataFrames using the same permutation
        dirty_shuffled = self.dirty_set.iloc[permutation[:amount]]
        clean_shuffled = self.clean_set.iloc[permutation[:amount]]

        return (dirty_shuffled, clean_shuffled)

    def generate_examples(self, column_id: int, amount: int = 1) -> str:
        """Returns a string with `amount` corrected random sample rows. There are no duplicate rows."""
        # this call already generates the sample
        sampleDataDirty, sampleDataClean = self.random_sample(amount)

        # it is now a matter of converting the sample into a string
        result_str = ""
        for i in range(amount):
            result_str += serialize_row(sampleDataDirty.iloc[i])

            # the example string also contains corrections of the sample rows
            error_string = " No"
            if sampleDataDirty.iloc[i, column_id] != sampleDataClean.iloc[i, column_id]:
                error_string = " Yes"
            result_str += error_string + "\n"

        return result_str[:-1]


class Flights(DataSet):
    def __init__(self):
        super().__init__("./prepared/flights_dirty.csv", "./prepared/flights_clean.csv")


class Food(DataSet):
    def __init__(self):
        super().__init__("./prepared/food_dirty.csv", "./prepared/food_clean.csv")


class Hospital(DataSet):
    def __init__(self):
        super().__init__(
            "./prepared/hospital_dirty.csv", "./prepared/hospital_clean.csv"
        )
