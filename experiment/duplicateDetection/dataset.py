from typing import List
import numpy as np
import pandas as pd


class DataSet:
    """This class encasulates all the functionality related to the duplicate detection datasets."""

    def __init__(self, path_data: str, path_mapping: str, name: str):
        self.data: pd.DataFrame = pd.read_csv(path_data, index_col=0)
        self.mapping: pd.DataFrame = pd.read_csv(path_mapping, header=None)
        self.name = name

    def isduplicate(self, id1: int, id2: int) -> bool:
        return id2 in self.mapping[self.mapping[0] == id1][1].values

    def random_sample(
        self,
        amount: int,
        rows_with_duplicates: int,
        multiple_duplicate_change: float = 0,
    ) -> pd.DataFrame:
        """Returns a DataFrame with `amount` random sample rows. There are no duplicate rows.

        Args:
            amount (int): The number of rows the returned DataFrame has.
            rows_with_duplicates (int): The number of rows that have duplicates
            multiple_duplicate_change (float): The probability that a row has an additional duplicate.

        Raises:
            ValueError: If any of the parameter are outside the allowed range

        Returns:
            pd.DataFrame: A subset of `self.data` with `amount` rows.
        """
        if rows_with_duplicates < 0 or amount < 0:
            raise ValueError("Amount and rows_with_duplicates have to be positive")
        if rows_with_duplicates > amount / 2:
            raise ValueError(
                "Amount has to be as least twice as large as rows_with_duplicates"
            )
        if multiple_duplicate_change < 0 or multiple_duplicate_change > 1:
            raise ValueError(
                "multiple_duplicate_change has to be a probability between 0 and 1"
            )

        random_indeces = np.random.permutation(self.data.index)[:amount]
        result_indeces = []
        skip_next = 0
        for index in random_indeces:
            if skip_next > 0:
                skip_next -= 1
                continue
            if rows_with_duplicates > 0:
                # get all ids that are an duplicate of index
                duplicate_ids = self.mapping[self.mapping[0] == index][1].values
                # remove the ids that are in the sampled indeces
                duplicate_ids = [
                    id
                    for id in duplicate_ids
                    if id not in random_indeces and id not in result_indeces
                ]
                n_duplicates = 0
                while n_duplicates < len(duplicate_ids):
                    rows_with_duplicates -= 1
                    skip_next += 1
                    n_duplicates += 1
                    if (
                        rows_with_duplicates <= 0
                        or np.random.random() >= multiple_duplicate_change
                    ):
                        break
                result_indeces += list(
                    np.random.choice(duplicate_ids, n_duplicates, replace=False)
                )
            result_indeces.append(index)

        return self.data.loc[result_indeces]


class Affiliation(DataSet):
    def __init__(self):
        super().__init__(
            path_data="data/duplicate_detection/affiliationstrings_ids.csv",
            path_mapping="data/duplicate_detection/affiliationstrings_mapping.csv",
            name="Affiliation",
        )
