from typing import Tuple
import numpy as np
import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display
import time
from sklearn.metrics import f1_score as f1
from DataSet import Flights, Food, Hospital
from experiment.errorDetection import ErrorDetection
from experiment.setupExperiment import compare_dataframes_by_row
from experiment.setupExperiment import ground_truth_as_int

MAXIMUM_ROW_COUNT = 100
DEBUG_MESSAGES = True
ERROR_DETECTION_FLIGHTS = ErrorDetection(dataset=Flights(), n_rows=MAXIMUM_ROW_COUNT)
# ERROR_DETECTION_FOOD = ErrorDetection(dataset=Food(), n_rows=MAXIMUM_ROW_COUNT)
# ERROR_DETECTION_HOSPITAL = ErrorDetection(dataset=Hospital(), n_rows=MAXIMUM_ROW_COUNT)

ITERATION_AMOUNT = 2


def flight_test_p1(data) -> pd.DataFrame:
    shuffled_flights_dirty, shuffled_flights_clean = Flights().random_sample(
        MAXIMUM_ROW_COUNT
    )
    comp = compare_dataframes_by_row(shuffled_flights_dirty, shuffled_flights_clean)
    intsFlight = ground_truth_as_int(comp)

    for i in range(ITERATION_AMOUNT):
        start_time = time.time()
        classified_zero_shot = ERROR_DETECTION_FLIGHTS.zero_shot(
            shuffled_flights_dirty,
            prompt_template="Is there an error in {attr}?\n\n{context}?\n\nRestrict your answer to a single word that is either yes or no.",
        )
        end_time = time.time()

        time_spent_zero_shot = end_time - start_time
        flights_zero_shot_score = f1(intsFlight, classified_zero_shot)

        start_time = time.time()
        classified_few_shot = ERROR_DETECTION_FLIGHTS.few_shot(
            shuffled_flights_dirty,
            promt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?\n\nRestrict your answer to a single word that is either yes or no.",
        )
        end_time = time.time()

        time_spent_few_shot = end_time - start_time
        flights_few_shot_score = f1(intsFlight, classified_few_shot)

        zero_res = pd.DataFrame(
            [["Flight", "ZS", time_spent_zero_shot, flights_zero_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        few_res = pd.DataFrame(
            [["Flight", "FS", time_spent_few_shot, flights_few_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        data = pd.concat([data, zero_res, few_res], ignore_index=True)
    return data


def flight_test_p2(
    data, promt: str = "Is there an error in {attr}?\n{context}?"
) -> pd.DataFrame:
    shuffled_flights_dirty, shuffled_flights_clean = Flights().random_sample(
        MAXIMUM_ROW_COUNT
    )
    comp = compare_dataframes_by_row(shuffled_flights_dirty, shuffled_flights_clean)
    intsFlight = ground_truth_as_int(comp)

    for i in range(ITERATION_AMOUNT):
        start_time = time.time()
        classified_zero_shot = ERROR_DETECTION_FLIGHTS.zero_shot(
            shuffled_flights_dirty,
            prompt_template="First take a deep breath!\n\nIs there an error in {attr}?\n\n{context}?",
        )
        end_time = time.time()

        time_spent_zero_shot = end_time - start_time
        flights_zero_shot_score = f1(intsFlight, classified_zero_shot)

        start_time = time.time()
        classified_few_shot = ERROR_DETECTION_FLIGHTS.few_shot(
            shuffled_flights_dirty,
            promt_template="First take a deep breath!\n\nIs there an error in {attr}?\n\n{example}\n\n{context}?",
        )
        end_time = time.time()

        time_spent_few_shot = end_time - start_time
        flights_few_shot_score = f1(intsFlight, classified_few_shot)

        zero_res = pd.DataFrame(
            [["Flight", "ZS", time_spent_zero_shot, flights_zero_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        few_res = pd.DataFrame(
            [["Flight", "FS", time_spent_few_shot, flights_few_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        data = pd.concat([data, zero_res, few_res], ignore_index=True)
    return data


def food_test(data) -> pd.DataFrame:
    shuffled_food_dirty, shuffled_food_clean = Food().random_sample(MAXIMUM_ROW_COUNT)
    comp = compare_dataframes_by_row(shuffled_food_dirty, shuffled_food_clean)
    intsFood = ground_truth_as_int(comp)

    for i in range(ITERATION_AMOUNT):
        start_time = time.time()
        classified_zero_shot = ERROR_DETECTION_FOOD.zero_shot(shuffled_food_dirty)
        end_time = time.time()

        time_spent_zero_shot = end_time - start_time
        food_zero_shot_score = f1(intsFood, classified_zero_shot)

        start_time = time.time()
        classified_few_shot = ERROR_DETECTION_FOOD.few_shot(shuffled_food_dirty)
        end_time = time.time()

        time_spent_few_shot = end_time - start_time
        food_few_shot_score = f1(intsFood, classified_few_shot)

        zero_res = pd.DataFrame(
            [["Food", "ZS", time_spent_zero_shot, food_zero_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        few_res = pd.DataFrame(
            [["Food", "FS", time_spent_few_shot, food_few_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        data = pd.concat([data, zero_res, few_res], ignore_index=True)
    return data


def hospital_test(data) -> pd.DataFrame:
    shuffled_hospital_dirty, shuffled_hospital_clean = Hospital().random_sample(
        MAXIMUM_ROW_COUNT
    )
    comp = compare_dataframes_by_row(shuffled_hospital_dirty, shuffled_hospital_clean)
    ints_hospital = ground_truth_as_int(comp)

    for i in range(ITERATION_AMOUNT):
        start_time = time.time()
        classified_zero_shot = ERROR_DETECTION_HOSPITAL.zero_shot(
            shuffled_hospital_dirty
        )
        end_time = time.time()

        time_spent_zero_shot = end_time - start_time
        hospital_zero_shot_score = f1(ints_hospital, classified_zero_shot)

        start_time = time.time()
        classified_few_shot = ERROR_DETECTION_HOSPITAL.few_shot(shuffled_hospital_dirty)
        end_time = time.time()

        time_spent_few_shot = end_time - start_time
        hospital_few_shot_score = f1(ints_hospital, classified_few_shot)

        zero_res = pd.DataFrame(
            [["Hospital", "ZS", time_spent_zero_shot, hospital_zero_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        few_res = pd.DataFrame(
            [["Hospital", "FS", time_spent_few_shot, hospital_few_shot_score]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        data = pd.concat([data, zero_res, few_res], ignore_index=True)
    return data


if __name__ == "__main__":
    df = pd.DataFrame([], columns=["Dataset", "Type", "Time", "F1-Score"])
    result_name = "flight_restrict_answer_test_100_2"
    print("START Flight 1")
    df = flight_test_p1(df)
    df.to_csv(f"./analysis/data/{result_name}.csv")

    df = pd.DataFrame([], columns=["Dataset", "Type", "Time", "F1-Score"])
    result_name = "flight_deep_breath_test_100_2"
    print("START Flight 2")
    df = flight_test_p2(df)
    df.to_csv(f"./analysis/data/{result_name}.csv")

    # print("START Food")
    # df = food_test(df)
    # print("START Hospital")
    # df = hospital_test(df)
    # print("END")
