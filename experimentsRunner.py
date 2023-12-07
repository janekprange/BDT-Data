import pandas as pd
from experiment import Flights, Food, Hospital
from experiment import ErrorDetection
from llama_cpp.llama_grammar import LlamaGrammar
import csv
from typing import Union

MAXIMUM_ROW_COUNT = 20
MAXIMUM_EXAMPLE_COUNT = round(MAXIMUM_ROW_COUNT / 2)
ITERATION_AMOUNT = 5
GRAMMAR = ErrorDetection.GRAMMAR_YES_OR_NO
# GRAMMAR = None

ERROR_DETECTION_FLIGHTS = ErrorDetection(dataset=Flights())
ERROR_DETECTION_FOOD = ErrorDetection(dataset=Food())
ERROR_DETECTION_HOSPITAL = ErrorDetection(dataset=Hospital())


def test(
    ed: ErrorDetection, result_path: str, grammar: Union[LlamaGrammar, None] = None
) -> None:
    with open(result_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")  # type: ignore
        for i in range(ITERATION_AMOUNT):
            runtime_zeroshot, f1_zeroshot = ed.zero_shot(
                prompt_template="Is there an error in {attr}?\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                log_id=f"_p1_{i}",
                grammar=grammar,
            )

            writer.writerow([ed.dataset.name, "ZS", runtime_zeroshot, f1_zeroshot])

            runtime_fewshot, f1_fewshot = ed.few_shot(
                prompt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                log_id=f"_p1_{i}",
                example_count=MAXIMUM_EXAMPLE_COUNT,
                grammar=grammar,
            )

            writer.writerow([ed.dataset.name, "FS", runtime_fewshot, f1_fewshot])
            csv_file.flush()


if __name__ == "__main__":
    result_name = "grammar_yes_or_no_5x20"
    result_path = f"./analysis/data/{result_name}.csv"

    with open(result_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")  # type: ignore
        writer.writerow(["Dataset", "Type", "Time", "F1-Score"])

    print("START Flight")
    test(ERROR_DETECTION_FLIGHTS, result_path, GRAMMAR)
    print("START Food")
    test(ERROR_DETECTION_FOOD, result_path, GRAMMAR)
    # print("START Hospital")
    # test(ERROR_DETECTION_HOSPITAL, result_path, GRAMMAR)


"""

def flight_test(result_path) -> None:
    with open(result_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',') # type: ignore
        for i in range(ITERATION_AMOUNT):
            runtime_zeroshot, f1_zeroshot = ERROR_DETECTION_FLIGHTS.zero_shot(
                prompt_template="Is there an error in {attr}?\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                id=f"_p1_{i}",
            )

            writer.writerow(["Flight", "ZS", runtime_zeroshot, f1_zeroshot])

            runtime_fewshot, f1_fewshot = ERROR_DETECTION_FLIGHTS.few_shot(
                prompt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                id=f"_p1_{i}",
                example_count=MAXIMUM_EXAMPLE_COUNT,
            )

            writer.writerow(["Flight", "FS", runtime_fewshot, f1_fewshot])

            # zero_res = pd.DataFrame(
            #     [["Flight", "ZS", runtime_zeroshot, f1_zeroshot]],
            #     columns=["Dataset", "Type", "Time", "F1-Score"],
            # )
            # few_res = pd.DataFrame(
            #     [["Flight", "FS", runtime_fewshot, f1_fewshot]],
            #     columns=["Dataset", "Type", "Time", "F1-Score"],
            # )

            # data = pd.concat([data, zero_res, few_res], ignore_index=True)
    # return data

def food_test(result_path):
    with open(result_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',') # type: ignore
        for i in range(ITERATION_AMOUNT):
            runtime_zeroshot, f1_zeroshot = ERROR_DETECTION_FOOD.zero_shot(
                prompt_template="Is there an error in {attr}?\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                id=f"_p2_{i}",
            )

            writer.writerow(["Flight", "ZS", runtime_zeroshot, f1_zeroshot])

            runtime_fewshot, f1_fewshot = ERROR_DETECTION_FOOD.few_shot(
                prompt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                id=f"_p2_{i}",
                example_count=MAXIMUM_EXAMPLE_COUNT,
            )

            writer.writerow(["Flight", "FS", runtime_fewshot, f1_fewshot])

            # zero_res = pd.DataFrame(
            #     [["Food", "ZS", runtime_zeroshot, f1_zeroshot]],
            #     columns=["Dataset", "Type", "Time", "F1-Score"],
            # )
            # few_res = pd.DataFrame(
            #     [["Food", "FS", runtime_fewshot, f1_fewshot]],
            #     columns=["Dataset", "Type", "Time", "F1-Score"],
            # )
            # data = pd.concat([data, zero_res, few_res], ignore_index=True)
    # return data


def hospital_test(result_path):
    with open(result_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',') # type: ignore
        for i in range(ITERATION_AMOUNT):
            runtime_zeroshot, f1_zeroshot = ERROR_DETECTION_HOSPITAL.zero_shot(
                prompt_template="Is there an error in {attr}?\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                id=f"_p3_{i}",
            )

            writer.writerow(["Flight", "ZS", runtime_zeroshot, f1_zeroshot])

            runtime_fewshot, f1_fewshot = ERROR_DETECTION_HOSPITAL.few_shot(
                prompt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?",
                n_samples=MAXIMUM_ROW_COUNT,
                id=f"_p3_{i}",
                example_count=MAXIMUM_EXAMPLE_COUNT,
            )

            writer.writerow(["Flight", "FS", runtime_fewshot, f1_fewshot])

            # zero_res = pd.DataFrame(
            #     [["Hospital", "ZS", runtime_zeroshot, f1_zeroshot]],
            #     columns=["Dataset", "Type", "Time", "F1-Score"],
            # )
            # few_res = pd.DataFrame(
            #     [["Hospital", "FS", runtime_fewshot, f1_fewshot]],
            #     columns=["Dataset", "Type", "Time", "F1-Score"],
            # )
            # data = pd.concat([data, zero_res, few_res], ignore_index=True)
    # return data


# if __name__ == "__main__":
#     df = pd.DataFrame([], columns=["Dataset", "Type", "Time", "F1-Score"])
#     result_name = "test"
#     print("START Flight")
#     df = flight_test(df)
#     print("START Food")
#     df = food_test(df)
#     print("START Hospital")
#     df = hospital_test(df)
#     df.to_csv(f"./analysis/data/{result_name}.csv")
"""
