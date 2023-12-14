import os
import pandas as pd
from llama_cpp.llama_grammar import LlamaGrammar
import csv
from typing import Union
from experiment.errorDetection import ErrorDetection, Flights, Food, Hospital, CustomDataSet
from experiment.duplicateDetection import DuplicateDetection, Affiliation

MAXIMUM_ROW_COUNT = 20
# MAXIMUM_EXAMPLE_COUNT = round(MAXIMUM_ROW_COUNT / 2)
MAXIMUM_EXAMPLE_COUNT = 5
ITERATION_AMOUNT = 10
# GRAMMAR = ErrorDetection.GRAMMAR_YES_OR_NO
GRAMMAR = None
# ERROR_DETECTION_FLIGHTS = ErrorDetection(dataset=Flights())
# ERROR_DETECTION_FOOD = ErrorDetection(dataset=Food())
# ERROR_DETECTION_HOSPITAL = ErrorDetection(dataset=Hospital())
ERROR_DETECTION_CUSTOM_EASY = ErrorDetection(dataset=CustomDataSet(pd.read_csv("data/error_detection/custom/dirty_dataframe_typos.csv"),
                                                              pd.read_csv("data/error_detection/custom/clean_dataframe.csv"),
                                                              "Custom_Typos"
                                                            ))
ERROR_DETECTION_CUSTOM_MID = ErrorDetection(dataset=CustomDataSet(pd.read_csv("data/error_detection/custom/dirty_data_wrong_cities.csv"),
                                                              pd.read_csv("data/error_detection/custom/clean_dataframe.csv"),
                                                              "Custom_Cities"
                                                            ))
ERROR_DETECTION_CUSTOM_HARD = ErrorDetection(dataset=CustomDataSet(pd.read_csv("data/error_detection/custom/dirty_dataframe_all_errors.csv"),
                                                              pd.read_csv("data/error_detection/custom/clean_dataframe.csv"),
                                                              "Custom_All"
                                                            ))

def test_errorDetection(
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
                grammar=grammar,
                custom_examples = "Country: Vietnam, City: Paris, Population: 5253000? Yes\n"+
                "Country: New Zealand, City: Welington, Population: 212700? Yes\n"+
                "Country: Saint Lucia, City: Castries, Population: 200000? Yes\n"+
                "Country: Estonia, City: Montevideo, Population: 1? Yes\n"+
                "Country: Australia, City: Canberra, Population: 395790? No\n"
            )

            writer.writerow([ed.dataset.name, "FS", runtime_fewshot, f1_fewshot])
            csv_file.flush()


def test_duplicateDetection_different_grammar(logging_path: str, n_iterations=3):
    N_ROWS = 50
    N_DUPLICATES = 20
    CHANCE_MULTIPLE_DUPLICATES = 0.3
    if os.path.exists(logging_path):
        raise FileExistsError(f"The directory '{logging_path}' already exists.")
    dd = DuplicateDetection(Affiliation(), logging_path=logging_path)
    for iteration in range(n_iterations):
        dd.zero_shot(
            n_samples=N_ROWS,
            rows_with_duplicates=N_DUPLICATES,
            multiple_duplicate_chance=CHANCE_MULTIPLE_DUPLICATES,
            grammar=dd.GRAMMAR_YES_OR_NO,
            experiment_name=f"WithGrammar-{iteration}",
        )
        dd.zero_shot(
            n_samples=N_ROWS,
            rows_with_duplicates=N_DUPLICATES,
            multiple_duplicate_chance=CHANCE_MULTIPLE_DUPLICATES,
            experiment_name=f"NoGrammar-{iteration}",
        )


if __name__ == "__main__":
    # test_duplicateDetection_different_grammar(
    #     logging_path="keep_logs/duplicateDetection_grammar"
    # )
    # exit()
    result_name = "custom_no_grammar_10x20"
    # result_name = "custom_grammar_yes_or_no_10x20"
    result_path = f"./analysis/data/{result_name}.csv"

    with open(result_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")  # type: ignore
        writer.writerow(["Dataset", "Type", "Time", "F1-Score"])

    print("START Easy")
    test_errorDetection(ERROR_DETECTION_CUSTOM_EASY, result_path, GRAMMAR)
    print("START Mid")
    test_errorDetection(ERROR_DETECTION_CUSTOM_MID, result_path, GRAMMAR)
    print("START Hard")
    test_errorDetection(ERROR_DETECTION_CUSTOM_HARD, result_path, GRAMMAR)

    # print("START Flight")
    # test_errorDetection(ERROR_DETECTION_FLIGHTS, result_path, GRAMMAR)
    # print("START Food")
    # test_errorDetection(ERROR_DETECTION_FOOD, result_path, GRAMMAR)
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
