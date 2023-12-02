import pandas as pd
from experiment import Flights, Food, Hospital
from experiment import ErrorDetection

MAXIMUM_ROW_COUNT = 10
MAXIMUM_EXAMPLE_COUNT = round(MAXIMUM_ROW_COUNT / 2)
ITERATION_AMOUNT = 5

ERROR_DETECTION_FLIGHTS = ErrorDetection(dataset=Flights())
ERROR_DETECTION_FOOD = ErrorDetection(dataset=Food())
ERROR_DETECTION_HOSPITAL = ErrorDetection(dataset=Hospital())


def flight_test(data) -> pd.DataFrame:
    for i in range(ITERATION_AMOUNT):
        runtime_zeroshot, f1_zeroshot = ERROR_DETECTION_FLIGHTS.zero_shot(
            prompt_template="Is there an error in {attr}?\n\n{context}?",
            n_samples=MAXIMUM_ROW_COUNT,
            id=f"_p1_{i}",
        )

        runtime_fewshot, f1_fewshot = ERROR_DETECTION_FLIGHTS.few_shot(
            prompt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?",
            n_samples=MAXIMUM_ROW_COUNT,
            id=f"_p1_{i}",
            example_count=MAXIMUM_EXAMPLE_COUNT,
        )

        zero_res = pd.DataFrame(
            [["Flight", "ZS", runtime_zeroshot, f1_zeroshot]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        few_res = pd.DataFrame(
            [["Flight", "FS", runtime_fewshot, f1_fewshot]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )

        data = pd.concat([data, zero_res, few_res], ignore_index=True)
    return data


# def flight_test_p2(
#     data, promt: str = "Is there an error in {attr}?\n{context}?"
# ) -> pd.DataFrame:
#     for i in range(ITERATION_AMOUNT):
#         runtime_zeroshot, f1_zeroshot = ERROR_DETECTION_FLIGHTS.zero_shot(
#             prompt_template="First take a deep breath!\n\nIs there an error in {attr}?\n\n{context}?",
#             id=f"_p2_{i}",
#         )

#         runtime_fewshot, f1_fewshot = ERROR_DETECTION_FLIGHTS.few_shot(
#             prompt_template="First take a deep breath!\n\nIs there an error in {attr}?\n\n{example}\n\n{context}?",
#             id=f"_p2_{i}",
#         )

#         zero_res = pd.DataFrame(
#             [["Flight", "ZS", runtime_zeroshot, f1_zeroshot]],
#             columns=["Dataset", "Type", "Time", "F1-Score"],
#         )
#         few_res = pd.DataFrame(
#             [["Flight", "FS", runtime_fewshot, f1_fewshot]],
#             columns=["Dataset", "Type", "Time", "F1-Score"],
#         )
#         data = pd.concat([data, zero_res, few_res], ignore_index=True)
#     return data


def food_test(data) -> pd.DataFrame:
    for i in range(ITERATION_AMOUNT):
        runtime_zeroshot, f1_zeroshot = ERROR_DETECTION_FOOD.zero_shot(
            prompt_template="Is there an error in {attr}?\n\n{context}?",
            n_samples=MAXIMUM_ROW_COUNT,
            id=f"_p2_{i}",
        )

        runtime_fewshot, f1_fewshot = ERROR_DETECTION_FOOD.few_shot(
            prompt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?",
            n_samples=MAXIMUM_ROW_COUNT,
            id=f"_p2_{i}",
            example_count=MAXIMUM_EXAMPLE_COUNT,
        )

        zero_res = pd.DataFrame(
            [["Food", "ZS", runtime_zeroshot, f1_zeroshot]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        few_res = pd.DataFrame(
            [["Food", "FS", runtime_fewshot, f1_fewshot]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        data = pd.concat([data, zero_res, few_res], ignore_index=True)
    return data


def hospital_test(data) -> pd.DataFrame:
    for i in range(ITERATION_AMOUNT):
        runtime_zeroshot, f1_zeroshot = ERROR_DETECTION_HOSPITAL.zero_shot(
            prompt_template="Is there an error in {attr}?\n\n{context}?",
            n_samples=MAXIMUM_ROW_COUNT,
            id=f"_p3_{i}",
        )

        runtime_fewshot, f1_fewshot = ERROR_DETECTION_HOSPITAL.few_shot(
            prompt_template="Is there an error in {attr}?\n\n{example}\n\n{context}?",
            n_samples=MAXIMUM_ROW_COUNT,
            id=f"_p3_{i}",
            example_count=MAXIMUM_EXAMPLE_COUNT,
        )

        zero_res = pd.DataFrame(
            [["Hospital", "ZS", runtime_zeroshot, f1_zeroshot]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        few_res = pd.DataFrame(
            [["Hospital", "FS", runtime_fewshot, f1_fewshot]],
            columns=["Dataset", "Type", "Time", "F1-Score"],
        )
        data = pd.concat([data, zero_res, few_res], ignore_index=True)
    return data


if __name__ == "__main__":
    df = pd.DataFrame([], columns=["Dataset", "Type", "Time", "F1-Score"])
    result_name = "general_10_5"
    print("START Flight")
    df = flight_test(df)
    print("START Food")
    df = food_test(df)
    print("START Hospital")
    df = hospital_test(df)
    df.to_csv(f"./analysis/data/{result_name}.csv")

    # df = pd.DataFrame([], columns=["Dataset", "Type", "Time", "F1-Score"])
    # result_name = "flight_deep_breath_test_100_2"
    # print("START Flight 2")
    # df = flight_test_p2(df)
    # df.to_csv(f"./analysis/data/{result_name}.csv")

    # print("START Food")
    # df = food_test(df)
    # print("START Hospital")
    # df = hospital_test(df)
    # print("END")
