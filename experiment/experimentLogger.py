import logging
import json
from pathlib import Path
import os.path


class Logger:
    def __init__(
        self, path: str, name: str, level_file=logging.DEBUG, level_console=logging.INFO
    ) -> None:
        Path(f"{path}/{name}/responses/").mkdir(parents=True, exist_ok=True)

        with open(f"{path}/logs.csv", "w") as file:
            file.write("Name, Level, Time, Message\n")

        with open(f"{path}/{name}/prompt-results.csv", "w") as file:
            file.write("ID, Predicted, Correct\n")

        if not os.path.isfile(f"{path}/experiment-results.csv"):
            with open(f"{path}/experiment-results.csv", "w") as file:
                file.write(f"Name, Dataset, Number of Rows, Runtime, F1, Precision \n")

        logging.basicConfig(
            level=level_file,
            format="%(name)s, %(levelname)s, %(asctime)s, %(message)s",
            datefmt="%H:%M:%S",
            filename=f"{path}/logs.csv",
            filemode="a",
        )
        self.path = path
        self.name = name
        self.logger = logging.getLogger(name)

        console = logging.StreamHandler()
        console.setLevel(level_console)
        console.setFormatter(
            logging.Formatter("%(name)s (%(levelname)s):\t%(message)s")
        )
        self.logger.addHandler(console)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def log_response(self, id: str, response) -> None:
        with open(f"{self.path}/{self.name}/responses/{id}.json", "w") as file:
            json.dump(response, file)

    def log_prompting_result(self, id: str, predicted: int, correct: int) -> None:
        with open(f"{self.path}/{self.name}/prompt-results.csv", "a") as file:
            file.write(f"{id}, {predicted}, {correct}\n")

    def log_experiment_result(
        self,
        name: str,
        runtime: str,
        n_rows: int,
        dataset: str,
        f1: float,
        precision: float,
    ):
        with open(f"{self.path}/experiment-results.csv", "a") as file:
            file.write(f"{name}, {dataset}, {n_rows}, {runtime}, {f1}, {precision} \n")
