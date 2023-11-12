from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import pandas as pd
from typing import List
from .experimentLogger import Logger


class SetupExperiment:
    def __init__(self, skip_prompting: bool) -> None:
        self.skip_prompting = skip_prompting
        if skip_prompting:
            return
        # select the model
        model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
        model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
        # download the model
        model_path = hf_hub_download(
            repo_id=model_name_or_path, filename=model_basename
        )
        # https://llama-cpp-python.readthedocs.io/en/stable/api-reference/
        self.llama = Llama(
            model_path=model_path,
            n_ctx=1024,  # Context window (used to be 40960)
            n_gpu_layers=-1,  # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
            # n_threads=64, # CPU cores
            # n_batch=5120, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            # n_gpu_layers=1, # Change this value based on your model and your GPU VRAM pool.
            # tensor_split=8, #List of floats to split the model across multiple GPUs. If None, the model is not split
            verbose=False,  # Print verbose output to stderr. -> Sadly this does not work due to an issue in the library https://github.com/abetlen/llama-cpp-python/issues/729
        )

    def _prompt(self, prompt: str, id: str, logger: Logger):
        if self.skip_prompting:
            return f"skipped prompting for: {prompt}"

        # https://llama-cpp-python.readthedocs.io/en/stable/api-reference/#llama_cpp.Llama.create_completion
        response = self.llama(
            prompt=prompt,
            max_tokens=10,
            temperature=0.5,
            top_p=0.95,
            repeat_penalty=1.2,
            top_k=50,
            stop=["USER:"],  # Dynamic stopping when such token is detected.
            echo=False,  # return the prompt
        )
        logger.log_response(id=id, response=response)

        return response["choices"][0]["text"]  # type: ignore


def compare_dataframes_by_row(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Check if the DataFrames have the same shape
    if not df1.shape == df2.shape:
        raise ValueError("DataFrames must have the same shape for row-wise comparison.")

    # Compare the two DataFrames element-wise and create a Boolean DataFrame
    comparison_result = df1 != df2

    return comparison_result


def ground_truth_as_int(gt: pd.DataFrame) -> pd.DataFrame:
    new_df = gt.astype(dtype="int")
    return new_df.values.ravel().tolist()


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    true_positives = sum(1 for a, b in zip(y_true, y_pred) if a == b == 1)
    false_positives = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    false_negatives = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    if true_positives == 0:
        return 0.0  # Handle the case where true_positives is 0

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
