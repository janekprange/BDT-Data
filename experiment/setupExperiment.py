from collections import defaultdict
from llama_cpp import Llama
import pandas as pd
from typing import List, Literal, Union
from .experimentLogger import Logger
from llama_cpp.llama_grammar import LlamaGrammar
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer
import transformers
import torch

import time


class SetupExperiment:
    MAX_TOKENS = 128

    GRAMMAR_YES_OR_NO = LlamaGrammar.from_string('root ::= ("Yes" | "No")')

    def __init__(
        self,
        skip_prompting: bool = False,
        model_size: Literal["small", "medium", "large"] = "small",
        model_source: Literal["meta", "bloke"] = "meta",
    ) -> None:
        self.skip_prompting = skip_prompting
        self.model_source = model_source
        if skip_prompting:
            return
        if model_source == "meta":
            # select the model
            model_name = ""
            match model_size:
                case "small":
                    model_name = "meta-llama/Llama-2-7b-chat-hf"
                case "medium":
                    model_name = "meta-llama/Llama-2-13b-chat-hf"
                case "large":
                    model_name = "meta-llama/Llama-2-70b-chat-hf"
                case _:
                    raise ValueError("Unknown model size")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif model_source == "bloke":
            # select the model
            model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
            model_basename = ""
            match model_size:
                case "small":
                    model_basename = "llama-2-13b-chat.Q3_K_S.gguf"
                case "medium":
                    model_basename = "llama-2-13b-chat.Q4_K_M.gguf"
                case "large":
                    model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
                case _:
                    raise ValueError("Unknown model size")

            # download the model
            model_path = hf_hub_download(
                repo_id=model_name_or_path, filename=model_basename
            )
            # https://llama-cpp-python.readthedocs.io/en/stable/api-reference/
            self.llama = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window (used to be 40960)
                n_gpu_layers=-1,  # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
                # n_threads=64, # CPU cores
                # n_batch=5120, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                # n_gpu_layers=1, # Change this value based on your model and your GPU VRAM pool.
                # tensor_split=8, #List of floats to split the model across multiple GPUs. If None, the model is not split
                verbose=False,  # Print verbose output to stderr. -> Sadly this does not work due to an issue in the library https://github.com/abetlen/llama-cpp-python/issues/729
            )
        else:
            raise ValueError("Unknown model source")

    def _prompt(
        self,
        prompt: str,
        id: str,
        logger: Logger,
        correct_answer: bool,
        grammar: Union[LlamaGrammar, None] = None,
    ) -> str:
        if self.skip_prompting:
            return f"skipped prompting for: {prompt}"

        start_time = time.time()

        if self.model_source == "bloke":
            max_tokens = self.MAX_TOKENS
            if grammar == self.GRAMMAR_YES_OR_NO:
                max_tokens = 3
            # https://llama-cpp-python.readthedocs.io/en/stable/api-reference/#llama_cpp.Llama.create_completion
            response = self.llama.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                # temperature=0.5,
                # top_p=0.95,
                # repeat_penalty=1.1,
                # top_k=50,
                # stop=["USER:"],  # Dynamic stopping when such token is detected.
                echo=False,  # return the prompt
                grammar=grammar,  # restrict llamas responses to the given grammar
            )
            end_time = time.time()
            log_response = {**response, "prompt": prompt, "correct_answer": correct_answer, "runtime": end_time - start_time}  # type: ignore
            logger.log_response(id=id, response=log_response)

            return response["choices"][0]["text"]  # type: ignore

        elif self.model_source == "meta":
            sequences = self.pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    # self.tokenizer.encode("\n", add_special_tokens=False)[-1], # only return the first line
                ],
                max_length=500,
            )

            end_time = time.time()
            if sequences[0]["generated_text"] is None:  # type: ignore
                logger.error(f"Empty respone for id {id}")
                return ""
            response = sequences[0]["generated_text"].replace(prompt, "").strip()  # type: ignore
            log_response = {
                "response": response,
                "prompt": prompt,
                "correct_answer": str(correct_answer),
                "runtime": end_time - start_time,
            }
            logger.log_response(id=id, response=log_response)

            return response  # type: ignore
        else:
            raise ValueError("Unknown model source")

    def _prompt_probabilities(
        self,
        prompt: str,
        id: str,
        logger: Logger,
        correct_answer: bool,
        grammar: Union[LlamaGrammar, None] = None,
    ):
        inputs = (
            torch.tensor(self.tokenizer.encode(prompt))
            .unsqueeze(0)
            .to(self.pipeline.device)  # type: ignore
        )  # type: ignore
        outputs = self.pipeline.model(inputs)
        probs = outputs[0][:, -1, :]
        probs = torch.softmax(probs, dim=-1)
        probs = probs.cpu().detach().numpy()[0]
        token_probs = defaultdict(float)
        for token, prob in enumerate(probs):
            token_probs[self.tokenizer.decode(token)] += prob

        for token, prob in sorted(
            token_probs.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"{token:<10}{prob:.2%}")


def serialize_row(row: pd.Series) -> str:
    result = ""
    for index, value in row.items():
        result += f"{index}: {value}, "
    return result[:-2]


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

    # first the edgecases with zeros:

    # no mistakes is always perfect
    if false_positives == 0 and false_negatives == 0:
        return 1.0

    # nothing is done correctly
    if true_positives == 0:
        return 0.0

    # the rest is just following the formular
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
