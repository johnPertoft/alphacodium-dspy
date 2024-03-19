import os
import shutil
import tempfile
import zipfile

import dspy
import typer
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from huggingface_hub import snapshot_download

from .alphacodium import AlphaCodium
from .alphacodium import TestCase
from .alphacodium import TestExecutionSuccess
from .alphacodium import run_test

# TODO:
# - Create some tooling to inspect the history of what actually gets sent to the LLM.
# - Evaluate before and after optimizing with dspy compile features.
# - Increase max_tokens?
# - Some examples don't seem to have any private tests?
# - Running into problems with context length with openai when optimizing.


def main():
    llm = dspy.OpenAI(
        model="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
        model_type="chat",
        max_tokens=2048,
    )
    dspy.settings.configure(lm=llm)

    ds = load_dataset()
    train_set = make_examples(ds["valid"].select(range(10)))
    test_set = make_examples(ds["test"].select(range(10)), include_private_tests=True)

    optimizer = BootstrapFewShotWithRandomSearch(
        metric=evaluate_code_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        num_candidate_programs=10,
        num_threads=4,
    )

    alphacodium = AlphaCodium()
    alphacodium_optimized = optimizer.compile(alphacodium, trainset=train_set)

    for example in test_set:
        code = alphacodium_optimized(**example.without("private_tests"))
        for test in example.private_tests:
            res = run_test(code, test)
            if isinstance(res, TestExecutionSuccess):
                print("Test passed")
            else:
                print(f"Test failed\n{res.error_str}\n")
        print("=" * 80)

    breakpoint()


def load_dataset() -> DatasetDict:
    local_path = "datasets/valid_and_test_processed"
    try:
        return load_from_disk(local_path)
    except FileNotFoundError:
        pass

    download_path = snapshot_download(
        repo_id="talrid/CodeContests_valid_and_test_AlphaCodium",
        revision="main",
        repo_type="dataset",
    )
    zipfile_path = f"{download_path}/codecontests_valid_and_test_processed_alpha_codium.zip"
    extract_path = tempfile.mkdtemp()
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    dataset_path = f"{extract_path}/valid_and_test_processed"

    ds = load_from_disk(dataset_path)
    ds.save_to_disk(local_path)
    shutil.rmtree(extract_path)

    return ds


def make_examples(ds: Dataset, include_private_tests: bool = False) -> list[dspy.Example]:
    examples = []
    for x in ds:
        if include_private_tests:
            example = dspy.Example(
                problem_description=x["description"],  # type: ignore
                public_tests=make_tests(x["public_tests"]),  # type: ignore
                private_tests=make_tests(x["private_tests"]),  # type: ignore
            ).with_inputs("problem_description", "public_tests", "private_tests")
        else:
            example = dspy.Example(
                problem_description=x["description"],  # type: ignore
                public_tests=make_tests(x["public_tests"]),  # type: ignore
            ).with_inputs("problem_description", "public_tests")
        examples.append(example)
    return examples


def make_tests(tests_dict: dict[str, list[str]]) -> list[TestCase]:
    tests = []
    assert len(tests_dict["input"]) == len(tests_dict["output"])
    for test_input, test_output in zip(tests_dict["input"], tests_dict["output"]):
        tests.append(TestCase(input=test_input, output=test_output))
    return tests


def evaluate_code_metric(
    example: dspy.Example, pred: dict[str, str], trace: str | None = None
) -> float:
    # TODO: Should this just test whether all tests are passing instead?
    predicted_code = pred["code"]
    results = [run_test(predicted_code, test) for test in example.public_tests]
    return sum([1 for res in results if isinstance(res, TestExecutionSuccess)]) / len(results)


if __name__ == "__main__":
    typer.run(main)
