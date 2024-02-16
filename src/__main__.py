import os
import shutil
import tempfile
import zipfile

import dspy
import typer
from huggingface_hub import snapshot_download

from datasets import DatasetDict
from datasets import load_from_disk

from .alphacodium import AlphaCodium
from .alphacodium import TestCase
from .alphacodium import TestExecutionSuccess
from .alphacodium import run_test

# TODO:
# - Create some tooling to inspect the history of what actually gets sent to the LLM.
# - Use the dspy compile features to improve prompts.
# - Increase max_tokens?
# - Some examples don't seem to have any private tests?


def main():
    llm = dspy.OpenAI(
        model="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
        model_type="chat",
        max_tokens=2048,
    )
    dspy.settings.configure(lm=llm)

    ds = load_dataset()
    ds = ds["test"]

    problem_index = 69
    problem_description = ds[problem_index]["description"]
    public_tests_original = ds[problem_index]["public_tests"]
    private_tests_original = ds[problem_index]["private_tests"]

    public_tests = []
    assert len(public_tests_original["input"]) == len(public_tests_original["output"])
    for test_input, test_output in zip(
        public_tests_original["input"], public_tests_original["output"]
    ):
        public_tests.append(TestCase(input=test_input, output=test_output))

    alphacodium = AlphaCodium()
    code = alphacodium(
        problem_description=problem_description,
        public_tests=public_tests,
    )

    private_tests = []
    assert len(private_tests_original["input"]) == len(private_tests_original["output"])
    for test_input, test_output in zip(
        private_tests_original["input"], private_tests_original["output"]
    ):
        private_tests.append(TestCase(input=test_input, output=test_output))

    # TODO: It kind of sucks:)
    for i, test in enumerate(private_tests):
        res = run_test(code, test)
        if isinstance(res, TestExecutionSuccess):
            print(f"Test {i} passed")
        else:
            print(f"Test {i} failed\n{res.error_str}\n")

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


if __name__ == "__main__":
    typer.run(main)
