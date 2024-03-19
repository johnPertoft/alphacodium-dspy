import sys
from pathlib import Path

import dspy
from loguru import logger
from pydantic import BaseModel
from pydantic import Field

# TODO:
# - Tune temperatures a bit? I think they use much lower in the paper.
# - Dspy uses the names of input/output fields to construct the prompts. Maybe take another
#   pass to make sure that they make sense.
# - Use the prefix field of e.g. dspy.OutputField to add instructions?
# - Include the validation steps mentioned in the paper too?
# - I think dspy uses json encoding when using pydantic classes, but the paper makes arguments
#   for using yaml. Maybe we should use yaml instead? Can we?
# - Consider adding more constraints to the output types of the signatures, i.e. in the Field
# - How to add demonstrations for different submodules?

logger = logger.opt(colors=True)


class TestCase(BaseModel):
    input: str = Field(description="The stdin input for the test case")
    output: str = Field(description="The expected stdout output for the test case")


class TestExecutionSuccess(BaseModel):
    pass


class TestExecutionFailure(BaseModel):
    error_str: str
    d_tot: float


class ProblemReflectionSignature(dspy.Signature):
    problem_description: str = dspy.InputField()
    tests: list[TestCase] = dspy.InputField()
    problem_reflection: str = dspy.OutputField()


class TestReflectionSignature(dspy.Signature):
    problem_description: str = dspy.InputField()
    tests: list[TestCase] = dspy.InputField()
    test_reflection: str = dspy.OutputField()


class SolutionStrategyGenerationSignature(dspy.Signature):
    """
    Come up with a strategy/algorithm sketch to solve the problem.
    Don't write any code yet.
    """

    problem_description: str = dspy.InputField()
    problem_reflection: str = dspy.InputField()
    test_reflection: str = dspy.InputField()
    solution_description: str = dspy.OutputField()


class RankSolutionStrategiesSignature(dspy.Signature):
    """
    Reason about the given solutions and select the best one.
    Copy the best solution to the output.
    """

    problem_description: str = dspy.InputField()
    problem_reflection: str = dspy.InputField()
    test_reflection: str = dspy.InputField()
    solutions: list[str] = dspy.InputField()
    best_solution: str = dspy.OutputField()


class TestGenerationSignature(dspy.Signature):
    problem_description: str = dspy.InputField()
    problem_reflection: str = dspy.InputField()
    test_reflection: str = dspy.InputField()
    given_tests: list[TestCase] = dspy.InputField()
    additional_tests: list[TestCase] = dspy.OutputField()


class CodeGenerationSignature(dspy.Signature):
    """
    Generate python 3 code that solves the problem.
    Only output the code.
    Do not include any explanation of the code.
    Code should be modular and have a single entrypoint (`if __name__ == "__main__":`).
    Only output one code solution.
    """

    problem_description: str = dspy.InputField()
    problem_reflection: str = dspy.InputField()
    test_reflection: str = dspy.InputField()
    solution_strategy: str = dspy.InputField()
    code_solution: str = dspy.OutputField()


class CodeImprovementSignature(dspy.Signature):
    """
    Given the current code solution and the resulting error message, improve and fix the code.
    Always output a new code solution.
    """

    problem_description: str = dspy.InputField()
    problem_reflection: str = dspy.InputField()
    test_reflection: str = dspy.InputField()
    code_solution: str = dspy.InputField()
    error: str = dspy.InputField()
    improved_code_solution: str = dspy.OutputField()


class AlphaCodium(dspy.Module):
    def __init__(self) -> None:
        self.problem_reflection = dspy.TypedPredictor(ProblemReflectionSignature)
        self.test_reflection = dspy.TypedPredictor(TestReflectionSignature)
        self.solution_generation = dspy.TypedPredictor(SolutionStrategyGenerationSignature)
        self.rank_solutions = dspy.TypedPredictor(RankSolutionStrategiesSignature)
        self.test_generation = dspy.TypedPredictor(TestGenerationSignature)
        self.code_generation = dspy.TypedPredictor(CodeGenerationSignature)
        self.code_improvement = dspy.TypedPredictor(CodeImprovementSignature)

    def forward(
        self,
        problem_description: str,
        public_tests: list[TestCase],
    ) -> dict[str, str]:
        logger.info("Starting AlphaCodium pipeline")

        logger.info("Running problem reflection")
        problem_reflection = self.problem_reflection(
            problem_description=problem_description,
            tests=public_tests,
        ).problem_reflection

        logger.info("Running test reflection")
        test_reflection = self.test_reflection(
            problem_description=problem_description,
            tests=public_tests,
        ).test_reflection

        # TODO:
        # - The caching behavior means that each call of the function inside the loop returns
        #   the same thing. Seems like a bug?
        #   https://github.com/stanfordnlp/dspy/issues/578
        # - Is it better to just ask for three solutions directly instead? Now when it's not aware
        #   of what it has previously generated it will generate basically the same thing multiple
        #   times.
        # - Ask for a list output instead here instead of the loop?
        # Note: The caching behavior of DSPy means that the calls inside the loop will be cached
        # after the first iteration and thus identical. To force slightly different calls we set
        # the temperature slightly differently for each call.
        logger.info("Running solution generation")
        solutions = []
        for i in range(3):
            solution = self.solution_generation(
                problem_description=problem_description,
                problem_reflection=problem_reflection,
                test_reflection=test_reflection,
                config={"temperature": 0.7 + (0.1 * i)},
            ).solution_description
            solutions.append(solution)

        # TODO:
        # - Should use pydantic outputs here as well
        # - And maybe actually ask for a ranking too so we can use higher ranked ones
        #   with higher preference later.
        # Rank solutions and choose best one.
        logger.info("Running solution ranking")
        best_solution = self.rank_solutions(
            problem_description=problem_description,
            problem_reflection=problem_reflection,
            test_reflection=test_reflection,
            solutions=solutions,
        ).best_solution

        # Initial code solution.
        logger.info("Running initial code generation")
        code, public_test_results = self.initial_code_generation(
            problem_description=problem_description,
            problem_reflection=problem_reflection,
            test_reflection=test_reflection,
            best_solution=best_solution,
            solutions=solutions,
            public_tests=public_tests,
        )
        public_tests_passing = all(
            isinstance(result, TestExecutionSuccess) for result in public_test_results
        )
        logger.info(
            f"Initial code generation: {'<green>passing</green>' if public_tests_passing else '<yellow>failing</yellow>'} public tests."  # noqa: E501
        )

        # TODO:
        # - Paper says to generate 6-8 additional tests.
        # Generate additional test cases.
        logger.info("Running test generation")
        generated_tests = self.test_generation(
            problem_description=problem_description,
            problem_reflection=problem_reflection,
            test_reflection=test_reflection,
            given_tests=public_tests,
        ).additional_tests

        # TODO:
        # - The code differs a bit from the paper here, the paper says to just go test by test
        #   but the code seems to make sure that previous tests are still passing after each fix
        #   at this stage too.
        # - Otherwise we might run into a situation where we fix one test but break another and then
        #   go back and forth between two solutions.
        # Iterate on public tests.
        if not public_tests_passing:
            logger.info("Running public test code fix iteration")
            public_test_results_fixed = []
            for test, result in zip(public_tests, public_test_results):
                if isinstance(result, TestExecutionSuccess):
                    public_test_results_fixed.append(result)
                    continue
                code, result = self.fix_code(
                    problem_description=problem_description,
                    problem_reflection=problem_reflection,
                    test_reflection=test_reflection,
                    code=code,
                    test=test,
                    failure=result,
                )
                public_test_results_fixed.append(result)
                # TODO: If the fix doesn't work, what should we do? Which code version to
                # continue with?
            public_tests_passing = all(
                isinstance(result, TestExecutionSuccess) for result in public_test_results_fixed
            )
            logger.info(
                f"Public test code fix: {'<green>passing</green>' if public_tests_passing else '<yellow>failing</yellow>'} public tests."  # noqa: E501
            )

        # Iterate on generated tests.
        logger.info("Running generated test code fix iteration")
        test_anchors = public_tests.copy()
        for test in generated_tests:
            result = run_test(code, test)

            if isinstance(result, TestExecutionSuccess):
                test_anchors.append(test)
                continue

            fixed_code, result = self.fix_code(
                problem_description=problem_description,
                problem_reflection=problem_reflection,
                test_reflection=test_reflection,
                code=code,
                test=test,
                failure=result,
            )

            # Continue to the next test if the fix didn't work.
            if isinstance(result, TestExecutionFailure):
                continue

            # Test on all test anchors to make sure we didn't break anything.
            anchor_test_results = [run_test(fixed_code, test) for test in test_anchors]
            anchor_test_results_passing = all(
                isinstance(result, TestExecutionSuccess) for result in anchor_test_results
            )
            if anchor_test_results_passing:
                code = fixed_code
                test_anchors.append(test)

        # TODO: Without a dict it fails in the optimization code, doesn't seem intended though.
        return {"code": code}

    def initial_code_generation(
        self,
        problem_description: str,
        problem_reflection: str,
        test_reflection: str,
        best_solution: str,
        solutions: list[str],
        public_tests: list[TestCase],
    ) -> tuple[str, list[TestExecutionSuccess | TestExecutionFailure]]:
        code = self.code_generation(
            problem_description=problem_description,
            problem_reflection=problem_reflection,
            test_reflection=test_reflection,
            solution_strategy=best_solution,
        ).code_solution
        code = clean_code(code)
        test_results = [run_test(code, test) for test in public_tests]
        tests_passing = all(isinstance(result, TestExecutionSuccess) for result in test_results)

        if tests_passing:
            return code, test_results

        for attempt in range(3):
            code_alternative = self.code_generation(
                problem_description=problem_description,
                problem_reflection=problem_reflection,
                test_reflection=test_reflection,
                solution_strategy=solutions[attempt % len(solutions)],
            ).code_solution
            code_alternative = clean_code(code_alternative)
            test_results = [run_test(code, test) for test in public_tests]
            tests_passing = all(isinstance(result, TestExecutionSuccess) for result in test_results)
            if tests_passing:
                return code_alternative, test_results

        if not tests_passing:
            # TODO: Should pick the best solution according to d_tot here.
            pass

        return code, test_results

    def fix_code(
        self,
        problem_description: str,
        problem_reflection: str,
        test_reflection: str,
        code: str,
        test: TestCase,
        failure: TestExecutionFailure | None,
    ) -> tuple[str, TestExecutionSuccess | TestExecutionFailure]:
        if failure is None:
            res = run_test(code, test)
            if isinstance(res, TestExecutionSuccess):
                return code, TestExecutionSuccess()
            else:
                failure = res

        for _ in range(3):
            improved_code = self.code_improvement(
                problem_description=problem_description,
                problem_reflection=problem_reflection,
                test_reflection=test_reflection,
                code_solution=code,
                error=failure.error_str,
            ).improved_code_solution
            improved_code = clean_code(improved_code)
            res = run_test(improved_code, test)
            if isinstance(res, TestExecutionSuccess):
                return improved_code, TestExecutionSuccess()
            else:
                failure = res

        return code, res


def clean_code(code: str) -> str:
    code = code.rstrip("` \n")
    if code.startswith("```python"):
        code = code[10:]
    elif code.startswith("python"):
        code = code[6:]
    return code


def run_test(code: str, test: TestCase) -> TestExecutionSuccess | TestExecutionFailure:
    # TODO: Make this installable instead?
    alpha_codium_path = str(Path(__file__).parent.parent / "AlphaCodium")
    if alpha_codium_path not in sys.path:
        sys.path.append(alpha_codium_path)
    from alpha_codium.gen.stages.run_tests import (
        run_tests as run_tests_alphacodium_contrib,
    )

    logger.disable("alpha_codium")

    example = {
        "name": "dummy-value",
        "code_recent_solution": code,
    }
    (
        _problem,
        all_passed,
        _non_empty_output,
        error_str,
        _trace_str,
        tests_timeout,
        d_tot,
    ) = run_tests_alphacodium_contrib(
        self=None,
        problem=example,
        counter=0,
        test_inputs=[test.input],
        test_outputs=[test.output],
    )

    # TODO: Should we add something to the error_str in the case of wrong output?
    # I guess it's sort of clear because it says "expected_output" etc.
    # How do we know that was the problem though and not just a crash/timeout?

    if tests_timeout:
        # TODO: What does the error_str look like in this case?
        breakpoint()

    if all_passed:
        return TestExecutionSuccess()

    return TestExecutionFailure(
        error_str=error_str,
        d_tot=d_tot,
    )
