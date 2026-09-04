from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from gaia_max.domain import (
    AnswerType,
    Modality,
    OutputContract,
    Question,
    StringAnswer,
    TaskClass,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.solvers.python_code import (
    ConstrainedPythonExecutor,
    PythonCodeSolver,
    PythonCodeSolverError,
    SandboxExecutionError,
    SandboxLimits,
    UnsafePythonSourceError,
    decode_python_source,
    extract_final_numeric_output,
    inspect_python_source,
    register_python_code_solver,
    replay_simple_final_numeric,
)
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis


def analysis(
    *,
    task_class: TaskClass = TaskClass.CODE,
    modality: Modality = Modality.PYTHON,
    answer_type: AnswerType = AnswerType.STRING,
) -> ReconciledTaskAnalysis:
    return ReconciledTaskAnalysis(
        task_id="python-task",
        modality=modality,
        task_class=task_class,
        requested_operation="determine final numeric program output",
        output_contract=OutputContract(answer_type=answer_type),
        risk_flags=("numeric_subtype_requires_source", "sandbox_required"),
        field_sources={
            "modality": AnalysisAuthority.PROFILE,
            "task_class": AnalysisAuthority.PROFILE,
            "requested_operation": AnalysisAuthority.PROFILE,
            "temporal_constraint": AnalysisAuthority.PROFILE,
            "filters": AnalysisAuthority.PROFILE,
            "output_contract": AnalysisAuthority.PROFILE,
            "risk_flags": AnalysisAuthority.PROFILE,
        },
        contract_parse_status=ContractParseStatus.COMPLETE,
        profile_used=True,
        model_used=False,
        requires_review=False,
    )


def source_provider(
    source: bytes,
) -> Callable[[Question, ReconciledTaskAnalysis], Awaitable[bytes]]:
    async def provide(
        question: Question,
        task_analysis: ReconciledTaskAnalysis,
    ) -> bytes:
        assert question.file_name == "python-task.py"
        assert task_analysis.task_class is TaskClass.CODE
        return source

    return provide


def test_declared_python_encoding_is_decoded_without_execution() -> None:
    source = "# -*- coding: latin-1 -*-\nlabel = 'cafÃ©'\nprint(7)\n".encode("latin-1")

    decoded = decode_python_source(source)

    assert "cafÃ©" in decoded


@pytest.mark.parametrize("content", [b"", b"print(1)\x00", b"# coding: no-such-codec\n"])
def test_empty_binary_or_badly_encoded_source_fails(content: bytes) -> None:
    with pytest.raises(PythonCodeSolverError):
        decode_python_source(content)


def test_static_inspection_allows_pure_standard_library_math() -> None:
    report = inspect_python_source("import math\nprint(math.factorial(6))\n")

    assert report.imported_modules == ("math",)
    assert report.node_count > 0
    assert "filesystem_network_process_blocklist" in report.checks


@pytest.mark.parametrize(
    "source",
    [
        "import socket\nprint(1)\n",
        "import subprocess\nsubprocess.run([\"whoami\"])\n",
        "import os\nprint(os.environ)\n",
        "print(open(\"secret.txt\").read())\n",
        "print(eval(\"40 + 2\"))\n",
        "print((1).__class__)\n",
        "import collections\nprint(collections._sys)\n",
        "from collections import _sys\nprint(1)\n",
        "import operator\nprint(operator.add(1, 2))\n",
        "class Escape:\n    pass\nprint(1)\n",
        "from math import *\nprint(factorial(3))\n",
    ],
)
def test_network_process_filesystem_and_dynamic_escape_shapes_are_blocked(
    source: str,
) -> None:
    with pytest.raises(UnsafePythonSourceError):
        inspect_python_source(source)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("42\n", "42"),
        ("intermediate\n-1.25e+3\n", "-1.25e+3"),
        ("\n(3+4j)\n", "(3+4j)"),
    ],
)
def test_final_numeric_output_preserves_the_printed_literal(stdout: str, expected: str) -> None:
    assert extract_final_numeric_output(stdout) == expected


@pytest.mark.parametrize("stdout", ["", "answer: 42\n", "True\n", "1 2\n"])
def test_non_numeric_final_output_fails(stdout: str) -> None:
    with pytest.raises(SandboxExecutionError):
        extract_final_numeric_output(stdout)


def test_simple_static_replay_independently_reproduces_arithmetic() -> None:
    source = "total = 6 * 7\ntotal += 5\nprint(total)\n"

    assert replay_simple_final_numeric(source) == "47"
    assert replay_simple_final_numeric("for value in range(3):\n    print(value)\n") is None


def test_safe_program_runs_twice_in_fresh_bounded_processes() -> None:
    source = "x = 6\ny = 7\nprint(x * y)\n"

    result = ConstrainedPythonExecutor().execute_verified(source)

    assert result.final_numeric_output == "42"
    assert result.runs_completed == 2
    assert result.fresh_process_outputs_match is True
    assert result.static_replay_supported is True
    assert result.static_replay_confirmed is True


def test_safe_function_program_runs_when_static_replay_declines() -> None:
    source = "def square(value):\n    return value * value\n\nprint(square(9))\n"

    result = ConstrainedPythonExecutor().execute_verified(source)

    assert result.final_numeric_output == "81"
    assert result.fresh_process_outputs_match is True
    assert result.static_replay_supported is False
    assert result.static_replay_confirmed is False


def test_infinite_program_is_killed_by_wall_timeout() -> None:
    executor = ConstrainedPythonExecutor(SandboxLimits(wall_seconds=0.25))

    with pytest.raises(SandboxExecutionError, match="wall timeout"):
        executor.execute_verified("while True:\n    pass\n")


def test_unbounded_output_is_killed() -> None:
    limits = SandboxLimits(wall_seconds=2, max_output_bytes=1024)
    source = "while True:\n    print(1234567890)\n"

    with pytest.raises(SandboxExecutionError, match="output limit"):
        ConstrainedPythonExecutor(limits).execute_verified(source)


def test_memory_hungry_program_cannot_complete() -> None:
    limits = SandboxLimits(wall_seconds=3, memory_bytes=256 * 1024 * 1024)
    source = "payload = bytearray(1024 * 1024 * 1024)\nprint(len(payload))\n"

    with pytest.raises(SandboxExecutionError, match=r"memory limit|exited with code"):
        ConstrainedPythonExecutor(limits).execute_verified(source)


@pytest.mark.asyncio
async def test_solver_returns_typed_exact_numeric_text() -> None:
    question = Question(
        task_id="python-task",
        question="What is the final numeric output?",
        file_name="python-task.py",
    )
    solver = PythonCodeSolver(source_provider(b"value = 10 / 4\nprint(value)\n"))

    result = await solver.solve(question, analysis(), SolverRoute.CODE.value)

    assert result == StringAnswer(value="2.5")


@pytest.mark.asyncio
async def test_registered_solver_dispatches_through_existing_registry() -> None:
    question = Question(
        task_id="python-task",
        question="What is the final numeric output?",
        file_name="python-task.py",
    )
    registry = SolverRegistry()
    registered = register_python_code_solver(registry, source_provider(b"print(73)\n"))

    result = await registry.solve(SolverRoute.CODE, question, analysis())

    assert isinstance(registered, PythonCodeSolver)
    assert result == StringAnswer(value="73")


@pytest.mark.asyncio
async def test_wrong_route_or_metadata_is_rejected_before_source_access() -> None:
    question = Question(
        task_id="python-task",
        question="What is the final numeric output?",
        file_name="python-task.py",
    )
    called = False

    async def provider(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
    ) -> bytes:
        nonlocal called
        called = True
        return b"print(1)\n"

    solver = PythonCodeSolver(provider)
    with pytest.raises(PythonCodeSolverError, match="wrong route"):
        await solver.solve(question, analysis(), SolverRoute.STRUCTURED_TABLE.value)
    with pytest.raises(PythonCodeSolverError, match="metadata"):
        await solver.solve(
            question,
            analysis(task_class=TaskClass.DEEP_RESEARCH, modality=Modality.WEB),
            SolverRoute.CODE.value,
        )

    assert called is False
