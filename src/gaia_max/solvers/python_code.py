"""Static inspection and constrained execution for attached Python programs."""

from __future__ import annotations

import ast
import asyncio
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import tokenize
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from gaia_max.domain import AnswerType, Modality, Question, StringAnswer, TaskClass
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis


class PythonCodeSolverError(ValueError):
    """Base failure for an invalid, unsafe, or incompatible Python task."""


class UnsafePythonSourceError(PythonCodeSolverError):
    """Static policy found capabilities that this sandbox does not permit."""

    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__("unsafe Python source: " + "; ".join(violations))


class SandboxExecutionError(PythonCodeSolverError):
    """The isolated child failed, exceeded a limit, or produced invalid output."""


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Small defaults suitable for the course's deterministic code attachment."""

    wall_seconds: float = 2.0
    cpu_seconds: int = 2
    memory_bytes: int = 256 * 1024 * 1024
    max_output_bytes: int = 64 * 1024
    max_source_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if self.wall_seconds <= 0 or self.cpu_seconds < 1:
            raise ValueError("sandbox time limits must be positive")
        if self.memory_bytes < 32 * 1024 * 1024:
            raise ValueError("sandbox memory limit is too small to start Python reliably")
        if self.max_output_bytes < 1 or self.max_source_bytes < 1:
            raise ValueError("sandbox byte limits must be positive")


@dataclass(frozen=True, slots=True)
class PythonInspection:
    """Answer-free facts from a passing AST inspection."""

    node_count: int
    imported_modules: tuple[str, ...]
    checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PythonExecutionResult:
    """Final numeric text plus deterministic replay audit facts."""

    final_numeric_output: str
    runs_completed: int
    fresh_process_outputs_match: bool
    static_replay_supported: bool
    static_replay_confirmed: bool
    inspection: PythonInspection


CodeSourceProvider = Callable[
    [Question, ReconciledTaskAnalysis],
    Awaitable[bytes],
]


_ALLOWED_IMPORT_ROOTS = {
    "collections",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "math",
    "statistics",
}
_BANNED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
_BANNED_ATTRIBUTES = {
    "connect",
    "fork",
    "kill",
    "load",
    "loads",
    "open",
    "popen",
    "request",
    "run",
    "save",
    "socket",
    "spawn",
    "system",
    "unlink",
}
_NUMERIC_TYPES = (int, float, complex)
_REAL_NUMBER = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.imported_modules: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.partition(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                self.violations.append(f"line {node.lineno}: import {root!r} is blocked")
            else:
                self.imported_modules.add(root)
            if alias.asname is not None and (
                alias.asname in _BANNED_NAMES or alias.asname.startswith("__")
            ):
                self.violations.append(
                    f"line {node.lineno}: import alias {alias.asname!r} is blocked"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").partition(".")[0]
        if node.level or root not in _ALLOWED_IMPORT_ROOTS:
            self.violations.append(f"line {node.lineno}: import from {root!r} is blocked")
        else:
            self.imported_modules.add(root)
        if any(alias.name == "*" for alias in node.names):
            self.violations.append(f"line {node.lineno}: wildcard imports are blocked")
        if any(alias.name.startswith("_") for alias in node.names):
            self.violations.append(f"line {node.lineno}: private imports are blocked")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BANNED_NAMES or (node.id.startswith("__") and node.id != "__name__"):
            self.violations.append(f"line {node.lineno}: name {node.id!r} is blocked")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        normalized = node.attr.casefold()
        if node.attr.startswith("_") or normalized in _BANNED_ATTRIBUTES:
            self.violations.append(f"line {node.lineno}: attribute {node.attr!r} is blocked")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.violations.append(f"line {node.lineno}: class definitions are blocked")


def decode_python_source(content: bytes, *, max_bytes: int = 1024 * 1024) -> str:
    """Decode source using Python's declared-encoding rules without executing it."""

    if not content or len(content) > max_bytes or b"\x00" in content:
        raise PythonCodeSolverError("Python source is empty, oversized, or contains NUL bytes")
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(content).readline)
        return content.decode(encoding)
    except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
        raise PythonCodeSolverError("Python source encoding is invalid") from exc


def inspect_python_source(source: str, *, max_nodes: int = 10_000) -> PythonInspection:
    """Parse source and enforce a narrow pure-computation capability policy."""

    try:
        tree = ast.parse(source, filename="<attachment>")
    except SyntaxError as exc:
        raise PythonCodeSolverError("Python source has invalid syntax") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > max_nodes:
        raise UnsafePythonSourceError(("AST node limit exceeded",))
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    violations = tuple(dict.fromkeys(visitor.violations))
    if violations:
        raise UnsafePythonSourceError(violations)
    return PythonInspection(
        node_count=len(nodes),
        imported_modules=tuple(sorted(visitor.imported_modules)),
        checks=(
            "ast_parse",
            "ast_node_limit",
            "pure_stdlib_import_allowlist",
            "dynamic_execution_blocklist",
            "filesystem_network_process_blocklist",
        ),
    )


def _numeric_literal(text: str) -> str:
    candidate = text.strip()
    if _REAL_NUMBER.fullmatch(candidate):
        return candidate
    try:
        value = ast.literal_eval(candidate)
    except (SyntaxError, ValueError) as exc:
        raise SandboxExecutionError("final non-empty output line is not numeric") from exc
    if isinstance(value, bool) or not isinstance(value, _NUMERIC_TYPES):
        raise SandboxExecutionError("final non-empty output line is not numeric")
    return candidate


def extract_final_numeric_output(stdout: str) -> str:
    """Return the last non-empty output line only when it is a numeric literal."""

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SandboxExecutionError("program produced no non-empty stdout")
    return _numeric_literal(lines[-1])


class _SimpleNumericReplay:
    """Tiny independent evaluator for straight-line numeric assignments and print."""

    _binary_operators: ClassVar[
        dict[type[ast.operator], Callable[[Any, Any], Any]]
    ] = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left**right,
    }
    _unary_operators: ClassVar[
        dict[type[ast.unaryop], Callable[[Any], Any]]
    ] = {
        ast.UAdd: lambda value: +value,
        ast.USub: lambda value: -value,
    }

    def __init__(self) -> None:
        self.values: dict[str, int | float | complex] = {}
        self.last_printed: str | None = None

    def run(self, tree: ast.Module) -> str | None:
        try:
            for statement in tree.body:
                self._statement(statement)
        except (ArithmeticError, TypeError, ValueError, OverflowError, KeyError):
            return None
        return self.last_printed

    def _statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                raise ValueError
            self.values[target.id] = self._expression(node.value)
            return
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            left = self.values[node.target.id]
            node_type = type(node.op)
            operator = self._binary_operators.get(node_type)
            if operator is None:
                raise ValueError
            self.values[node.target.id] = self._bounded(
                operator(left, self._expression(node.value))
            )
            return
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
            and len(node.value.args) == 1
            and not node.value.keywords
        ):
            self.last_printed = str(self._expression(node.value.args[0]))
            return
        raise ValueError

    def _expression(self, node: ast.expr) -> int | float | complex:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, _NUMERIC_TYPES):
                raise ValueError
            return node.value
        if isinstance(node, ast.Name):
            return self.values[node.id]
        if isinstance(node, ast.UnaryOp):
            operator = self._unary_operators.get(type(node.op))
            if operator is None:
                raise ValueError
            return self._bounded(operator(self._expression(node.operand)))
        if isinstance(node, ast.BinOp):
            operator = self._binary_operators.get(type(node.op))
            if operator is None:
                raise ValueError
            left = self._expression(node.left)
            right = self._expression(node.right)
            if isinstance(node.op, ast.Pow) and isinstance(right, int) and abs(right) > 10_000:
                raise ValueError
            return self._bounded(operator(left, right))
        raise ValueError

    @staticmethod
    def _bounded(value: object) -> int | float | complex:
        if isinstance(value, bool) or not isinstance(value, _NUMERIC_TYPES):
            raise ValueError
        if isinstance(value, int) and value.bit_length() > 100_000:
            raise ValueError
        return value


def replay_simple_final_numeric(source: str) -> str | None:
    """Independently evaluate simple straight-line arithmetic, or decline safely."""

    tree = ast.parse(source, filename="<static-replay>")
    replayed = _SimpleNumericReplay().run(tree)
    return _numeric_literal(replayed) if replayed is not None else None


_BOOTSTRAP = textwrap.dedent(
    """
    import os
    import sys
    import builtins

    memory_bytes = int(sys.argv[2])
    cpu_seconds = int(sys.argv[3])
    output_bytes = int(sys.argv[4])

    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    source_path = sys.argv[1]
    with open(source_path, "rb") as source_file:
        code = compile(source_file.read(), source_path, "exec")
    allowed_imports = {
        "collections", "decimal", "fractions", "functools",
        "itertools", "math", "statistics",
    }
    blocked_builtins = {
        "breakpoint", "compile", "delattr", "eval", "exec", "exit",
        "getattr", "globals", "help", "input", "locals", "open", "quit",
        "setattr", "vars",
    }
    real_import = builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.partition(".")[0] not in allowed_imports:
            raise ImportError("sandbox import blocked")
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = {
        name: value for name, value in vars(builtins).items()
        if name not in blocked_builtins
    }
    safe_builtins["__import__"] = restricted_import
    namespace = {
        "__name__": "__main__",
        "__file__": source_path,
        "__builtins__": safe_builtins,
    }
    exec(code, namespace, namespace)
    """
).strip()


def _windows_peak_working_set_bytes(process: subprocess.Popen[bytes]) -> int | None:
    """Read peak child memory while the Popen process handle is still valid."""

    if os.name != "nt":
        return None
    import ctypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    raw_handle = getattr(process, "_handle", None)
    if raw_handle is None:
        return None
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.c_void_p(int(raw_handle))
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return max(
        int(counters.PeakWorkingSetSize),
        int(counters.PeakPagefileUsage),
        int(counters.PrivateUsage),
    )


class ConstrainedPythonExecutor:
    """Run inspected source twice in fresh bounded interpreter processes."""

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()

    def execute_verified(self, source: str) -> PythonExecutionResult:
        encoded = source.encode("utf-8")
        if len(encoded) > self.limits.max_source_bytes:
            raise PythonCodeSolverError("Python source exceeds sandbox source limit")
        inspection = inspect_python_source(source)
        first = self._run_once(source)
        second = self._run_once(source)
        first_numeric = extract_final_numeric_output(first)
        second_numeric = extract_final_numeric_output(second)
        if first_numeric != second_numeric:
            raise SandboxExecutionError("fresh process outputs disagree")

        replayed = replay_simple_final_numeric(source)
        replay_supported = replayed is not None
        replay_confirmed = replayed == first_numeric if replay_supported else False
        if replay_supported and not replay_confirmed:
            raise SandboxExecutionError("static arithmetic replay disagrees with execution")
        return PythonExecutionResult(
            final_numeric_output=first_numeric,
            runs_completed=2,
            fresh_process_outputs_match=True,
            static_replay_supported=replay_supported,
            static_replay_confirmed=replay_confirmed,
            inspection=inspection,
        )

    def _run_once(self, source: str) -> str:
        with tempfile.TemporaryDirectory(prefix="gaia-python-") as temporary:
            root = Path(temporary)
            source_path = root / "attachment.py"
            bootstrap_path = root / "sandbox_bootstrap.py"
            stdout_path = root / "stdout.bin"
            stderr_path = root / "stderr.bin"
            source_path.write_text(source, encoding="utf-8", newline="\n")
            bootstrap_path.write_text(_BOOTSTRAP, encoding="utf-8", newline="\n")
            # Some Windows virtual environments expose a launcher that spawns the
            # real interpreter. Killing or metering that launcher would leave the
            # sandboxed child alive, so execute the base interpreter directly.
            interpreter = (
                getattr(sys, "_base_executable", sys.executable)
                if os.name == "nt"
                else sys.executable
            )
            command = [
                interpreter,
                "-I",
                "-S",
                "-B",
                "-X",
                "utf8",
                str(bootstrap_path),
                str(source_path),
                str(self.limits.memory_bytes),
                str(self.limits.cpu_seconds),
                str(self.limits.max_output_bytes),
            ]
            environment = {
                key: os.environ[key]
                for key in ("SYSTEMROOT", "WINDIR")
                if key in os.environ
            }
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            failure: SandboxExecutionError | None = None
            with (
                stdout_path.open("wb") as stdout_file,
                stderr_path.open("wb") as stderr_file,
                subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    shell=False,
                    creationflags=creation_flags,
                ) as process,
            ):
                deadline = time.monotonic() + self.limits.wall_seconds
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        process.kill()
                        process.wait()
                        if os.name == "nt":
                            time.sleep(0.1)
                        failure = SandboxExecutionError(
                            "Python sandbox exceeded wall timeout"
                        )
                        break
                    if (
                        stdout_path.stat().st_size > self.limits.max_output_bytes
                        or stderr_path.stat().st_size > self.limits.max_output_bytes
                    ):
                        process.kill()
                        process.wait()
                        if os.name == "nt":
                            time.sleep(0.1)
                        failure = SandboxExecutionError(
                            "Python sandbox exceeded output limit"
                        )
                        break
                    peak_memory = _windows_peak_working_set_bytes(process)
                    if peak_memory is not None and peak_memory > self.limits.memory_bytes:
                        process.kill()
                        process.wait()
                        if os.name == "nt":
                            time.sleep(0.1)
                        failure = SandboxExecutionError(
                            "Python sandbox exceeded memory limit"
                        )
                        break
                    time.sleep(0.01)
                peak_memory = _windows_peak_working_set_bytes(process)
                if (
                    failure is None
                    and peak_memory is not None
                    and peak_memory > self.limits.memory_bytes
                ):
                    failure = SandboxExecutionError(
                        "Python sandbox exceeded memory limit"
                    )

            if failure is not None:
                raise failure

            stdout = stdout_path.read_bytes()
            stderr = stderr_path.read_bytes()
            if (
                len(stdout) >= self.limits.max_output_bytes
                or len(stderr) >= self.limits.max_output_bytes
            ):
                raise SandboxExecutionError("Python sandbox exceeded output limit")
            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise SandboxExecutionError(
                    f"Python sandbox exited with code {process.returncode}: {detail[:300]}"
                )
            try:
                return stdout.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SandboxExecutionError("Python sandbox stdout is not UTF-8") from exc


class PythonCodeSolver:
    """Resolve validated source bytes and return verified final numeric text."""

    def __init__(
        self,
        source_provider: CodeSourceProvider,
        executor: ConstrainedPythonExecutor | None = None,
    ) -> None:
        self._source_provider = source_provider
        self._executor = executor or ConstrainedPythonExecutor()

    async def solve(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> object:
        if route != SolverRoute.CODE.value:
            raise PythonCodeSolverError("Python-code solver received wrong route")
        if (
            analysis.task_class is not TaskClass.CODE
            or analysis.modality is not Modality.PYTHON
            or analysis.output_contract.answer_type is not AnswerType.STRING
            or question.file_name is None
            or Path(question.file_name).suffix.casefold() != ".py"
        ):
            raise PythonCodeSolverError("Python-code solver received incompatible task metadata")

        content = await self._source_provider(question, analysis)
        if not isinstance(content, bytes):
            raise PythonCodeSolverError("source provider did not return bytes")
        source = decode_python_source(content, max_bytes=self._executor.limits.max_source_bytes)
        result = await asyncio.to_thread(self._executor.execute_verified, source)
        return StringAnswer(value=result.final_numeric_output)


def register_python_code_solver(
    registry: SolverRegistry,
    source_provider: CodeSourceProvider,
    executor: ConstrainedPythonExecutor | None = None,
) -> PythonCodeSolver:
    """Register B04 explicitly with its answer-blind source dependency."""

    instance = PythonCodeSolver(source_provider, executor)
    registry.register(SolverRoute.CODE, instance)
    return instance
