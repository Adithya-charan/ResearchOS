"""Sandbox runners. Experiment code never runs in the calling process.

DockerSandbox is the real isolation boundary: disposable, non-root, no network, read-only root, capped CPU / memory /
processes, all capabilities dropped. SubprocessSandbox is a *convenience fallback with no security boundary*: it only
gives a separate interpreter, a scrubbed environment, a scratch directory, a timeout and an output cap. It cannot block
network or file access, so it refuses model-generated code unless the caller explicitly accepts that."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from research.experiments.spec import ExperimentSpec

DOCKER_IMAGE = "python:3.12-slim"
_FORBIDDEN_IMPORTS = {"socket", "subprocess", "ctypes", "multiprocessing", "http", "urllib", "ftplib", "smtplib",
                      "requests", "shutil", "pty", "signal"}
_FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__"}


class SandboxRefused(Exception):
    """The sandbox will not run this spec (policy, not a failure of the code)."""


@dataclass
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    duration_s: float
    isolation: str  # "container" | "process"
    network_blocked: bool
    notes: list[str] = field(default_factory=list)


class Sandbox(Protocol):
    name: str
    container: bool

    def run(self, spec: ExperimentSpec, workdir: Path) -> SandboxResult: ...


def screen_code(code: str) -> list[str]:
    """Static screen for the weak fallback only: obvious network/process modules. Trivially bypassable - NOT a boundary."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        bad += [n for n in names if n.split(".")[0] in _FORBIDDEN_IMPORTS]
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_CALLS:
            bad.append(node.id)
    return sorted(set(bad))


def _cap(data: bytes, limit: int) -> tuple[str, bool]:
    return data[:limit].decode("utf-8", errors="replace"), len(data) > limit


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class SubprocessSandbox:
    name = "subprocess"
    container = False

    def __init__(self, allow_model_generated: bool = False, python: str | None = None):
        self.allow_model_generated = allow_model_generated
        self.python = python or sys.executable

    def run(self, spec: ExperimentSpec, workdir: Path) -> SandboxResult:
        if spec.generated_by_model and not self.allow_model_generated:
            raise SandboxRefused("model-generated code needs a container sandbox (Docker); the process fallback has no isolation")
        blocked = screen_code(spec.code)
        if blocked:
            raise SandboxRefused("code refused by the static screen: " + ", ".join(blocked))
        env = {k: v for k, v in os.environ.items() if k in {"SYSTEMROOT", "PATH", "TEMP", "TMP", "LANG"}}
        env.update({"PYTHONHASHSEED": str(spec.seed), "PYTHONDONTWRITEBYTECODE": "1", "EXPERIMENT_SEED": str(spec.seed)})
        t0 = time.perf_counter()
        timed_out = False
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            proc = subprocess.Popen([self.python, "-I", "-B", "main.py"], cwd=workdir, env=env, stdout=out_f,
                                    stderr=err_f, stdin=subprocess.DEVNULL)
            try:
                proc.wait(timeout=spec.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_tree(proc)
            dur = time.perf_counter() - t0
            out_f.seek(0)
            err_f.seek(0)
            out, t1 = _cap(out_f.read(spec.max_output_bytes + 1), spec.max_output_bytes)
            err, t2 = _cap(err_f.read(spec.max_output_bytes + 1), spec.max_output_bytes)
        return SandboxResult(None if timed_out else proc.returncode, out, err, timed_out, t1 or t2, dur, "process", False,
                             ["NO ISOLATION: network and filesystem are not restricted; memory/CPU limits are not enforced. "
                              "Run only code you have read."])


class DockerSandbox:
    name = "docker"
    container = True

    def __init__(self, image: str = DOCKER_IMAGE, docker: str | None = None):
        self.image = image
        self.docker = docker or shutil.which("docker")

    @property
    def available(self) -> bool:
        if not self.docker:
            return False
        try:
            return subprocess.run([self.docker, "info"], capture_output=True, timeout=15).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def command(self, spec: ExperimentSpec, workdir: Path, name: str) -> list[str]:
        """The exact docker invocation (separate from run() so it can be reviewed and tested without Docker)."""
        out_dir = (workdir / "out").resolve()
        return [self.docker or "docker", "run", "--rm", "--name", name,
                "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--user", "65534:65534", "--memory", f"{spec.memory_mb}m", "--memory-swap", f"{spec.memory_mb}m",
                "--cpus", str(spec.cpus), "--pids-limit", "64", "--tmpfs", "/tmp:rw,size=64m,noexec",
                "-v", f"{workdir.resolve()}:/work:ro", "-v", f"{out_dir}:/work/out:rw",
                "-w", "/work", "-e", f"EXPERIMENT_SEED={spec.seed}", "-e", f"PYTHONHASHSEED={spec.seed}",
                "-e", "PYTHONDONTWRITEBYTECODE=1", self.image, "python", "-I", "-B", "main.py"]

    def run(self, spec: ExperimentSpec, workdir: Path) -> SandboxResult:
        if not self.available:
            raise SandboxRefused("Docker is not available on this machine")
        (workdir / "out").mkdir(exist_ok=True)
        name = f"researchos-exp-{os.getpid()}-{int(time.time() * 1000)}"
        t0 = time.perf_counter()
        timed_out = False
        try:
            cp = subprocess.run(self.command(spec, workdir, name), capture_output=True, timeout=spec.timeout_s + 10)
            code, out_b, err_b = cp.returncode, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired as e:
            timed_out, code, out_b, err_b = True, None, e.stdout or b"", e.stderr or b""
        finally:
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True)  # container cleanup, always
        out, t1 = _cap(out_b, spec.max_output_bytes)
        err, t2 = _cap(err_b, spec.max_output_bytes)
        return SandboxResult(code, out, err, timed_out, t1 or t2, time.perf_counter() - t0, "container", True,
                             ["Container: no network, read-only root, non-root user, capped memory/CPU/processes."])


def default_sandbox(allow_weak: bool = False) -> Sandbox:
    """Docker if it works; otherwise the weak process fallback, only when the caller explicitly allows it."""
    d = DockerSandbox()
    if d.available:
        return d
    if allow_weak:
        return SubprocessSandbox()
    raise SandboxRefused("Docker is not available; pass allow_weak=True to use the non-isolating process fallback")
