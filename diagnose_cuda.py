"""Diagnose CUDA visibility for the current Python/process environment.

This script reports:
- Python executable and version
- PyTorch version and CUDA build information
- CUDA availability and device metadata
- `nvidia-smi` status
- Common environment variables that affect GPU visibility

Run with:

    python diagnose_cuda.py
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from textwrap import indent


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n== {title} ==")


def run_command(args: list[str]) -> tuple[int, str, str]:
    """Run a subprocess and capture exit code, stdout, and stderr."""
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover
        return 1, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def main() -> None:
    """Run CUDA diagnostics for the current environment."""
    print("CUDA environment diagnosis")

    print_section("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version:    {sys.version.splitlines()[0]}")
    print(f"Platform:   {platform.platform()}")

    print_section("Environment Variables")
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "PATH",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
    ]:
        value = os.environ.get(key, "<unset>")
        if key in {"PATH", "LD_LIBRARY_PATH"} and value != "<unset>":
            print(f"{key}:")
            print(indent(value, prefix="  "))
        else:
            print(f"{key}: {value}")

    print_section("PyTorch")
    try:
        import torch

        print(f"torch.__version__:      {torch.__version__}")
        print(f"torch.version.cuda:     {torch.version.cuda}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        print(f"torch.backends.cuda.is_built: {torch.backends.cuda.is_built()}")

        cudnn_available = torch.backends.cudnn.is_available()
        print(f"torch.backends.cudnn.is_available: {cudnn_available}")
        if cudnn_available:
            print(f"torch.backends.cudnn.version: {torch.backends.cudnn.version()}")

        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            print(f"current_device: {current}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(
                    f"device {idx}: name={props.name}, "
                    f"capability={props.major}.{props.minor}, "
                    f"total_memory={props.total_memory / 1024**3:.2f} GiB"
                )
        else:
            try:
                torch.cuda.init()
            except Exception as exc:
                print(f"torch.cuda.init() error: {type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"PyTorch import/inspection failed: {type(exc).__name__}: {exc}")

    print_section("nvidia-smi")
    nvidia_smi = shutil.which("nvidia-smi")
    print(f"binary: {nvidia_smi or '<not found>'}")
    if nvidia_smi:
        code, stdout, stderr = run_command(["nvidia-smi"])
        print(f"exit_code: {code}")
        if stdout:
            print("stdout:")
            print(indent(stdout, prefix="  "))
        if stderr:
            print("stderr:")
            print(indent(stderr, prefix="  "))

    print_section("Driver Devices")
    code, stdout, stderr = run_command(["bash", "-lc", "ls -l /dev/nvidia* 2>/dev/null || true"])
    if stdout:
        print(stdout)
    else:
        print("No /dev/nvidia* device nodes visible.")
        if stderr:
            print(indent(stderr, prefix="  "))


if __name__ == "__main__":
    main()
