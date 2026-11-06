import subprocess
from pathlib import Path
from typing import List, Union
from src.utils.logger import console


def run_command(cmd: List[Union[str, Path]], desc: str = "") -> subprocess.CompletedProcess:
    """Execute a system command securely via subprocess and report errors."""
    str_cmd = [str(arg) for arg in cmd]
    if desc:
        console.print(f"[info]Executing {desc}...[/info]")

    result = subprocess.run(
        str_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        console.print(f"[error]Command failed with exit code {result.returncode}[/error]")
        console.print(f"[error]{result.stderr.strip()}[/error]")
        raise RuntimeError(f"Command execution error: {' '.join(str_cmd)}")

    return result