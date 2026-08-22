import os
import shutil
import subprocess
import uuid

CURRENT_CWD = os.getcwd()

# Cap roughly in the Terminus/KIRA class (~30KB) to limit context bloat.
MAX_OUTPUT_CHARS = 30000
COMMAND_TIMEOUT_SEC = 120


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if not text or len(text) <= max_chars:
        return text
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]

    truncate_count = len(text) - max_chars
    marker = f"\n\n[...{truncate_count} CHARACTERS TRUNCATED BY EXECUTOR...]\n\n"

    return head + marker + tail


def run_command(cmd: str):
    global CURRENT_CWD
    bash_path = shutil.which("bash") or "/bin/bash"

    marker = uuid.uuid4().hex
    # Run on its own line (not "&&") so heredocs terminate correctly,
    # and capture $? explicitly so a trailing `pwd` doesn't clobber the real exit code.
    wrapped_cmd = f"{cmd}\n__rc=$?\npwd\necho {marker}:$__rc"

    try:
        if bash_path and os.path.isfile(bash_path):
            res = subprocess.run(
                [bash_path, "-c", wrapped_cmd],
                capture_output=True,
                text=True,
                cwd=CURRENT_CWD,
                timeout=COMMAND_TIMEOUT_SEC,
            )
        else:
            res = subprocess.run(
                wrapped_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=CURRENT_CWD,
                timeout=COMMAND_TIMEOUT_SEC,
            )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        timeout_msg = (
            f"Command timed out after {COMMAND_TIMEOUT_SEC} seconds.\n{stderr}"
        )
        return truncate_output(stdout), truncate_output(timeout_msg), 124

    stdout = res.stdout or ""
    stderr = res.stderr or ""
    returncode = res.returncode

    lines = stdout.rstrip("\n").split("\n") if stdout else []
    if lines and lines[-1].startswith(f"{marker}:"):
        try:
            returncode = int(lines[-1].split(":", 1)[1])
        except ValueError:
            pass
        lines = lines[:-1]

    if lines:
        new_dir = lines[-1].strip()
        if os.path.isdir(new_dir):
            CURRENT_CWD = new_dir
            lines = lines[:-1]

    stdout = "\n".join(lines)
    return truncate_output(stdout), truncate_output(stderr), returncode
