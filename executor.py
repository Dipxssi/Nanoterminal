import subprocess
import shutil
import os 

CURRENT_CWD = os.getcwd()

MAX_OUTPUT_CHARS = 2000

def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if not text or len(text) <= max_chars:
        return text
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]
    
    truncate_count = len(text) - max_chars
    marker= f"\n\n[...{truncate_output} CHARACTERS TRUNCATED BY EXECUTOR...]\n\n"
    
    return head + marker + tail

def run_command(cmd: str):
    global CURRENT_CWD
    bash_path = shutil.which("bash")
    
    wrapped_cmd = f"{cmd} && pwd"
    
    if bash_path:
        res = subprocess.run([bash_path, "-c", cmd], capture_output=True, text=True, cwd = CURRENT_CWD)
    else:
        res = subprocess.run(wrapped_cmd, shell=True, capture_output=True, text=True, cwd= CURRENT_CWD)
    stdout = res.stdout
    stderr = res.stderr
    returncode = res.returncode
        
    if returncode == 0 and stdout:
        lines = stdout.strip().split("\n")
        
        new_dir = lines[-1].strip()
        
        if os.path.exists(new_dir):
            CURRENT_CWD = new_dir
            stdout = "\n".join(lines[:-1])
            
    truncate_stdout = truncate_output(stdout)
    truncate_stderr = truncate_output(stderr)
    
    return truncate_stdout , truncate_stderr , returncode 