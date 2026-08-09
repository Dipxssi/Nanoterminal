import re

# High-risk patterns that require explicit user confirmation
HIGH_RISK_PATTERNS = [
    r"\brm\s+-[rfRF]+",          # Recursive/force delete (rm -rf, rm -f)
    r"\bdel\b",                  # Windows file delete
    r"\brmdir\s+/s\b",           # Windows recursive directory delete
    r"\bgit\s+reset\s+--hard\b", # Hard git reset (destroys uncommitted work)
    r"\bgit\s+clean\s+-[fF]+",   # Force cleaning untracked git files
    r"\bformat\b",               # Disk format commands
    r"\bdd\b",                   # Raw disk write operations
    r"\bdrop\b",                 # SQL drop operations
    r"\bchmod\s+-R\s+777\b",     # Overly permissive permission changes
    r">\s*/dev/sd",              # Overwriting raw block devices
]

def analyze_command_risk(cmd: str) -> tuple[bool, str]:
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True, f"Matched risky pattern: '{pattern}'"

    return False, "Low risk"