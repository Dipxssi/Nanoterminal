import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
try:
    from langfuse import observe
except ImportError:
    # Safe fallback decorator if langfuse is missing or misconfigured
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
load_dotenv()

_IS_LINUX_ENV = os.environ.get("NANOTERMINAL_ENV", "").lower() == "linux"

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _shell_description() -> str:
    if _IS_LINUX_ENV:
        return "Executes a bash command in the Linux task environment."
    return "Executes a bash command in Git Bash on the user's terminal."


def _system_instruction() -> str:
    if _IS_LINUX_ENV:
        return """
You are an AI CLI agent running inside a Linux terminal in a Docker task environment.
Your goal is to complete user tasks step-by-step.

Guidelines:
1. Call `execute_bash` to run commands and inspect outputs.
2. Work from /app unless the task says otherwise.
3. Create or edit files with standard shell tools (cat, printf, tee, heredocs, etc.).
4. Compile and test your work before calling `finish_task`.
5. Once the task objective is completely met and verified, IMMEDIATELY call `finish_task`.
"""
    return """
You are an AI CLI agent running inside Git Bash on Windows.
Your goal is to complete user tasks step-by-step.

Guidelines:
1. Call `execute_bash` to run commands and inspect outputs.
2. Once the task objective is completely met, IMMEDIATELY call `finish_task`.
"""


TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="execute_bash",
            description=_shell_description(),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "command": types.Schema(
                        type="STRING",
                        description="The single bash command or chained commands (using &&) to execute.",
                    )
                },
                required=["command"],
            ),
        ),
        types.FunctionDeclaration(
            name="finish_task",
            description="Call this function when the user's request is completely fulfilled and verified.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "summary": types.Schema(
                        type="STRING",
                        description="Brief summary of what was accomplished.",
                    )
                },
                required=["summary"],
            ),
        ),
    ]
)

@observe(name="gemini_command_generation")
def ask_gemini(contents: list | str) -> tuple[str, dict]:
    """Returns (function_name, arguments_dict)"""
    last_error = None
    for attempt in range(5):
        try:
            res = get_client().models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_system_instruction(),
                    tools=[TOOLS],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY"
                        )
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    temperature=0.2,
                ),
            )

            if res.function_calls:
                call = res.function_calls[0]
                return call.name, call.args
            
            return "execute_bash", {"command": res.text.strip()}

        except errors.APIError as e:
            last_error = e
            if e.code in (429, 500, 502, 503, 504):
                wait = 10 * (attempt + 1)
                print(f"\n[{e.code} — {e.message} — retrying in {wait}s...]")
                time.sleep(wait)
            else:
                raise e

    raise Exception(f"Exhausted retries. Last error: {last_error}")