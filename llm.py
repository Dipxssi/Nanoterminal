from google import genai
from google.genai import errors , types
import time
from langfuse import Langfuse
from langfuse.decorators import observe 

client = genai.Client()

BASH_TOOL = types.Tool(
  function_declarations=[
    types.FunctionDeclaration(
      name="execute_bash",
      description="Executes a bash command in Git Bash on the user's terminal.",
      parameters = types.Schema(
        type = "OBJECT",
        properties={
        "command": types.Schema(
          type="STRING",
          description="The single bash command or chained commands (using &&) to execute. ",
        )
      },
      required=["command"],
      
    ),
    )
  ]
)

SYSTEM_INSTRUCTION = """
You are an AI CLI agent running inside Git Bash on Windows.
Your goal is to complete user tasks by invoking the `execute_bash` function.

Guidelines:
1. Always call `execute_bash` with valid bash commands.
2. If creating multi-line files, write them using 'cat << \'EOF\' > filename' inside the command.
3. Inspect previous command execution results (STDOUT/STDERR) to decide your next action.


"""
@observe(name="gemini_command_generation")
def ask_gemini(contents: list | str) -> str:
    
    for attempt in range(3):
        try:
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[BASH_TOOL],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY"  
                        )
                    ),
                    temperature=0.2,
                ),
            )

            if res.function_calls:
                call = res.function_calls[0]
                return call.args.get("command", "").strip()
            
            return res.text.strip()

        except errors.APIError as e:
            if e.code == 429:
                print("\n[Rate limit hit. Waiting 15 seconds before retrying...]")
                time.sleep(15)
            else:
                raise e

    raise Exception("Exhausted retries due to rate limits.")
