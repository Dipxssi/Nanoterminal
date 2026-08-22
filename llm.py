import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
try:
    from langfuse import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
load_dotenv()


def get_xai_api_key() -> str | None:
    return os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")


def require_grok_api_key() -> str:
    key = get_xai_api_key()
    if not key:
        raise SystemExit(
            "Grok provider selected but no API key found.\n"
            "Add to .env in the project root (one line, no quotes needed):\n"
            "  XAI_API_KEY=xai-...\n"
            "Get a key at https://console.x.ai"
        )
    return key


def get_groq_api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY")


def require_groq_api_key() -> str:
    key = get_groq_api_key()
    if not key:
        raise SystemExit(
            "Groq provider selected but no API key found.\n"
            "Add to .env in the project root:\n"
            "  GROQ_API_KEY=gsk_...\n"
            "Get a free key at https://console.groq.com"
        )
    return key

_IS_LINUX_ENV = os.environ.get("NANOTERMINAL_ENV", "").lower() == "linux"

# Strongest free-tier model that works on typical new API keys.
# gemini-2.5-pro is retired for new keys; gemini-3.1-pro-preview needs paid quota.
# gemini-3.6-flash free tier is very tight (~20 RPD) — prefer 3.5-flash for Harbor.
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_EXTRACT_MODEL = "gemini-2.5-flash"
DEFAULT_GROK_MODEL = "grok-4-1-fast-non-reasoning"
DEFAULT_GROQ_MODEL = "groq/compound-mini"
DEFAULT_THINKING_BUDGET = 8192
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Cloudflare (in front of Groq/xAI) blocks Python-urllib's default User-Agent (error 1010).
_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def get_model_name() -> str:
    return os.environ.get("NANOTERMINAL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_extract_model_name() -> str:
    return (
        os.environ.get("NANOTERMINAL_EXTRACT_MODEL", DEFAULT_EXTRACT_MODEL).strip()
        or DEFAULT_EXTRACT_MODEL
    )


def get_llm_provider() -> str:
    """gemini (default), groq, or grok."""
    raw = os.environ.get("NANOTERMINAL_LLM_PROVIDER", "").strip().lower()
    if raw in ("grok", "xai"):
        return "grok"
    if raw == "groq":
        return "groq"
    if raw in ("gemini", "google"):
        return "gemini"
    if get_groq_api_key():
        return "groq"
    if get_xai_api_key():
        return "grok"
    return "gemini"


def get_grok_model_name() -> str:
    return (
        os.environ.get("NANOTERMINAL_GROK_MODEL", DEFAULT_GROK_MODEL).strip()
        or DEFAULT_GROK_MODEL
    )


def get_groq_model_name() -> str:
    return (
        os.environ.get("NANOTERMINAL_GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
        or DEFAULT_GROQ_MODEL
    )


def _quota_exhausted_message(message: str) -> bool:
    lower = (message or "").lower()
    return "quota" in lower and ("exceeded" in lower or "exhausted" in lower)


def _billing_blocked_message(message: str) -> bool:
    lower = (message or "").lower()
    return (
        "permission-denied" in lower
        or "doesn't have any credits" in lower
        or "no credits" in lower
        or "billing" in lower
    )


def _cloudflare_blocked_message(message: str) -> bool:
    return "error code: 1010" in (message or "").lower()


_groq_last_call_ts = 0.0


def _groq_qa_max_tokens() -> int:
    raw = os.environ.get("NANOTERMINAL_GROQ_QA_MAX_TOKENS", "512")
    try:
        return max(128, int(raw))
    except ValueError:
        return 512


def _parse_retry_seconds(body: str) -> float | None:
    """Parse Groq/OpenAI-style 'try again in Xs' / 'Xms' from error JSON."""
    if not body:
        return None
    match = re.search(r"try again in ([0-9.]+)\s*s", body, re.I)
    if match:
        return float(match.group(1)) + 0.5
    match = re.search(r"try again in ([0-9.]+)\s*ms", body, re.I)
    if match:
        return (float(match.group(1)) / 1000.0) + 0.2
    return None


def _groq_throttle() -> None:
    """Free Groq tier is tight on TPM — pace requests."""
    global _groq_last_call_ts
    min_gap = float(os.environ.get("NANOTERMINAL_GROQ_MIN_INTERVAL", "2.0"))
    elapsed = time.time() - _groq_last_call_ts
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed)


def _mark_groq_call() -> None:
    global _groq_last_call_ts
    _groq_last_call_ts = time.time()


def _http_api_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": _HTTP_USER_AGENT,
    }


def _openai_compatible_chat(
    *,
    url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    provider_label: str,
    missing_key_hint: str,
    max_tokens: int | None = None,
    max_attempts: int = 6,
) -> str:
    import json
    import urllib.error
    import urllib.request

    if not api_key:
        print(f"\n[{provider_label} failed: {missing_key_hint}]")
        return ""

    body_obj: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if max_tokens is not None:
        body_obj["max_tokens"] = max_tokens
    payload = json.dumps(body_obj).encode("utf-8")

    last_error = None
    for attempt in range(max_attempts):
        req = urllib.request.Request(
            url,
            data=payload,
            headers=_http_api_headers(api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()
            if content:
                return content
            # Reasoning models (e.g. openai/gpt-oss-20b) may exhaust max_tokens on
            # internal reasoning before emitting content.
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                return str(reasoning).strip()
            return ""
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {body[:400]}"
            if _cloudflare_blocked_message(body):
                print(
                    f"\n[{provider_label} failed: Cloudflare blocked the request "
                    f"(error 1010). Requests never reach {provider_label}'s API.]"
                )
                return ""
            if e.code in (401, 403) or _billing_blocked_message(body):
                print(f"\n[{provider_label} failed: {last_error}]")
                return ""
            if e.code == 429 and _quota_exhausted_message(body):
                print(f"\n[{provider_label} failed: daily quota exhausted for {model}]")
                return ""
            if e.code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                wait = _parse_retry_seconds(body) or (5 * (attempt + 1))
                print(f"\n[{provider_label} {e.code} — retrying in {wait:.1f}s...]")
                time.sleep(wait)
                continue
            print(f"\n[{provider_label} failed: {last_error}]")
            return ""
        except Exception as e:
            print(f"\n[{provider_label} failed: {e}]")
            return ""
    if last_error:
        print(f"\n[{provider_label} exhausted retries: {last_error}]")
    return ""


def _thinking_enabled() -> bool:
    raw = os.environ.get("NANOTERMINAL_THINKING", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _thinking_budget() -> int:
    raw = os.environ.get("NANOTERMINAL_THINKING_BUDGET", str(DEFAULT_THINKING_BUDGET))
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_THINKING_BUDGET


def _thinking_config(model: str | None = None) -> types.ThinkingConfig | None:
    if not _thinking_enabled():
        return None
    name = (model or get_model_name()).lower()
    # Gemini 3.x prefers thinking_level; 2.5 uses thinking_budget.
    if name.startswith("gemini-3") or "gemini-3." in name:
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
    budget = _thinking_budget()
    if budget <= 0:
        return None
    return types.ThinkingConfig(thinking_budget=budget)


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
You must complete the entire task without any human intervention. There is no user to ask.
Your submission via `finish_task` is FINAL.

Guidelines:
1. Call `execute_bash` to run commands and inspect outputs.
2. Work from /app unless the task says otherwise.
3. Create or edit files with standard shell tools (cat, printf, tee, heredocs, sed, etc.).
4. Prefer lightweight tools already on PATH. Avoid installing heavy packages unless required.
5. If a command fails with "command not found", use `which` / `command -v` / `ls` / `apt-cache` to inspect the environment before retrying.
6. Compile, run tests, and verify outputs against the original instructions before finishing.
7. Do NOT call `finish_task` until you have verified the objective is fully met. Premature finish will be rejected.
8. Once verified, call `finish_task` with a brief summary of what was accomplished and how you verified it.
"""
    return """
You are an AI CLI agent running inside Git Bash on Windows.
Your goal is to complete user tasks step-by-step without asking the user for help.

Guidelines:
1. Call `execute_bash` to run commands and inspect outputs.
2. Verify the objective is met before calling `finish_task`.
3. Once the task objective is completely met and verified, IMMEDIATELY call `finish_task`.
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
            description=(
                "Call only when the user's request is completely fulfilled AND you have "
                "verified results against the original instructions (tests, file checks, etc.)."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "summary": types.Schema(
                        type="STRING",
                        description=(
                            "Brief summary of what was accomplished and how it was verified."
                        ),
                    )
                },
                required=["summary"],
            ),
        ),
    ]
)


def _generate_config(*, with_tools: bool, model: str | None = None) -> types.GenerateContentConfig:
    thinking = _thinking_config(model)
    kwargs: dict = {
        "temperature": 0.2,
    }
    if thinking is not None:
        kwargs["thinking_config"] = thinking
    if with_tools:
        kwargs["system_instruction"] = _system_instruction()
        kwargs["tools"] = [TOOLS]
        kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        )
        kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )
    else:
        kwargs["temperature"] = 0.0
    return types.GenerateContentConfig(**kwargs)


@observe(name="gemini_command_generation")
def ask_gemini(contents: list | str) -> tuple[str, dict]:
    """Returns (function_name, arguments_dict)"""
    last_error = None
    model = get_model_name()
    for attempt in range(5):
        try:
            res = get_client().models.generate_content(
                model=model,
                contents=contents,
                config=_generate_config(with_tools=True, model=model),
            )

            if res.function_calls:
                call = res.function_calls[0]
                return call.name, call.args

            return "execute_bash", {"command": (res.text or "").strip()}

        except errors.APIError as e:
            last_error = e
            if e.code in (429, 500, 502, 503, 504):
                wait = 15 * (attempt + 1)
                if e.code == 429:
                    wait = max(wait, 30 * (attempt + 1))
                print(f"\n[{e.code} — {e.message} — retrying in {wait}s...]")
                time.sleep(wait)
            else:
                raise e

    raise Exception(f"Exhausted retries. Last error: {last_error}")


def ask_gemini_raw(prompt: str) -> str:
    """Lightweight Gemini text generation (no tools). Never raises to callers."""
    model = get_extract_model_name()
    last_error = None
    for attempt in range(3):
        try:
            res = get_client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return res.text or ""
        except errors.APIError as e:
            last_error = e
            if e.code == 429 and _quota_exhausted_message(e.message):
                print(f"\n[extract failed: daily quota exhausted for {model}]")
                return ""
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                wait = 20 * (attempt + 1)
                print(f"\n[extract {e.code} — retrying in {wait}s...]")
                time.sleep(wait)
                continue
            print(f"\n[extract failed: {e.code} — {e.message}]")
            return ""
        except Exception as e:
            print(f"\n[extract failed: {e}]")
            return ""
    if last_error:
        print(f"\n[extract exhausted retries: {last_error}]")
    return ""


def ask_grok_raw(prompt: str) -> str:
    """Lightweight Grok/xAI text via OpenAI-compatible chat completions."""
    return _openai_compatible_chat(
        url=XAI_CHAT_URL,
        api_key=get_xai_api_key(),
        model=get_grok_model_name(),
        prompt=prompt,
        provider_label="grok",
        missing_key_hint="set XAI_API_KEY in .env",
    )


def ask_groq_raw(prompt: str, *, max_tokens: int | None = 512) -> str:
    """Lightweight Groq text via OpenAI-compatible chat completions."""
    _groq_throttle()
    result = _openai_compatible_chat(
        url=GROQ_CHAT_URL,
        api_key=get_groq_api_key(),
        model=get_groq_model_name(),
        prompt=prompt,
        provider_label="groq",
        missing_key_hint="set GROQ_API_KEY in .env",
        max_tokens=max_tokens,
    )
    _mark_groq_call()
    return result


def ask_text_raw(
    prompt: str,
    *,
    provider: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Provider-routed raw text call (Lychee extract, LoCoMo QA, etc.)."""
    name = (provider or get_llm_provider()).strip().lower()
    if name in ("grok", "xai"):
        return ask_grok_raw(prompt)
    if name == "groq":
        tokens = 512 if max_tokens is None else max_tokens
        return ask_groq_raw(prompt, max_tokens=tokens)
    return ask_gemini_raw(prompt)


def active_text_model_label(provider: str | None = None) -> str:
    name = (provider or get_llm_provider()).strip().lower()
    if name in ("grok", "xai"):
        return f"grok:{get_grok_model_name()}"
    if name == "groq":
        return f"groq:{get_groq_model_name()}"
    return f"gemini:{get_extract_model_name()}"
