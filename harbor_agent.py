import os
import shlex
from pathlib import Path
from typing import override

from dotenv import load_dotenv
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    EnvVar,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Load project .env into the *host* process so ENV_VARS / _get_env see the key.
# Harbor does not auto-load .env; `$GEMINI_API_KEY` in the shell is often empty.
load_dotenv(Path(__file__).resolve().parent / ".env")

INSTALL_DIR = "/installed-agent/nanoterminal"
VENV_PYTHON = f"{INSTALL_DIR}/.venv/bin/python"
AGENT_LOG = "/logs/agent/nanoterminal.txt"

# Top-level modules copied into the container.
SOURCE_FILES = (
    "cli_entrypoint.py",
    "executor.py",
    "llm.py",
    "logger.py",
)

# Memory package modules (uploaded under INSTALL_DIR/memory/).
MEMORY_FILES = (
    "__init__.py",
    "buffer.py",
    "controller.py",
    "embeddings.py",
    "engine.py",
    "plans.py",
    "schemas.py",
    "segmenter.py",
    "state.py",
    "store.py",
)

# Runtime deps for the agent + MemCon memory layer.
PIP_PACKAGES = (
    "google-genai",
    "python-dotenv",
    "rich",
    "pydantic",
    "numpy",
    "fastembed",
)


class NanoTerminalAgent(BaseInstalledAgent):
    """Harbor adapter that installs and runs NanoTerminal inside the task container."""

    CLI_FLAGS = [
        CliFlag(
            "max_turns",
            cli="--max-turns",
            type="int",
            default=30,
        ),
        CliFlag(
            "debug_memory",
            cli="--debug-memory",
            type="bool",
            default=False,
        ),
    ]

    ENV_VARS = [
        EnvVar(
            kwarg="gemini_api_key",
            env="GEMINI_API_KEY",
            type="str",
            env_fallback="GEMINI_API_KEY",
        ),
    ]

    @staticmethod
    @override
    def name() -> str:
        return "nanoterminal"

    @override
    def version(self) -> str | None:
        return self._version or "0.1.0"

    @override
    def get_version_command(self) -> str | None:
        return f"{VENV_PYTHON} {INSTALL_DIR}/cli_entrypoint.py --version"

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent

    async def _upload_sources(self, environment: BaseEnvironment) -> None:
        project_root = self._project_root()

        for filename in SOURCE_FILES:
            source = project_root / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing NanoTerminal source file: {source}")
            await environment.upload_file(source, f"{INSTALL_DIR}/{filename}")

        memory_dir = project_root / "memory"
        if not memory_dir.is_dir():
            raise FileNotFoundError(f"Missing memory package directory: {memory_dir}")

        await self.exec_as_root(
            environment,
            command=f"mkdir -p {INSTALL_DIR}/memory && chmod -R a+rwX {INSTALL_DIR}/memory",
        )

        for filename in MEMORY_FILES:
            source = memory_dir / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing memory module: {source}")
            await environment.upload_file(
                source, f"{INSTALL_DIR}/memory/{filename}"
            )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && apt-get install -y "
                "python3 python3-pip python3-venv curl ca-certificates"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        await self.exec_as_root(
            environment,
            command=(
                f"mkdir -p {INSTALL_DIR} /logs/agent/trajectories "
                f"&& chmod -R a+rwX {INSTALL_DIR} /logs/agent"
            ),
        )

        await self._upload_sources(environment)

        packages = " ".join(PIP_PACKAGES)
        await self.exec_as_agent(
            environment,
            command=(
                f"python3 -m venv {INSTALL_DIR}/.venv && "
                f"{INSTALL_DIR}/.venv/bin/pip install --no-cache-dir {packages}"
            ),
            cwd=INSTALL_DIR,
        )

        # Warm the embedding model so the first task is not blocked on download.
        await self.exec_as_agent(
            environment,
            command=(
                f"{VENV_PYTHON} -c "
                "\"from memory.embeddings import EmbeddingModel; "
                "EmbeddingModel(); "
                "from memory.engine import MemoryEngine; "
                "print('memory-ok')\""
            ),
            cwd=INSTALL_DIR,
        )

        await self.exec_as_agent(
            environment,
            command=f"{VENV_PYTHON} {INSTALL_DIR}/cli_entrypoint.py --version",
            cwd=INSTALL_DIR,
        )

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)
        cli_flags = self.build_cli_flags()
        extra_flags = f"{cli_flags} " if cli_flags else ""

        env = {"NANOTERMINAL_ENV": "linux"}
        # Prefer Harbor --ae / --ak, but ignore empty values so a blank
        # `--ae GEMINI_API_KEY=$GEMINI_API_KEY` does not shadow .env.
        gemini_api_key = self._get_env("GEMINI_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Put it in .env, or pass "
                "`--env-file .env` / `--ae GEMINI_API_KEY=...`."
            )
        env["GEMINI_API_KEY"] = gemini_api_key

        await self.exec_as_agent(
            environment,
            command=(
                f"cd {INSTALL_DIR} && "
                f"{VENV_PYTHON} cli_entrypoint.py "
                f"{extra_flags}"
                f"--log-dir /logs/agent/trajectories "
                f"--prompt {escaped_instruction} "
                f"2>&1 | stdbuf -oL tee {AGENT_LOG}"
            ),
            env=env,
        )
