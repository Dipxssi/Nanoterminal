import shlex
from pathlib import Path
from typing import override

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    EnvVar,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

INSTALL_DIR = "/installed-agent/nanoterminal"
VENV_PYTHON = f"{INSTALL_DIR}/.venv/bin/python"
AGENT_LOG = "/logs/agent/nanoterminal.txt"

SOURCE_FILES = (
    "cli_entrypoint.py",
    "executor.py",
    "llm.py",
    "logger.py",
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

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        project_root = self._project_root()

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
            command=f"mkdir -p {INSTALL_DIR} /logs/agent/trajectories && chmod -R a+rwX {INSTALL_DIR} /logs/agent",
        )

        for filename in SOURCE_FILES:
            source = project_root / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing NanoTerminal source file: {source}")
            await environment.upload_file(source, f"{INSTALL_DIR}/{filename}")

        await self.exec_as_agent(
            environment,
            command=(
                f"python3 -m venv {INSTALL_DIR}/.venv && "
                f"{INSTALL_DIR}/.venv/bin/pip install --no-cache-dir "
                "google-genai python-dotenv rich"
            ),
            cwd=INSTALL_DIR,
        )

        await self.exec_as_agent(
            environment,
            command=f"{VENV_PYTHON} {INSTALL_DIR}/cli_entrypoint.py --version",
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
        gemini_api_key = self._get_env("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Pass it with "
                "`harbor run --ae GEMINI_API_KEY=...` or set it in the environment."
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
