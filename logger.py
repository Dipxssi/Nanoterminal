import json
import os
import time
from datetime import datetime


class TrajectoryLogger:

    def __init__(self, log_dir: str = "trajectories"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.log_dir, f"run_{timestamp}.json")

        self.trajectory = {
            "session_id": f"run_{timestamp}",
            "start_time": datetime.now().isoformat(),
            "goal": "",
            "turns": [],
            "status": "IN_PROGRESS",
        }

    def set_goal(self, goal: str):
        self.trajectory["goal"] = goal
        self._save()

    def log_turn(
        self,
        turn_num: int,
        command: str,
        stdout: str,
        stderr: str,
        exit_code: int,
        latency_seconds: float,
        is_high_risk: bool = False,
    ):
        turn_data = {
            "turn": turn_num,
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "exit_code": exit_code,
            "latency_seconds": round(latency_seconds, 2),
            "is_high_risk": is_high_risk,
            "stdout": stdout,
            "stderr": stderr,
        }
        self.trajectory["turns"].append(turn_data)
        self._save()

    def finalize(self, status: str = "SUCCESS"):
        self.trajectory["status"] = status
        self.trajectory["end_time"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.trajectory, f, indent=2)