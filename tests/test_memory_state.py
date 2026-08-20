import unittest
from unittest.mock import patch

from memory.state import (
    DEFAULT_ACTIONS,
    IntentType,
    LearningPhase,
    MemoryOp,
    StepPhase,
    allowable_actions,
    extract_state,
    is_shell_command,
    prior_for,
)


class ShellCommandDetectionTests(unittest.TestCase):
    def test_empty_and_none(self):
        self.assertFalse(is_shell_command(""))
        self.assertFalse(is_shell_command("   "))
        self.assertFalse(is_shell_command(None))  # type: ignore[arg-type]

    def test_builtins_without_path(self):
        self.assertTrue(is_shell_command("pwd"))
        self.assertTrue(is_shell_command("cd /tmp"))
        self.assertTrue(is_shell_command("echo hello"))
        self.assertTrue(is_shell_command("export FOO=bar"))
        self.assertTrue(is_shell_command("source ~/.bashrc"))

    def test_path_and_env_assignment(self):
        self.assertTrue(is_shell_command("./run.sh"))
        self.assertTrue(is_shell_command("../bin/tool"))
        self.assertTrue(is_shell_command("~/bin/job"))
        self.assertTrue(is_shell_command("PORT=8080 python app.py"))

    def test_operators_without_spaces(self):
        self.assertTrue(is_shell_command("ls|grep foo"))
        self.assertTrue(is_shell_command("make && make test"))
        self.assertTrue(is_shell_command("cmd; other"))

    def test_natural_language_is_not_command(self):
        self.assertFalse(is_shell_command("how do I list files?"))
        self.assertFalse(is_shell_command("what is the current directory"))
        self.assertFalse(is_shell_command("test this idea"))
        self.assertFalse(is_shell_command("set the timeout"))
        self.assertFalse(is_shell_command("help me debug this"))

    def test_ambiguous_builtin_with_flags_is_command(self):
        self.assertTrue(is_shell_command("test -f ./README.md"))
        self.assertTrue(is_shell_command("set -e"))

    def test_strips_prompt_and_fences(self):
        self.assertTrue(is_shell_command("$ pwd"))
        self.assertTrue(is_shell_command("`cd /tmp`"))


class ExtractStateTests(unittest.TestCase):
    def test_chat_query_when_store_empty(self):
        state = extract_state("what failed last time?", 0)
        self.assertEqual(state.intent_type, IntentType.CHAT_QUERY.value)
        self.assertTrue(state.store_empty)
        self.assertEqual(state.mem_size_bin, 0)
        self.assertEqual(state.step_phase, StepPhase.EARLY.value)
        self.assertEqual(state.learning_phase, LearningPhase.COLD.value)
        self.assertFalse(state.is_stuck)

    def test_error_recovery_from_nonzero_exit(self):
        state = extract_state("ls", 4, last_exit_code=1)
        self.assertEqual(state.intent_type, IntentType.ERROR_RECOVERY.value)

    def test_command_intent(self):
        state = extract_state("cd src", 4, last_exit_code=0)
        self.assertEqual(state.intent_type, IntentType.COMMAND.value)
        self.assertFalse(state.store_empty)

    def test_stuck_from_consecutive_failures(self):
        state = extract_state("make", 3, consecutive_failures=2)
        self.assertTrue(state.is_stuck)

    def test_stuck_from_repeated_command(self):
        state = extract_state(
            "make",
            3,
            last_command="make",
            current_command="make",
        )
        self.assertTrue(state.is_stuck)

    def test_step_and_learning_bins(self):
        early = extract_state("pwd", 0, step_index=7)
        mid = extract_state("pwd", 0, step_index=8)
        late = extract_state("pwd", 0, step_index=18)
        self.assertEqual(early.step_phase, "early")
        self.assertEqual(mid.step_phase, "mid")
        self.assertEqual(late.step_phase, "late")

        cold = extract_state("pwd", 0, task_index=15)
        warm = extract_state("pwd", 0, task_index=16)
        self.assertEqual(cold.learning_phase, "cold")
        self.assertEqual(warm.learning_phase, "warm")

    def test_mem_size_bin_matches_paper(self):
        self.assertEqual(extract_state("pwd", 0).mem_size_bin, 0)
        self.assertEqual(extract_state("pwd", 9).mem_size_bin, 0)
        self.assertEqual(extract_state("pwd", 10).mem_size_bin, 1)
        self.assertEqual(extract_state("pwd", 50).mem_size_bin, 5)
        self.assertEqual(extract_state("pwd", 999).mem_size_bin, 5)

    def test_cwd_bin(self):
        self.assertEqual(extract_state("pwd", 0, unique_cwds=2).cwd_bin, 0)
        self.assertEqual(extract_state("pwd", 0, unique_cwds=3).cwd_bin, 1)
        self.assertEqual(extract_state("pwd", 0, unique_cwds=20).cwd_bin, 4)

    def test_guards_bad_numeric_inputs(self):
        state = extract_state("pwd", -5, consecutive_failures=-2, step_index=-1)
        self.assertTrue(state.store_empty)
        self.assertFalse(state.is_stuck)
        self.assertEqual(state.step_phase, "early")

    def test_to_key_is_stable_and_hashable(self):
        state = extract_state("pwd", 12, last_exit_code=0, step_index=9, task_index=20)
        self.assertEqual(state.to_key(), "command:mid:0:0:1:1:0:warm")
        {state: "ok"}  # hashable
        {state.to_key(): "ok"}


class AllowableActionTests(unittest.TestCase):
    def test_default_set_has_nine_actions(self):
        self.assertEqual(len(DEFAULT_ACTIONS), 9)
        ops = {action.op for action in DEFAULT_ACTIONS}
        self.assertEqual(
            ops,
            {
                MemoryOp.RETRIEVE,
                MemoryOp.PLANINJECT,
                MemoryOp.RE_RETRIEVE,
                MemoryOp.CONSOLIDATE,
                MemoryOp.FORGET,
                MemoryOp.NOOP,
            },
        )

    def test_empty_store_only_noop(self):
        state = extract_state("pwd", 0)
        allowed = allowable_actions(state)
        self.assertEqual([a.op for a in allowed], [MemoryOp.NOOP])

    def test_empty_store_with_plan_allows_inject(self):
        state = extract_state("pwd", 0, plan_available=True)
        ops = {a.op for a in allowable_actions(state)}
        self.assertEqual(ops, {MemoryOp.NOOP, MemoryOp.PLANINJECT})

    def test_sparse_store_still_allows_retrieve(self):
        state = extract_state("pwd", 3)
        ops = {a.op for a in allowable_actions(state)}
        self.assertIn(MemoryOp.RETRIEVE, ops)
        self.assertIn(MemoryOp.CONSOLIDATE, ops)
        self.assertIn(MemoryOp.FORGET, ops)
        self.assertNotIn(MemoryOp.RE_RETRIEVE, ops)

    def test_stuck_with_memory_allows_reretrieve(self):
        state = extract_state("make", 3, consecutive_failures=2)
        ops = {a.op for a in allowable_actions(state)}
        self.assertIn(MemoryOp.RE_RETRIEVE, ops)

    def test_priors_match_paper(self):
        by_op = {action.op: prior_for(action) for action in DEFAULT_ACTIONS}
        self.assertEqual(by_op[MemoryOp.RETRIEVE], 0.5)
        self.assertEqual(by_op[MemoryOp.PLANINJECT], 0.3)
        self.assertEqual(by_op[MemoryOp.RE_RETRIEVE], 0.1)
        self.assertEqual(by_op[MemoryOp.CONSOLIDATE], 0.0)
        self.assertEqual(by_op[MemoryOp.FORGET], -0.1)
        self.assertEqual(by_op[MemoryOp.NOOP], -0.2)


class WhichLookupTests(unittest.TestCase):
    def test_path_hit_is_command(self):
        with patch("memory.state.shutil.which", side_effect=lambda name: "/usr/bin/git" if name == "git" else None):
            self.assertTrue(is_shell_command("git status"))

    def test_unknown_token_without_syntax_is_chat(self):
        with patch("memory.state.shutil.which", return_value=None):
            self.assertFalse(is_shell_command("blargleschnort please"))


if __name__ == "__main__":
    unittest.main()
