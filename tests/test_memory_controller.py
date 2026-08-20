import json
import tempfile
import unittest
from pathlib import Path

from memory.controller import ALPHA, GAMMA, LAMBDA_EFF, R_FAIL, R_SUCC, MemConBandit
from memory.state import MemoryOp, extract_state, prior_for


def _populated_state():
    return extract_state("pwd", total_records_in_store=12, last_exit_code=0)


def _empty_state():
    return extract_state("pwd", total_records_in_store=0)


def _plan_empty_state():
    return extract_state("pwd", total_records_in_store=0, plan_available=True)


class MemConBanditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "q.json")
        self.bandit = MemConBandit(persist_path=self.path, flush_interval=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cold_start_prefers_retrieve_prior_not_noop(self):
        action = self.bandit.select_action(_populated_state())
        self.assertEqual(action.op, MemoryOp.RETRIEVE)
        self.assertEqual(action.label, "shallow")

    def test_empty_store_selects_noop(self):
        action = self.bandit.select_action(_empty_state())
        self.assertEqual(action.op, MemoryOp.NOOP)

    def test_empty_store_with_plan_prefers_inject(self):
        action = self.bandit.select_action(_plan_empty_state())
        self.assertEqual(action.op, MemoryOp.PLANINJECT)

    def test_unvisited_span_feasible_set_before_ucb(self):
        state = _populated_state()
        seen = []
        for _ in range(7):
            seen.append(self.bandit.select_action(state).op)
        self.assertEqual(seen[0], MemoryOp.RETRIEVE)
        self.assertIn(MemoryOp.NOOP, seen)
        self.assertIn(MemoryOp.CONSOLIDATE, seen)
        self.assertIn(MemoryOp.FORGET, seen)

    def test_reward_success_includes_efficiency(self):
        reward = self.bandit.compute_reward(True, steps=10)
        expected = R_SUCC + LAMBDA_EFF * (1.0 - 10 / 30)
        self.assertAlmostEqual(reward, expected)

    def test_reward_failure_subtracts_r_fail(self):
        reward = self.bandit.compute_reward(False, steps=30)
        self.assertAlmostEqual(reward, -R_FAIL)

    def test_reverse_discount_credits_last_decision_more(self):
        state = _populated_state()
        first = self.bandit.select_action(state)
        second = self.bandit.select_action(state)
        self.assertNotEqual(first.to_key(), second.to_key())

        reward = self.bandit.end_episode(success=True, steps=10)
        q = self.bandit.q_table[state.to_key()]
        q_first = q[first.to_key()]
        q_second = q[second.to_key()]

        credit_first = (GAMMA ** 1) * reward
        credit_second = (GAMMA ** 0) * reward
        expected_first = prior_for(first) + ALPHA * (credit_first - prior_for(first))
        expected_second = prior_for(second) + ALPHA * (credit_second - prior_for(second))
        self.assertAlmostEqual(q_first, expected_first)
        self.assertAlmostEqual(q_second, expected_second)
        self.assertGreater(q_second, q_first)

    def test_begin_episode_discards_partial_trajectory(self):
        state = _populated_state()
        self.bandit.select_action(state)
        self.bandit.begin_episode()
        reward = self.bandit.end_episode(success=True, steps=1)
        # Counts still rose on select, but Q stays at the prior (no credits).
        q_row = self.bandit.q_table[state.to_key()]
        self.assertTrue(all(abs(q_row[a.to_key()] - prior_for(a)) < 1e-9 for a in self.bandit.all_actions))
        self.assertGreater(reward, 0)

    def test_persists_and_reloads(self):
        state = _populated_state()
        action = self.bandit.select_action(state)
        self.bandit.end_episode(success=True, steps=5)

        reloaded = MemConBandit(persist_path=self.path, flush_interval=1)
        key = state.to_key()
        self.assertIn(key, reloaded.q_table)
        self.assertGreater(reloaded.counts[key][action.to_key()], 0)
        self.assertNotAlmostEqual(
            reloaded.q_table[key][action.to_key()],
            prior_for(action),
        )

    def test_load_fills_missing_actions_with_priors(self):
        state = _populated_state()
        key = state.to_key()
        payload = {
            "q_table": {key: {"RETRIEVE:shallow:1:3:1": 0.9}},
            "counts": {key: {"RETRIEVE:shallow:1:3:1": 4}},
        }
        Path(self.path).write_text(json.dumps(payload), encoding="utf-8")
        loaded = MemConBandit(persist_path=self.path)
        noop = next(a for a in loaded.all_actions if a.op is MemoryOp.NOOP)
        self.assertAlmostEqual(loaded.q_table[key][noop.to_key()], prior_for(noop))
        self.assertEqual(loaded.counts[key][noop.to_key()], 0)
        self.assertAlmostEqual(loaded.q_table[key]["RETRIEVE:shallow:1:3:1"], 0.9)

    def test_corrupt_checkpoint_starts_fresh(self):
        Path(self.path).write_text("{not json", encoding="utf-8")
        loaded = MemConBandit(persist_path=self.path)
        self.assertEqual(loaded.q_table, {})
        action = loaded.select_action(_populated_state())
        self.assertEqual(action.op, MemoryOp.RETRIEVE)


if __name__ == "__main__":
    unittest.main()
