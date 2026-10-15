"""Small parity check against each historical backup's original Argmax rollout."""

from __future__ import annotations

import sys
from pathlib import Path

from selected_rollout import generate_routes, load_policy, selected_starts


PROJECT = Path(r"D:\Project\球粒计算\GB-MACO_python")
DATA = PROJECT / "benachmark_STSP_Complex_Multiopt_Test" / "test" / "core_032_128" / "n032"
CASE = DATA / "mstsp_n032_rectangle_01.tsp"


def check(name: str) -> None:
    backup = PROJECT / "最优结果的实验" / name
    policy, _, _ = load_policy(backup, "cpu")
    from gb_maco_POMO.pomo import rollout
    from gb_maco_POMO.runner import prepare_tsp

    prepared = prepare_tsp(CASE)
    starts = selected_starts(32, 32)
    actual = generate_routes(policy, prepared, starts, samples=0, seed=42)
    expected = rollout(policy, prepared, sample_actions=False).routes
    assert actual == expected, f"Argmax routes differ for {name}"
    print(f"{name}: all 32 Argmax routes match original rollout")


if __name__ == "__main__":
    check(sys.argv[1])
