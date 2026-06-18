from __future__ import annotations

from pathlib import Path

from mosaic.eval.ablations import run_ablations


def test_ablations_write_artifacts(tmp_path: Path) -> None:
    results = run_ablations(tmp_path, dataset="builtin")
    assert len(results) == 4
    assert (tmp_path / "ablation_results.json").exists()
    assert (tmp_path / "leaderboard.json").exists()
