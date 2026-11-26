"""Check validation units and the allowlisted trainer resume without fitting a model."""
from pathlib import Path
from types import SimpleNamespace
import math

import torch

from .io import sha256
from .trainer_compatibility import reports_only_upgrade
from .train_pdbbind import _validate


class PredeterminedScores(torch.nn.Module):
    def forward(self, score):
        return {"score": torch.tensor(score)}


def main():
    rows = [{"p_affinity": 8., "affinity_endpoint": "Kd"}, {"p_affinity": 4., "affinity_endpoint": "Ki"}]
    dataset = SimpleNamespace(rows=rows, sample=(0.5, -2.0).__getitem__)
    report = _validate(PredeterminedScores(), dataset, {"mean": 6., "std": 2.}, SimpleNamespace(reason=None))
    assert report["standardized_mse"] == 0.625
    assert report["rmse_score"] == math.sqrt(0.625) and report["mae_score"] == 0.75
    assert set(report) == {"samples", "standardized_mse", "rmse_score", "mae_score", "per_endpoint"}
    assert report["per_endpoint"]["Kd"]["mae_score"] == 0.5 and report["per_endpoint"]["Ki"]["mae_score"] == 1.
    previous = "54e55323a0459a707ee667c88c78409bca3db7ddca17f82c2ba0e9261632c5ed"
    current = sha256(Path(__file__).with_name("train_pdbbind.py"))
    assert reports_only_upgrade(previous, current)
    assert not reports_only_upgrade("0" * 64, current) and not reports_only_upgrade(previous, "0" * 64)
    print("Validation score units, endpoint aggregates, and exact-version resume compatibility: OK")


if __name__ == "__main__":
    main()
