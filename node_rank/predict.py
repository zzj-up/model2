"""Score explicitly posed complexes, then rank targets within each compound."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .io import allowed_path, rank_rows, save_new, sha256, utc_now
from .target_ranking import DEFAULT_RANK_CHECKPOINT, load_for_ranking, ranked_inputs, training_objective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Sharded feature index (.json) or policy-matched feature bundle (.pt).")
    parser.add_argument("--compound-id", help="Optionally select exactly this compound_id (canonical SMILES) in a larger index.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RANK_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-untrained-diagnostics", action="store_true")
    parser.add_argument("--save-attention", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(args.threads)
    output = allowed_path(args.output_dir)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing predictions will not be overwritten.")
    before = sha256(args.checkpoint)
    model, checkpoint = load_for_ranking(args.checkpoint, args.device)
    diagnostic = checkpoint["training_steps"] == 0
    if diagnostic and not args.allow_untrained_diagnostics:
        raise ValueError("Task parameters are untrained. Train with comparable target rankings first, or explicitly use --allow-untrained-diagnostics for runtime checks only.")
    features_hash = sha256(args.features)
    model.eval().requires_grad_(False)
    rows, attention = [], []
    with torch.inference_mode():
        for identity, sample in ranked_inputs(args.features, checkpoint, args.compound_id):
            result = model(sample, return_attention=args.save_attention)
            rows.append(identity | {
                "score": float(result["score"].cpu()),
                "status": "UNTRAINED_DIAGNOSTIC_ONLY" if diagnostic else "trained_raw_score"})
            if args.save_attention:
                attention.append({"pair_id": sample["pair_id"], **{
                    key: result[key].detach().cpu() for key in ("contact_index", "contact_distance", "ban_attention")
                }})
    rows = rank_rows(rows)
    if sha256(args.checkpoint) != before or sha256(args.features) != features_hash:
        raise RuntimeError("Checkpoint/features changed during prediction.")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "scores.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        columns = ["compound_id", "rank", "target_id"]
        if any("target_name" in row for row in rows):
            columns.append("target_name")
        writer = csv.DictWriter(stream, fieldnames=columns + ["pair_id", "score", "candidate_count", "status"])
        writer.writeheader()
        writer.writerows(rows)
    if attention:
        save_new(output / "contact_attention.pt", {"schema": "node_rank_contact_attention_v1", "records": attention,
            "interpretation": "learned_attention_weights_not_binding_energy_or_causal_explanation"})
    summary = {"status": "completed", "created_at_utc": utc_now(), "pairs": len(rows),
               "compounds": len({row["compound_id"] for row in rows}),
               "compound_groups_with_multiple_targets": len({row["compound_id"] for row in rows if row["candidate_count"] > 1}),
               "checkpoint_sha256": before, "features_sha256": features_hash,
               "task_training_steps": checkpoint["training_steps"], "diagnostic_only": diagnostic,
               "training_objective": training_objective(checkpoint), "score_meaning": checkpoint["score_meaning"],
               "output_scale": "raw_model_score", "output_file": "scores.csv", "not_probability": True,
               "ranking": "descending_within_same_compound_competition_ties",
               "new_training_performed": False, "task_weights_unchanged": True,
               "note": "Scores from random task weights have no validated screening meaning." if diagnostic else
                       "Ranks are relative to the supplied candidate targets and require appropriate validation."}
    save_new(output / "summary.json", summary, json_file=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
