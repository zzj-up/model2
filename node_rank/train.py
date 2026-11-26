"""Train explicit within-compound target preferences with a pairwise ranking loss."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .io import (DEFAULT_CHECKPOINT, allowed_path, checkpoint_payload, load_checkpoint,
                 load_features, read_labels, replace_run_file, save_new, sha256, utc_now)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-pairs-per-group", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.max_pairs_per_group, args.threads) < 1 or not 0 < args.lr < 1:
        parser.error("Positive epochs/pair limit/threads and 0 < lr < 1 are required.")
    torch.set_num_threads(args.threads)
    features_hash, labels_hash = sha256(args.features), sha256(args.labels)
    artifact = load_features(args.features)
    samples = artifact["samples"]
    groups = read_labels(args.labels, samples)
    audit = {"samples": len(samples), "comparable_groups": len(groups),
             "samples_used_in_comparisons": len({i for _, pairs in groups for pair in pairs for i in pair}),
             "comparable_pairs": sum(len(pairs) for _, pairs in groups),
             "feature_sha256": features_hash, "labels_sha256": labels_hash,
             "objective": "mean_by_group_pairwise_softplus_negative_score_difference",
             "higher_relevance_is_better": True, "missing_interactions_used_as_negatives": False,
             "task": "same_compound_different_target_ranking", "split": "explicit_training_inputs_only"}
    print(json.dumps(audit, indent=2), flush=True)
    if args.dry_run:
        return
    if args.run_dir is None:
        parser.error("--run-dir is required for training; choose a new directory.")
    run_dir = allowed_path(args.run_dir)
    if run_dir.exists():
        raise FileExistsError("Run directory exists. Choose a new run directory; old weights are preserved.")
    checkpoint_hash = sha256(args.checkpoint)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    if checkpoint["training_steps"] != 0:
        raise ValueError("This first-training entry requires an initialized node ranker, not old/previously trained weights.")
    if sha256(args.features) != features_hash or sha256(args.labels) != labels_hash:
        raise RuntimeError("Training inputs changed while loading.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    run_dir.mkdir(parents=True, exist_ok=False)
    save_new(run_dir / "run_config.json", {**audit, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "created_at_utc": utc_now(), "feature_policy": artifact["feature_policy"],
             "initial_checkpoint_sha256": checkpoint_hash,
             "validation_performed": False, "external_encoders_updated": False}, json_file=True)
    steps, metrics = 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(groups)))
        random.shuffle(order)
        group_losses = []
        sampled = 0
        for group_index in order:
            _, pairs = groups[group_index]
            chosen = random.sample(pairs, min(len(pairs), args.max_pairs_per_group))
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            # Accumulate one comparison at a time to bound autograd memory.
            for better, worse in chosen:
                high = model(samples[better])["score"]
                low = model(samples[worse])["score"]
                loss = F.softplus(-(high - low))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite ranking loss; no checkpoint published for this update.")
                (loss / len(chosen)).backward()
                total += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            sampled += len(chosen)
            group_losses.append(total / len(chosen))
        metric = {"epoch": epoch, "training_steps": steps,
                  "mean_training_group_loss": sum(group_losses) / len(group_losses),
                  "sampled_comparisons": sampled}
        metrics.append(metric)
        with (run_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric) + "\n")
        payload = checkpoint_payload(model, checkpoint["seed"], steps, artifact["feature_policy"],
                    objective=audit["objective"], training_feature_sha256=features_hash,
                    labels_sha256=labels_hash, epoch=epoch, optimization_seed=args.seed,
                    validation_performed=False, initial_checkpoint_sha256=checkpoint_hash)
        replace_run_file(run_dir / "last.pt", payload)
        print(json.dumps(metric), flush=True)
    if sha256(args.features) != features_hash or sha256(args.labels) != labels_hash or sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("An input artifact changed during training; invalidate this run.")
    load_checkpoint(run_dir / "last.pt")
    save_new(run_dir / "training_summary.json", {**audit, "status": "complete", "epochs": args.epochs,
        "training_steps": steps, "last_training_loss": metrics[-1]["mean_training_group_loss"],
        "checkpoint_sha256": sha256(run_dir / "last.pt"), "validation_performed": False,
        "performance_claim": "training_fit_only_no_generalization_estimate",
        "completed_at_utc": utc_now()}, json_file=True)


if __name__ == "__main__":
    main()
