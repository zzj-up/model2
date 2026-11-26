"""Initialize, inspect and audit the new node-level ranking model."""
import argparse
import json
from pathlib import Path

from .io import DEFAULT_CHECKPOINT, audit_manifest, initialize, sha256
from .target_ranking import DEFAULT_RANK_CHECKPOINT, load_for_ranking, training_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    init.add_argument("--seed", type=int, default=20261117)
    status = sub.add_parser("status")
    status.add_argument("--checkpoint", type=Path, default=DEFAULT_RANK_CHECKPOINT)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_manifest(args.manifest)
    else:
        if args.command == "init":
            initialize(args.output, args.seed)
        path = args.output if args.command == "init" else args.checkpoint
        model, payload = load_for_ranking(path)
        result = {key: payload[key] for key in ("architecture", "task_status", "training_steps", "score_meaning",
                                                        "external_encoders", "config")}
        result.update(checkpoint=str(path.resolve()), checkpoint_sha256=sha256(path),
                      task_parameters=sum(p.numel() for p in model.parameters()),
                      task_trained=payload["training_steps"] > 0,
                      training_objective=training_objective(payload),
                      ranking_output="raw_score_descending_within_compound",
                      screening_performance_validated=False,
                      old_model_preserved="fresh_dti/runs/pdbbind_20261116_1000_v1/model")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
