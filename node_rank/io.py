"""Safe artifacts and explicit within-compound ranking labels."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "node_rank_checkpoint_v1"
FEATURE_SCHEMA = "node_rank_features_v1"
DEFAULT_CHECKPOINT = ROOT / "node_rank/checkpoints/untrained_node_rank.pt"
SCORE_MEANING = "unitless_within_compound_target_ranking_score_higher_is_better"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in ("model.py", "graph.py")}


def allowed_path(path):
    path = Path(path).resolve()
    if path.is_relative_to((ROOT / "quarantine").resolve()):
        raise ValueError("Isolated historical data/weights cannot be used by node_rank.")
    return path


def save_new(path, payload, *, json_file=False):
    """Publish atomically without overwriting any artifact."""
    path = allowed_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if json_file:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_run_file(path, payload):
    """Only for progress within a newly created training run."""
    path = allowed_path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize(output=DEFAULT_CHECKPOINT, seed=20261117, config=None):
    from .model import NodeRanker
    if Path(output).exists():
        raise FileExistsError(f"Choose a new checkpoint path: {output}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = NodeRanker(config)
    payload = checkpoint_payload(model, seed, 0, None)
    save_new(output, payload)
    save_new(Path(output).with_suffix(".json"), {
        key: value for key, value in payload.items() if key != "model_state"
    } | {"checkpoint_sha256": sha256(output), "task_parameters": sum(p.numel() for p in model.parameters())}, json_file=True)
    return payload


def checkpoint_payload(model, seed, training_steps, feature_policy, **metadata):
    from .model import ARCHITECTURE
    return {
        "schema": SCHEMA, "architecture": ARCHITECTURE,
        "created_at_utc": utc_now(), "config": model.config, "seed": seed,
        "training_steps": training_steps, "task_status": "trained" if training_steps else "untrained",
        "score_meaning": SCORE_MEANING, "external_encoders": "official_pretrained_frozen",
        "feature_policy": feature_policy, "model_source_sha256": source_hashes(),
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, **metadata,
    }


def load_checkpoint(path, device="cpu"):
    from .model import ARCHITECTURE, NodeRanker
    payload = torch.load(allowed_path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("This is not a node_rank checkpoint. Old affinity weights are incompatible.")
    if (payload.get("architecture") != ARCHITECTURE or payload.get("score_meaning") != SCORE_MEANING
            or payload.get("model_source_sha256") != source_hashes()):
        raise ValueError("Checkpoint score contract or model source differs; use its recorded architecture version.")
    steps = payload.get("training_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("Invalid task training history.")
    if payload.get("task_status") != ("trained" if steps else "untrained"):
        raise ValueError("Task status and training steps disagree.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(payload["seed"])
        model = NodeRanker(payload["config"])
    if steps == 0:
        expected = model.state_dict()
        if expected.keys() != payload["model_state"].keys() or any(
            not torch.equal(value, payload["model_state"][key]) for key, value in expected.items()
        ):
            raise ValueError("A claimed untrained checkpoint differs from its seeded initialization.")
    for value in payload["model_state"].values():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint parameters.")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), payload


def load_features(path):
    artifact = torch.load(allowed_path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("schema") != FEATURE_SCHEMA:
        raise ValueError("Use real per-node features from rank-features; old pooled caches cannot be reused.")
    samples = artifact.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(artifact.get("feature_policy"), dict):
        raise ValueError("A nonempty feature artifact with feature_policy is required.")
    identifiers = set()
    identity_to_target, target_to_identity = {}, {}
    for sample in samples:
        for key in ("pair_id", "compound_id", "target_id"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError(f"Missing sample identifier: {key}")
        if sample["pair_id"] in identifiers:
            raise ValueError("Duplicate pair_id in feature artifact.")
        identifiers.add(sample["pair_id"])
        for key, width in (("sequence_features", 2560), ("ligand_features", 96), ("pocket_features", 96)):
            value = sample.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f"{key} must have width {width}; pooled feature caches are incompatible.")
        # Graph validation also checks dimensions, coordinates and mapping.
        from .graph import build_graph
        build_graph(sample)
        if "protein_sequence" in sample:
            sequence = sample["protein_sequence"]
            if len(sequence) != len(sample["sequence_features"]):
                raise ValueError("Per-residue feature rows must cover the complete protein sequence.")
            target = sample["target_id"]
            if target in target_to_identity and target_to_identity[target] != sequence:
                raise ValueError("One target_id names different protein sequences.")
            if sequence in identity_to_target and identity_to_target[sequence] != target:
                raise ValueError("One sequence is assigned different target_ids.")
            target_to_identity[target], identity_to_target[sequence] = sequence, target
    return artifact


def read_labels(path, samples):
    """Explicit audit group asserts comparable measurements, not inferred negatives."""
    with allowed_path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"pair_id", "relevance", "comparison_group", "endpoint"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Labels CSV requires pair_id,relevance,comparison_group,endpoint; higher relevance wins.")
        labels = list(reader)
    by_id = {sample["pair_id"]: i for i, sample in enumerate(samples)}
    groups, seen, compound_targets = defaultdict(list), set(), set()
    for label in labels:
        pair_id = label["pair_id"]
        if pair_id not in by_id or pair_id in seen:
            raise ValueError(f"Unknown or duplicate label pair_id: {pair_id}")
        seen.add(pair_id)
        index = by_id[pair_id]
        sample = samples[index]
        compound_target = (sample["compound_id"], sample["target_id"])
        if compound_target in compound_targets:
            raise ValueError("Select exactly one pose per compound-target before ranking training.")
        compound_targets.add(compound_target)
        relevance = float(label["relevance"])
        group, endpoint = label["comparison_group"].strip(), label["endpoint"].strip()
        if not math.isfinite(relevance) or not group or not endpoint:
            raise ValueError("Ranking relevance must be finite; comparison_group and endpoint must be explicit.")
        groups[(sample["compound_id"], group, endpoint)].append((index, relevance))
    if seen != set(by_id):
        raise ValueError("Labels must cover every feature sample exactly once; prepare a separate training feature file.")
    comparisons = []
    for key, entries in sorted(groups.items()):
        pairs = []
        for (i, a), (j, b) in itertools.combinations(entries, 2):
            if a == b:
                continue
            pairs.append((i, j) if a > b else (j, i))
        if pairs:
            comparisons.append((key, pairs))
    if not comparisons:
        raise ValueError("No comparable within-compound cross-target pairs. Each compound needs >=2 distinct targets with unequal explicit, comparable labels. Single-target PDBbind records and unobserved pairs are insufficient.")
    return comparisons


def rank_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not math.isfinite(row["score"]):
            raise ValueError("Cannot rank nonfinite scores.")
        grouped[row["compound_id"]].append(row)
    result = []
    for compound in sorted(grouped):
        group = sorted(grouped[compound], key=lambda row: (-row["score"], row["target_id"], row["pair_id"]))
        if len({row["target_id"] for row in group}) != len(group):
            raise ValueError("Select one pose per compound-target; implicit best-pose aggregation is disabled.")
        last_score, rank = None, 0
        for index, row in enumerate(group, 1):
            if last_score is None or row["score"] != last_score:
                rank = index
            result.append({**row, "rank": rank, "candidate_count": len(group)})
            last_score = row["score"]
    return result


def audit_manifest(path):
    from pbcnet_features import canonical_smiles, normalize_sequence
    records = json.loads(allowed_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected nonempty structure manifest list.")
    compounds = defaultdict(set)
    for record in records:
        compounds[canonical_smiles(record["smiles"])].add(normalize_sequence(record["protein_sequence"]))
    count = sum(len(targets) * (len(targets) - 1) // 2 for targets in compounds.values())
    return {"manifest": str(Path(path).resolve()), "manifest_sha256": sha256(path), "records": len(records),
            "compounds": len(compounds), "distinct_sequences": len(set().union(*compounds.values())),
            "compounds_with_multiple_targets": sum(len(x) > 1 for x in compounds.values()),
            "potential_within_compound_target_comparisons": count,
            "comparable_labels_verified": False, "training_performed": False,
            "note": "Explicit comparable labels remain required; missing interactions are not negatives."}
