"""Verified inputs for raw-score ranking of candidate targets for the same ligand."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import torch

from .io import ROOT, SCHEMA, allowed_path, load_checkpoint, load_features
from .train_pdbbind import INDEX_SCHEMA, SCHEMA as AFFINITY_SCHEMA, load_trained_score, verified_tensor_file


DEFAULT_RANK_CHECKPOINT = ROOT / "node_rank/runs/pdbbind2020_refined_20261118/model/best_validation.pt"
IDENTITY_FIELDS = ("pair_id", "compound_id", "target_id")


def load_for_ranking(path, device="cpu"):
    path = allowed_path(path)
    header = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    schema = header.get("schema") if isinstance(header, dict) else None
    del header
    if schema == AFFINITY_SCHEMA:
        model, payload = load_trained_score(path, device)
        if payload.get("score_direction") != "higher_is_stronger_predicted_binding":
            raise ValueError("This checkpoint does not declare the expected descending score direction.")
        return model, payload
    if schema == SCHEMA:
        return load_checkpoint(path, device)
    raise ValueError("Unsupported ranking checkpoint schema.")


def training_objective(checkpoint):
    if checkpoint["training_steps"] == 0:
        return "untrained"
    if checkpoint["schema"] == AFFINITY_SCHEMA:
        return checkpoint["training_task"]
    return checkpoint.get("objective", "within_compound_pairwise_ranking")


def candidates(rows, compound_id=None):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Provide a nonempty list of candidate targets.")
    selected, pair_ids, targets = [], set(), defaultdict(set)
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in IDENTITY_FIELDS):
            raise ValueError("Each row needs pair_id, compound_id and target_id.")
        if compound_id is not None and row["compound_id"] != compound_id:
            continue
        if row["pair_id"] in pair_ids:
            raise ValueError("Duplicate candidate pair_id.")
        if row["target_id"] in targets[row["compound_id"]]:
            raise ValueError("Provide one pose per compound and target sequence; duplicate target in candidate set.")
        if "target_name" in row and (not isinstance(row["target_name"], str) or not row["target_name"].strip()):
            raise ValueError("If supplied, target_name must be a nonempty display name.")
        targets[row["compound_id"]].add(row["target_id"])
        pair_ids.add(row["pair_id"])
        selected.append(row)
    if not selected:
        raise ValueError("No candidates match the requested compound_id (canonical SMILES).")
    if any(len(group) < 2 for group in targets.values()):
        raise ValueError("Each compound must have at least two distinct candidate targets; single-target scores cannot form a target ranking.")
    return selected


def _from_index(row, directory, policy):
    shards = {}
    for kind in ("sample", "sequence"):
        filename, checksum = row.get(kind + "_path"), row.get(kind + "_sha256")
        if not isinstance(filename, str) or not filename or not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"Missing {kind} shard path or SHA256 for {row['pair_id']}.")
        path = allowed_path(directory / filename)
        marker_path = allowed_path(path.parent / "metadata.json")
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
            if isinstance(marker, dict) and marker.get("schema") == "node_rank_pdbbind_cache_v1":
                identity = marker.get("identity", {})
                if (marker.get("payload_file") != path.name or marker.get("payload_sha256") != checksum
                        or identity.get("sequence_sha256") != row["target_id"]
                        or identity.get("policy") != policy.get("esm" if kind == "sequence" else "pbc")
                        or (kind == "sample" and (identity.get("pair_id"), identity.get("smiles")) != (row["pair_id"], row["compound_id"]))):
                    raise ValueError(f"Candidate identity or feature policy differs from the {kind} cache marker.")
        shards[kind] = verified_tensor_file(path, checksum)
    sample = shards["sample"]
    if not isinstance(sample, dict) or any(sample.get(key) != row[key] for key in IDENTITY_FIELDS):
        raise ValueError("Candidate identity differs from the atom-feature shard.")
    if sample.get("smiles") != row["compound_id"]:
        raise ValueError("Compound ID differs from the posed sample's canonical SMILES.")
    sequence, residues = sample.get("protein_sequence"), shards["sequence"]
    if (not isinstance(sequence, str) or not sequence or hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["target_id"]
            or not isinstance(residues, torch.Tensor) or residues.dtype != torch.float32 or residues.shape != (len(sequence), 2560)):
        raise ValueError("Target ID or residue features do not match the declared protein sequence.")
    return {**sample, "sequence_features": residues}


def ranked_inputs(path, checkpoint, compound_id=None):
    path = allowed_path(path)
    sharded = path.suffix.lower() == ".json"
    if sharded:
        artifact = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(artifact, dict) or artifact.get("schema") != INDEX_SCHEMA:
            raise ValueError(f"Expected a {INDEX_SCHEMA} JSON index for the selected candidates.")
    else:
        artifact = load_features(path)
    policy = artifact.get("feature_policy")
    if not isinstance(policy, dict) or not policy:
        raise ValueError("Prediction features need an explicit, nonempty feature policy.")
    if checkpoint["training_steps"] and policy != checkpoint["feature_policy"]:
        raise ValueError("Prediction feature policy differs from training. Use the feature extractor for this checkpoint.")
    selected = candidates(artifact.get("samples"), compound_id)
    for row in selected:
        identity = {key: row[key] for key in IDENTITY_FIELDS}
        if "target_name" in row:
            identity["target_name"] = row["target_name"]
        sample = _from_index(row, path.parent, policy) if sharded else row
        yield identity, sample
