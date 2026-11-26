"""Prepare fresh Morgan and frozen official ESM2 features for explicit structures.

Long proteins are covered by consecutive windows; no residues are truncated.
The persistent ESM window cache makes interrupted extraction resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from esm_runtime import esm_metadata, extract_esm_features, resolve_esm_runtime
from pbcnet_features import canonical_smiles, normalize_sequence


SCHEMA = "pdbbind_entity_features_v1"
MODEL_ID = "facebook/esm2_t36_3B_UR50D"
WINDOW_POLICY = "nonoverlapping_full_sequence_windows_residue_length_weighted_mean_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(records, window_residues):
    if window_residues < 1:
        raise ValueError("At least one residue per ESM window is required.")
    drugs = list(dict.fromkeys(canonical_smiles(record["smiles"]) for record in records))
    proteins = list(dict.fromkeys(normalize_sequence(record["protein_sequence"]) for record in records))
    windows, groups = [], []
    for sequence in proteins:
        group = []
        for start in range(0, len(sequence), window_residues):
            window = sequence[start:start + window_residues]
            group.append((len(windows), len(window)))
            windows.append(window)
        if sum(length for _, length in group) != len(sequence):
            raise RuntimeError("The window plan omitted protein residues.")
        groups.append(group)
    return drugs, proteins, windows, groups


def aggregate_windows(embeddings, groups):
    if embeddings.ndim != 2 or embeddings.shape[1] != 2560:
        raise ValueError("ESM windows must have shape [N, 2560].")
    proteins = []
    for group in groups:
        positions = torch.tensor([index for index, _ in group], dtype=torch.long)
        lengths = torch.tensor([length for _, length in group], dtype=torch.float32)
        pooled = (embeddings[positions].float() * lengths[:, None]).sum(0) / lengths.sum()
        proteins.append(pooled)
    output = torch.stack(proteins).contiguous()
    if not torch.isfinite(output).all():
        raise ValueError("Nonfinite pooled protein features.")
    return output


def morgan_features(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    rows = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError("A canonical input compound cannot be parsed.")
        rows.append(torch.from_numpy(generator.GetFingerprintAsNumPy(molecule)).to(torch.float32))
    return torch.stack(rows).contiguous()


def official_model_identity(model_path):
    manifest_path = model_path / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repository") != MODEL_ID:
        raise ValueError("Use the retained official ESM2 3B model, not a task checkpoint.")
    checked = []
    for record in manifest["files"]:
        name = record["rfilename"]
        path = (model_path / name).resolve()
        if not path.is_relative_to(model_path.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe official model file: {name}")
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Official model file size differs: {name}")
        expected = record.get("lfs", {}).get("sha256")
        if expected:
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Official ESM weight checksum differs: {name}")
            checked.append({"file": name, "sha256": actual, "size_bytes": path.stat().st_size})
    if not checked:
        raise ValueError("The official model manifest has no weight checksums.")
    return {"repository": manifest["repository"], "revision": manifest["revision"],
            "download_manifest_sha256": sha256(manifest_path), "verified_weights": checked}


def publish_new(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", mode="w",
                                     encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validate_existing(path, manifest_hash, drugs, proteins, max_length, dtype):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != SCHEMA or artifact.get("manifest_sha256") != manifest_hash:
        raise ValueError("The existing entity feature artifact belongs to another dataset/schema.")
    if artifact.get("drug_smiles") != drugs or artifact.get("protein_sequences") != proteins:
        raise ValueError("Existing feature identities or row order differ.")
    policy = artifact.get("provenance", {}).get("sequence_windowing", {})
    if policy.get("policy") != WINDOW_POLICY or policy.get("token_max_length") != max_length:
        raise ValueError("Existing ESM feature windowing settings differ.")
    if artifact["provenance"]["esm_runtime"]["dtype"] != str(dtype).replace("torch.", ""):
        raise ValueError("Existing ESM precision differs.")
    for field, shape in (("drug_features", (len(drugs), 2048)),
                         ("protein_features", (len(proteins), 2560))):
        tensor = artifact.get(field)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != torch.float32:
            raise ValueError(f"Existing {field} is malformed.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Existing {field} contains nonfinite values.")
    return artifact


def verify_helpers():
    """Verify window coverage, pooling and safe cache loading without ESM weights."""
    records = [{"smiles": "C(C)O", "protein_sequence": "ACDEFGHI"},
               {"smiles": "CCO", "protein_sequence": "ACD"}]
    drugs, proteins, windows, groups = build_plan(records, 3)
    assert drugs == ["CCO"] and proteins == ["ACDEFGHI", "ACD"]
    assert windows == ["ACD", "EFG", "HI", "ACD"]
    for sequence, group in zip(proteins, groups):
        assert "".join(windows[index] for index, _ in group) == sequence
    embeddings = torch.tensor([1., 3., 7., 1.])[:, None].expand(4, 2560).clone()
    pooled = aggregate_windows(embeddings, groups)
    assert torch.equal(pooled[0], torch.full((2560,), 3.25))
    assert torch.equal(pooled[1], torch.ones(2560))
    fingerprints = morgan_features(drugs)
    assert fingerprints.shape == (1, 2048) and fingerprints.dtype == torch.float32
    assert set(fingerprints.unique().tolist()).issubset({0., 1.})
    artifact = {"schema": SCHEMA, "manifest_sha256": "synthetic-not-experimental",
                "drug_smiles": drugs, "protein_sequences": proteins,
                "drug_features": fingerprints, "protein_features": pooled,
                "provenance": {"sequence_windowing": {"policy": WINDOW_POLICY,
                               "token_max_length": 5}, "esm_runtime": {"dtype": "float16"}}}
    with tempfile.TemporaryDirectory(prefix="entity_feature_synthetic_") as directory:
        path = Path(directory) / "features.pt"
        publish_new(path, lambda destination: torch.save(artifact, destination))
        loaded = validate_existing(path, "synthetic-not-experimental", drugs, proteins, 5, torch.float16)
        assert torch.equal(loaded["protein_features"], pooled)
        try:
            validate_existing(path, "wrong-manifest", drugs, proteins, 5, torch.float16)
        except ValueError:
            pass
        else:
            raise AssertionError("A mismatched manifest must be rejected.")
    print(json.dumps({"status": "passed", "data": "temporary_synthetic_helpers_only",
                      "checks": ["stable_canonical_identity_order", "all_residues_covered",
                                 "length_weighted_window_mean", "binary_Morgan_2048",
                                 "weights_only_feature_reload", "reject_other_manifest"],
                      "official_ESM_weights_loaded": False, "training_performed": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--esm-path", type=Path, default=Path.home() / "models/esm2_3b")
    parser.add_argument("--esm-device", default="cuda")
    parser.add_argument("--esm-dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--esm-batch-size", "--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int,
                        help="Total ESM tokens including two special tokens; all residues use consecutive windows.")
    parser.add_argument("--window-residues", type=int,
                        help="Residues per full-coverage ESM window; defaults to 510.")
    parser.add_argument("--sequence-cache", "--cache-dir", type=Path,
                        default=ROOT / "inputs/pdbbind_1000/training_features/esm_sequences")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_helpers()
        return
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required unless --verify-only is selected.")
    if args.max_length is not None and args.window_residues is not None:
        parser.error("Choose --max-length or --window-residues, not both.")
    args.max_length = args.window_residues + 2 if args.window_residues is not None else (args.max_length or 512)
    if args.threads < 1 or args.esm_batch_size < 1 or not 3 <= args.max_length <= 1024:
        parser.error("Threads/batch size must be positive and max length must be 3–1024.")
    if args.manifest.resolve().is_relative_to((ROOT / "quarantine").resolve()):
        parser.error("Isolated historical data cannot be used by this fresh feature workflow.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DTI_ESM_CACHE_DIR"] = str(args.sequence_cache.resolve())
    torch.set_num_threads(args.threads)
    records = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty explicit structure manifest is required.")
    manifest_hash = sha256(args.manifest)
    drugs, proteins, windows, groups = build_plan(records, args.max_length - 2)
    device, dtype = resolve_esm_runtime(args.esm_device, args.esm_dtype)
    plan = {"manifest_sha256": manifest_hash, "pairs": len(records),
            "unique_compounds": len(drugs), "unique_protein_sequences": len(proteins),
            "sequence_length_min": min(map(len, proteins)), "sequence_length_max": max(map(len, proteins)),
            "sequence_windowing": {"policy": WINDOW_POLICY, "token_max_length": args.max_length,
                "window_residues": args.max_length - 2, "total_windows": len(windows),
                "unique_window_sequences": len(set(windows)),
                "multiwindow_proteins": sum(len(group) > 1 for group in groups),
                "all_residues_covered": True, "final_feature_l2_normalized": False},
            "device": str(device), "dtype": str(dtype), "batch_size": args.esm_batch_size,
            "sequence_cache": str(args.sequence_cache.resolve()), "model_training_performed": False}
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if args.output.exists():
        validate_existing(args.output, manifest_hash, drugs, proteins, args.max_length, dtype)
        print(json.dumps({"status": "existing_valid_cache", "output": str(args.output.resolve()),
                          "sha256": sha256(args.output)}), flush=True)
        return
    status_path = args.output.with_suffix(".status.json")
    write_status(status_path, {**plan, "stage": "verifying_official_ESM_weights"})
    identity = official_model_identity(args.esm_path.resolve())
    config = AutoConfig.from_pretrained(args.esm_path, local_files_only=True)
    if config.model_type != "esm" or config.hidden_size != 2560:
        raise ValueError("The retained official ESM model does not provide 2560-dimensional features.")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_path, local_files_only=True)
    tokenized = tokenizer(windows, add_special_tokens=True, truncation=False)
    if any(len(ids) != len(sequence) + 2 or len(ids) > args.max_length
           for ids, sequence in zip(tokenized["input_ids"], windows)):
        raise ValueError("ESM tokenization did not preserve every window residue.")
    del tokenized, tokenizer
    drug_features = morgan_features(drugs)
    write_status(status_path, {**plan, "stage": "extracting_frozen_ESM_windows"})
    try:
        window_features = extract_esm_features(
            windows, args.esm_path, device, args.esm_batch_size, args.max_length,
            esm_device=str(device), esm_dtype=args.esm_dtype, expected_hidden_size=2560)
        protein_features = aggregate_windows(window_features, groups)
        if sha256(args.manifest) != manifest_hash:
            raise RuntimeError("The structure manifest changed during entity feature extraction.")
        artifact = {"schema": SCHEMA, "manifest_sha256": manifest_hash,
                    "drug_smiles": drugs, "protein_sequences": proteins,
                    "drug_features": drug_features, "protein_features": protein_features,
                    "provenance": {
                        "manifest_path": str(args.manifest.resolve()),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "script_sha256": sha256(__file__),
                        "morgan": {"radius": 2, "bits": 2048, "include_chirality": True,
                                   "representation": "binary_float32"},
                        "esm_official_identity": identity,
                        "esm_model_id": MODEL_ID,
                        "esm_config_sha256": sha256(args.esm_path / "config.json"),
                        "esm_runtime": esm_metadata(args.esm_path, args.max_length, device, dtype),
                        "sequence_windowing": plan["sequence_windowing"],
                        "sequence_cache": str(args.sequence_cache.resolve()),
                        "package_versions": {name: version(name) for name in ("torch", "transformers", "rdkit")},
                        "encoder_mode": "official_frozen_feature_extraction",
                        "task_training_performed": False,
                    }}
        publish_new(args.output, lambda path: torch.save(artifact, path))
        validate_existing(args.output, manifest_hash, drugs, proteins, args.max_length, dtype)
        report = {**plan, "stage": "complete", "output": str(args.output.resolve()),
                  "output_sha256": sha256(args.output), "drug_feature_shape": list(drug_features.shape),
                  "protein_feature_shape": list(protein_features.shape)}
        write_status(status_path, report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        write_status(status_path, {**plan, "stage": "failed", "error_type": type(error).__name__,
                                  "error": str(error), "window_cache_preserved_for_resume": True})
        raise


if __name__ == "__main__":
    main()
