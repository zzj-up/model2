"""Resumable sharded frozen features for a fixed PDBbind usable set.

This is a separate pipeline: prepare_features.py and its existing checkpoint
policy remain unchanged. PBC quality screening precedes the split lock. ESM
features are computed only for explicitly supplied train/validation manifests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

from .prepare_features import (ROOT, WINDOW, digest_json, load_records, publish,
                               reject_quarantine_path, sha256)

PBC_SCHEMA = "node_rank_pdbbind_pbc_index_v1"
INDEX_SCHEMA = "node_rank_pdbbind_feature_index_v1"
CACHE_SCHEMA = "node_rank_pdbbind_cache_v1"
VERSION = "pdbbind_sharded_frozen_nodes_v1"
_ENCODER = _GRAPH = _PROVENANCE = None


def write_json(path, value, replace=False):
    path = reject_quarantine_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", mode="w", encoding="utf-8", delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path):
    """Retain declared source digests; never silently refresh changed inputs."""
    path = reject_quarantine_path(path)
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    records = load_records(path)
    for original, record in zip(raw, records):
        for key in ("protein_pdb_sha256", "ligand_sdf_sha256"):
            if original.get(key) and original[key] != record[key]:
                raise RuntimeError(f"Declared structural source changed for {record['pair_id']}: {key}")
        if not isinstance(record.get("compound_group"), str) or not record["compound_group"]:
            raise ValueError("Every PDBbind record needs an explicit standardized compound_group.")
    return records


def cache_load(directory, identity, *, payload=True):
    """A metadata marker commits an independently recoverable cache entry."""
    import torch
    directory = reject_quarantine_path(directory)
    marker = directory / "metadata.json"
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("schema") != CACHE_SCHEMA or metadata.get("identity") != identity:
        raise ValueError(f"Cached preprocessing/encoder/source identity differs: {marker}")
    filename = metadata.get("payload_file", "")
    if not filename.startswith("payload_") or Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError(f"Unsafe cache payload filename: {marker}")
    path = directory / filename
    if sha256(path) != metadata["payload_sha256"]:
        raise ValueError(f"Cache checksum failed: {path}")
    result = {**metadata, "payload_path": str(path)}
    if payload:
        result["payload"] = torch.load(path, map_location="cpu", weights_only=True)
    return result


def cache_save(directory, value, identity, stats=None):
    directory = reject_quarantine_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A unique payload means an interruption before marker publication never
    # requires overwriting or deleting a completed user's file on resume.
    filename = "payload_" + uuid.uuid4().hex + ".pt"
    path = directory / filename
    publish(path, value)
    metadata = {"schema": CACHE_SCHEMA, "identity": identity, "payload_file": filename,
                "payload_sha256": sha256(path), "stats": stats or {}}
    try:
        write_json(directory / "metadata.json", metadata)
    except FileExistsError:
        # A concurrent identical cache producer may have committed first.
        return cache_load(directory, identity, payload=False)
    return {**metadata, "payload_path": str(path)}


def pbc_policy(provenance, max_atoms):
    return {"version": VERSION, "node_extractor_sha256": sha256(__file__),
            "base_node_extractor_sha256": sha256(ROOT / "node_rank/prepare_features.py"),
            "base_complex_extractor_sha256": sha256(ROOT / "prepare_pbcnet_features.py"),
            "graph_validator_sha256": sha256(ROOT / "node_rank/graph.py"),
            "official_commit": provenance["official_commit"], "checkpoint_sha256": provenance["checkpoint_sha256"],
            "official_source_sha256": provenance["source_sha256"], "package_versions": provenance["package_versions"],
            "output": "all_atom_tensornet_output_before_readout", "mapping": "unique_full_chain_alignment_exact_pdb_atom_order_v1",
            "pocket_radius_angstrom": 8.0, "max_atoms": max_atoms, "dtype": "float32", "device": "cpu"}


def pbc_identity(record, policy):
    # Labels, split membership, manifest path and full-manifest content are absent.
    # pair_id is retained because the raw sample shard carries this identity.
    return {"policy": policy, "pair_id": record["pair_id"], "smiles": record["smiles"],
            "sequence_sha256": record["target_id"], "protein_sha256": record["protein_pdb_sha256"],
            "ligand_sha256": record["ligand_sdf_sha256"], "pose_index": record.get("pose_index", 0),
            "protein_chain": record.get("protein_chain"),
            "keep_hetero_residues": sorted(set(record.get("keep_hetero_residues", [])))}


def validate_pbc_sample(sample):
    import torch
    from .graph import build_graph
    sequence = sample["protein_sequence"]
    for name, width in (("ligand_features", 96), ("pocket_features", 96)):
        tensor = sample[name]
        if tensor.ndim != 2 or tensor.shape[1] != width or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid PBC atom features: {name}")
    # This transient placeholder validates only graph topology before ESM exists.
    # It is never saved, used for learning, or substituted for sequence features.
    graph = build_graph({**sample, "sequence_features": torch.zeros((len(sequence), 1))})
    return {"ligand_atoms": len(sample["ligand_features"]), "pocket_atoms": len(sample["pocket_features"]),
            "residues": len(sequence), "graph_edges": graph.edge_index.shape[1],
            "contact_pairs": graph.contact_index.shape[1]}


def pbc_worker_init(repository, threads, expected_policy):
    global _ENCODER, _GRAPH, _PROVENANCE
    os.environ["DGLBACKEND"] = "pytorch"
    os.environ["DGL_FFI"] = "ctypes"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch
    from prepare_pbcnet_features import load_official_encoder
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _ENCODER, _GRAPH, _PROVENANCE = load_official_encoder(Path(repository))
    if pbc_policy(_PROVENANCE, expected_policy["max_atoms"]) != expected_policy:
        raise RuntimeError("PBC source changed after scheduling workers.")


def compute_pbc_sample(record, max_atoms):
    import numpy as np
    import torch
    from rdkit import Chem
    import dgl.function as fn
    from prepare_pbcnet_features import build_complex
    from .prepare_features import directed_bonds, pocket_mapping
    with tempfile.TemporaryDirectory(prefix="pdbbind_node_pbc_") as directory, torch.inference_mode():
        directory = Path(directory)
        graph, source = build_complex(record, ROOT, directory, _GRAPH, max_atoms)
        if source["protein_sha256"] != record["protein_pdb_sha256"] or source["ligand_sha256"] != record["ligand_sdf_sha256"]:
            raise RuntimeError("Structural sources changed after scheduling.")
        ligand = Chem.RemoveAllHs(Chem.MolFromMolFile(str(directory / "pose.sdf")))
        pocket = Chem.RemoveAllHs(Chem.MolFromPDBFile(str(directory / "pocket.pdb")))
        nl, npocket = ligand.GetNumAtoms(), pocket.GetNumAtoms()
        positions = torch.from_numpy(np.concatenate([ligand.GetConformer().GetPositions(), pocket.GetConformer().GetPositions()])).float()
        types = graph.nodes["atom"].data["type"]
        elements = torch.tensor([atom.GetAtomicNum() for mol in (ligand, pocket) for atom in mol.GetAtoms()])
        if (len(types) != nl + npocket or not (types[:nl] == 1).all() or not (types[nl:] == 0).all()
                or not torch.equal(graph.nodes["atom"].data["x"], elements)
                or not torch.allclose(graph.nodes["atom"].data["pos"], positions, atol=1e-5, rtol=0)):
            raise ValueError("Exact RDKit and official graph atom order differs.")
        mapping, atom_map = pocket_mapping(record["protein_pdb"], pocket, record["protein_sequence"],
                                           {row["chain"] for row in source["chain_alignment"]}, source["kept_hetero_residues"])
        src, dst = graph.edges(etype="int")
        graph.edges["int"].data["dist"] = (positions[src] - positions[dst]).square().sum(-1).sqrt()
        graph.apply_edges(fn.v_sub_u("pos", "pos", "vec"), etype="int")
        features, _ = _ENCODER.encoder(graph, False)
        features = features.detach().cpu().float().clone().contiguous()
        if features.shape != (nl + npocket, 96) or not torch.isfinite(features).all():
            raise ValueError("PBC returned invalid atom embeddings.")
        lb, lbo = directed_bonds(ligand)
        pb, pbo = directed_bonds(pocket)
        sample = {"pair_id": record["pair_id"], "compound_id": record["compound_id"], "target_id": record["target_id"],
                  "smiles": record["smiles"], "protein_sequence": record["protein_sequence"],
                  "ligand_features": features[:nl].clone(), "pocket_features": features[nl:].clone(),
                  "ligand_pos": positions[:nl].clone(), "pocket_pos": positions[nl:].clone(),
                  "ligand_bonds": lb, "pocket_bonds": pb, "ligand_bond_orders": lbo, "pocket_bond_orders": pbo,
                  "pocket_to_sequence": mapping, "source": {**source, "pocket_atom_map": atom_map}}
    return sample, validate_pbc_sample(sample)


def pbc_job(job):
    record, policy, cache_root, retry_failures = job
    identity = pbc_identity(record, policy)
    key = digest_json(identity)
    cache_root = Path(cache_root)
    directory = cache_root / "pbc" / key
    failure_path = cache_root / "pbc_failures" / (key + ".json")
    started = time.monotonic()
    base = {"pair_id": record["pair_id"], "compound_id": record["compound_id"], "target_id": record["target_id"],
            "compound_group": record["compound_group"], "cache_key": key,
            "protein_sha256": identity["protein_sha256"], "ligand_sha256": identity["ligand_sha256"]}
    cached = cache_load(directory, identity)
    if cached is not None:
        stats = validate_pbc_sample(cached["payload"])
        return {**base, "status": "success", "sample_path": cached["payload_path"], "sample_sha256": cached["payload_sha256"],
                "stats": stats, "cache_hit": True, "seconds": time.monotonic() - started}
    if failure_path.exists() and not retry_failures:
        failure = json.loads(failure_path.read_text())
        if failure["identity"] != identity:
            raise RuntimeError("Failure cache source identity differs.")
        return {**base, **failure["failure"], "cache_hit": True, "seconds": time.monotonic() - started}
    try:
        sample, stats = compute_pbc_sample(record, policy["max_atoms"])
        saved = cache_save(directory, sample, identity, stats)
        return {**base, "status": "success", "sample_path": saved["payload_path"], "sample_sha256": saved["payload_sha256"],
                "stats": stats, "cache_hit": False, "seconds": time.monotonic() - started}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, AssertionError) as error:
        # Runtime/OOM/I/O/source-integrity failures deliberately propagate: they
        # are not scientific reasons to exclude a complex from the usable set.
        failure = {"status": "data_quality_failed", "error_type": type(error).__name__, "error": str(error)}
        if not failure_path.exists():
            write_json(failure_path, {"identity": identity, "failure": failure})
        return {**base, **failure, "cache_hit": False, "seconds": time.monotonic() - started}


def pbc_stage(args):
    import torch
    from prepare_pbcnet_features import load_official_encoder
    started = time.monotonic()
    records = read_manifest(args.manifest)
    full_count = len(records)
    selected = records[:args.limit] if args.limit else records
    encoder, _, provenance = load_official_encoder(args.repository)
    policy = pbc_policy(provenance, args.max_atoms)
    del encoder
    audit = reject_quarantine_path(args.audit_dir)
    cache = reject_quarantine_path(args.cache_dir)
    audit.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    request = {"manifest_sha256": sha256(args.manifest), "source_records_count": full_count,
               "attempted_records_count": len(selected), "pbc_policy": policy}
    config_path = audit / "request.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != request:
            raise ValueError("Audit directory belongs to another request; choose a new audit-dir (cache remains reusable).")
    else:
        write_json(config_path, request)
    output = audit / "pbc_index.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("request") != request:
            raise ValueError("Published PBC audit differs from the current request.")
        for row in saved["samples"]:
            if sha256(audit / row["sample_path"]) != row["sample_sha256"]:
                raise ValueError("A committed PBC sample shard changed.")
        print(json.dumps({"stage": "pbc", "status": "verified_existing", "index": str(output)}), flush=True)
        return
    jobs = [(record, policy, str(cache), args.retry_failures) for record in selected]
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=pbc_worker_init,
                             initargs=(str(args.repository.resolve()), args.threads, policy)) as pool:
        futures = {pool.submit(pbc_job, job): job[0]["pair_id"] for job in jobs}
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                results.append(row)
                with (audit / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if completed % 10 == 0 or completed == len(jobs):
                    status = {"stage": "pbc", "completed": completed, "total": len(jobs),
                              "successes": sum(r["status"] == "success" for r in results),
                              "data_failures": sum(r["status"] != "success" for r in results),
                              "elapsed_seconds": time.monotonic() - started}
                    write_json(audit / "status.json", status, replace=True)
                    print(json.dumps(status), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            write_json(audit / "status.json", {"status": "interrupted_or_runtime_error", "completed": len(results),
                                                "cache_reusable": True}, replace=True)
            raise
    by_id = {row["pair_id"]: row for row in results}
    successes, failures = [], []
    for record in selected:
        row = by_id[record["pair_id"]]
        if row["status"] == "success":
            row["sample_path"] = os.path.relpath(row["sample_path"], audit)
            successes.append(row)
        else:
            failures.append(row)
    if sha256(args.manifest) != request["manifest_sha256"]:
        raise RuntimeError("Candidate manifest changed during PBC screening.")
    result = {"schema": PBC_SCHEMA, "status": "complete" if len(selected) == full_count else "partial_benchmark",
              **request, "request": request, "manifest_path": str(args.manifest.resolve()),
              "samples": successes, "failures": failures, "pbc_provenance": provenance,
              "elapsed_seconds": time.monotonic() - started, "supervision_used": False,
              "test_evaluation_performed": False}
    write_json(output, result)
    accepted = {row["pair_id"] for row in successes}
    write_json(audit / "usable_manifest.json", [row for row in selected if row["pair_id"] in accepted])
    print(json.dumps({"stage": "pbc_done", "index": str(output), "status": result["status"],
                      "successes": len(successes), "failures": len(failures), "seconds": result["elapsed_seconds"]}), flush=True)


def esm_policy(model, device):
    from importlib.metadata import version
    from fresh_dti.prepare_entity_features import official_model_identity
    identity = official_model_identity(model)
    return {"version": VERSION, "node_extractor_sha256": sha256(__file__),
            "official_identity": identity, "config_sha256": sha256(model / "config.json"),
            "tokenizer_sha256": {p.name: sha256(p) for p in sorted(model.glob("*token*.json"))},
            "window_residues": WINDOW, "window_policy": "nonoverlapping_full_coverage_concatenate_v1",
            "dtype": "float16" if device == "cuda" else "float32", "output": "last_hidden_state_per_residue_exclude_special_tokens",
            "package_versions": {name: version(name) for name in ("torch", "transformers", "tokenizers")}}


def sequence_identity(sequence, policy):
    import hashlib
    return {"policy": policy, "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(), "residues": len(sequence)}


def validate_sequence(value, sequence):
    import torch
    if not isinstance(value, torch.Tensor) or value.shape != (len(sequence), 2560) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError("Invalid ESM residue shard shape/dtype/finite values.")


def esm_stage(args):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .prepare_features import residue_windows
    if not args.train_manifest and not args.validation_manifest:
        raise ValueError("Provide train and/or validation manifests; holdout ESM extraction is not scheduled.")
    records, manifests = [], {}
    for role, path in (("train", args.train_manifest), ("validation", args.validation_manifest)):
        if path:
            values = read_manifest(path)
            records.extend(values)
            manifests[role] = {"path": str(path.resolve()), "sha256": sha256(path), "count": len(values)}
    torch.set_num_threads(args.threads)
    policy = esm_policy(args.esm_model, args.device)
    cache = reject_quarantine_path(args.cache_dir)
    namespace = digest_json(policy)
    sequences = {record["target_id"]: record["protein_sequence"] for record in records}
    missing, completed = [], []
    for key, sequence in sequences.items():
        identity = sequence_identity(sequence, policy)
        if key != identity["sequence_sha256"]:
            raise ValueError("Target ID differs from sequence digest.")
        directory = cache / "esm_sequences" / namespace / key
        cached = cache_load(directory, identity)
        if cached is None:
            missing.append((key, sequence, identity, directory))
        else:
            validate_sequence(cached["payload"], sequence)
            completed.append({"target_id": key, "sequence_path": cached["payload_path"], "sequence_sha256": cached["payload_sha256"], "cache_hit": True})
    print(json.dumps({"stage": "esm", "unique_sequences": len(sequences), "cache_hits": len(completed), "missing": len(missing)}), flush=True)
    model = tokenizer = None
    if missing:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")
        config = AutoConfig.from_pretrained(str(args.esm_model), local_files_only=True)
        if config.hidden_size != 2560:
            raise ValueError("Expected retained official ESM2 3B model width 2560.")
        tokenizer = AutoTokenizer.from_pretrained(str(args.esm_model), local_files_only=True)
        # This official checkpoint has no trained pooler. We only need the
        # pretrained per-residue hidden states, so do not create an unused one.
        model = AutoModel.from_pretrained(str(args.esm_model), local_files_only=True, add_pooling_layer=False,
                                          torch_dtype=getattr(torch, policy["dtype"]), low_cpu_mem_usage=True)
        model = model.to(args.device).eval().requires_grad_(False)
        with torch.inference_mode():
            for number, (key, sequence, identity, directory) in enumerate(missing, 1):
                chunks = []
                for start, window in residue_windows(sequence):
                    window_id = sequence_identity(window, policy)
                    window_dir = cache / "esm_windows" / namespace / window_id["sequence_sha256"]
                    cached = cache_load(window_dir, window_id)
                    if cached is None:
                        inputs = tokenizer(window, return_tensors="pt", truncation=False, add_special_tokens=True).to(args.device)
                        if inputs["input_ids"].shape != (1, len(window) + 2):
                            raise ValueError("Tokenizer must retain exactly one token per residue plus BOS/EOS.")
                        output = model(**inputs)
                        tensor = output.last_hidden_state[0, 1:len(window) + 1].detach().cpu().float().clone().contiguous()
                        validate_sequence(tensor, window)
                        cache_save(window_dir, tensor, window_id, {"residues": len(window)})
                        del inputs, output
                    else:
                        tensor = cached["payload"]
                        validate_sequence(tensor, window)
                    chunks.append(tensor)
                # A one-window sequence reuses its committed window payload
                # rather than creating a second copy of a large tensor.
                full = torch.cat(chunks, dim=0)
                validate_sequence(full, sequence)
                if len(chunks) == 1:
                    window_cached = cache_load(cache / "esm_windows" / namespace / key, identity, payload=False)
                    directory.mkdir(parents=True, exist_ok=True)
                    filename = "payload_" + uuid.uuid4().hex + ".pt"
                    os.link(window_cached["payload_path"], directory / filename)
                    marker = {"schema": CACHE_SCHEMA, "identity": identity, "payload_file": filename,
                              "payload_sha256": window_cached["payload_sha256"], "stats": {"residues": len(sequence)}}
                    write_json(directory / "metadata.json", marker)
                    saved = {**marker, "payload_path": str(directory / filename)}
                else:
                    saved = cache_save(directory, full, identity, {"residues": len(sequence)})
                completed.append({"target_id": key, "sequence_path": saved["payload_path"], "sequence_sha256": saved["payload_sha256"], "cache_hit": False})
                if number % 10 == 0 or number == len(missing):
                    print(json.dumps({"stage": "esm", "new_sequences_completed": number, "total_missing": len(missing), "last_residues": len(sequence)}), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for item in manifests.values():
        if sha256(item["path"]) != item["sha256"]:
            raise RuntimeError("Split manifest changed during ESM extraction.")
    namespace_path = cache / "esm_policies" / (namespace + ".json")
    if not namespace_path.exists():
        write_json(namespace_path, policy)
    result = {"schema": "node_rank_pdbbind_esm_audit_v1", "manifests": manifests, "esm_policy": policy,
              "sequence_cache_namespace": namespace, "sequences": completed, "official_encoder_updated": False,
              "holdout_manifest_used": False, "test_evaluation_performed": False}
    audit_path = args.audit_output or cache / "esm_audits" / (digest_json(manifests) + ".json")
    if audit_path.exists():
        existing = json.loads(audit_path.read_text())
        if existing["manifests"] != manifests or existing["esm_policy"] != policy:
            raise ValueError("Existing ESM audit belongs to other sources/policy.")
    else:
        write_json(audit_path, result)
    print(json.dumps({"stage": "esm_done", "unique_sequences": len(completed), "audit": str(audit_path)}), flush=True)


def index_stage(args):
    records = read_manifest(args.manifest)
    pbc = json.loads(args.pbc_index.read_text(encoding="utf-8-sig"))
    if pbc.get("schema") != PBC_SCHEMA or pbc.get("status") != "complete":
        raise ValueError("Use a completed full-candidate PBC audit, never a benchmark index.")
    esm = json.loads(args.esm_audit.read_text(encoding="utf-8-sig"))
    if esm.get("schema") != "node_rank_pdbbind_esm_audit_v1":
        raise ValueError("Unsupported ESM audit.")
    if esm["manifests"].get(args.split, {}).get("sha256") != sha256(args.manifest):
        raise ValueError("ESM audit does not contain the exact requested split manifest.")
    pbc_lookup = {row["pair_id"]: row for row in pbc["samples"]}
    esm_lookup = {row["target_id"]: row for row in esm["sequences"]}
    output = reject_quarantine_path(args.output)
    rows = []
    checked_sequences = set()
    for record in records:
        pbc_row = pbc_lookup[record["pair_id"]]
        if (pbc_row["compound_id"], pbc_row["target_id"], pbc_row["compound_group"]) != (record["compound_id"], record["target_id"], record["compound_group"]):
            raise ValueError("Compound/target identity differs from the screened usable set.")
        sample_path = reject_quarantine_path(args.pbc_index.parent / pbc_row["sample_path"])
        if sha256(sample_path) != pbc_row["sample_sha256"]:
            raise ValueError("Screened sample shard changed.")
        sequence_row = esm_lookup[record["target_id"]]
        sequence_path = reject_quarantine_path(sequence_row["sequence_path"])
        if sequence_path not in checked_sequences:
            if sha256(sequence_path) != sequence_row["sequence_sha256"]:
                raise ValueError("ESM sequence shard changed.")
            checked_sequences.add(sequence_path)
        rows.append({"pair_id": record["pair_id"], "compound_id": record["compound_id"], "target_id": record["target_id"],
                     "compound_group": record["compound_group"], "sample_path": os.path.relpath(sample_path, output.parent),
                     "sample_sha256": pbc_row["sample_sha256"], "sequence_path": os.path.relpath(sequence_path, output.parent),
                     "sequence_sha256": sequence_row["sequence_sha256"], "p_affinity": record["p_affinity"],
                     "affinity_endpoint": record["affinity_endpoint"], "pbc_cache_key": pbc_row["cache_key"]})
    feature_policy = {"version": VERSION, "pbc": pbc["pbc_policy"], "esm": esm["esm_policy"],
                      "shards": "separate_pbc_atom_sample_and_unique_esm_residue_tensor_v1"}
    result = {"schema": INDEX_SCHEMA, "split": args.split, "feature_policy": feature_policy, "samples": rows,
              "manifest_sha256": sha256(args.manifest), "split_manifest_sha256": sha256(args.manifest),
              "source_records_count": len(records), "pbc_audit_sha256": sha256(args.pbc_index),
              "esm_audit_sha256": sha256(args.esm_audit), "feature_index_source_sha256": sha256(__file__),
              "manifest_path": str(args.manifest.resolve()), "test_evaluation_performed": False}
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8-sig")) != result:
            raise ValueError("Existing feature index differs from the verified sources and policy.")
        status = "verified_existing"
    else:
        write_json(output, result)
        status = "published"
    print(json.dumps({"stage": "index_done", "status": status, "split": args.split,
                      "samples": len(rows), "output": str(output), "sha256": sha256(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    pbc = sub.add_parser("pbc")
    pbc.add_argument("--manifest", type=Path, required=True)
    pbc.add_argument("--cache-dir", type=Path, required=True)
    pbc.add_argument("--audit-dir", type=Path, required=True)
    pbc.add_argument("--repository", type=Path, default=ROOT / "third_party/PBCNet2.0")
    pbc.add_argument("--workers", type=int, default=2)
    pbc.add_argument("--threads", type=int, default=1)
    pbc.add_argument("--max-atoms", type=int, default=5000)
    pbc.add_argument("--limit", type=int)
    pbc.add_argument("--retry-failures", action="store_true")
    esm = sub.add_parser("esm")
    esm.add_argument("--train-manifest", type=Path)
    esm.add_argument("--validation-manifest", type=Path)
    esm.add_argument("--cache-dir", type=Path, required=True)
    esm.add_argument("--esm-model", type=Path, default=Path("/home/dti/models/esm2_3b"))
    esm.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    esm.add_argument("--threads", type=int, default=4)
    esm.add_argument("--audit-output", type=Path)
    index = sub.add_parser("index")
    index.add_argument("--manifest", type=Path, required=True)
    index.add_argument("--split", choices=("train", "validation"), required=True)
    index.add_argument("--pbc-index", type=Path, required=True)
    index.add_argument("--esm-audit", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("workers", "threads", "max_atoms", "limit"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    {"pbc": pbc_stage, "esm": esm_stage, "index": index_stage}[args.stage](args)


if __name__ == "__main__":
    main()
