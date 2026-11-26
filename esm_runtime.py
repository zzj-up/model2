"""Shared, memory-conscious ESM feature extraction and cache provenance."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


def resolve_device(request="auto", preferred=None):
    if request == "auto":
        request = str(preferred) if preferred is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu / --esm-device cpu.")
    return device


def resolve_esm_runtime(request="auto", dtype_name=None, preferred=None):
    device = resolve_device(request, preferred)
    dtype_name = dtype_name or ("float16" if device.type == "cuda" else "float32")
    dtype = getattr(torch, dtype_name)
    if device.type == "cpu" and dtype == torch.float16:
        warnings.warn("CPU float16 may be slow or unsupported; --esm-dtype float32 is recommended.")
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16; use --esm-dtype float16.")
    return device, dtype


def esm_metadata(esm_path, max_length, device, dtype):
    source = os.path.expanduser(str(esm_path))
    path = Path(source)
    metadata = {
        "model_source": str(path.resolve()) if path.exists() else source,
        "max_length": int(max_length),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "pooling": "residue_mean_exclude_first_last_special_tokens_v1",
    }
    config_path = path / "config.json"
    if config_path.is_file():
        metadata["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path = path / "download_manifest.json"
    if manifest_path.is_file():
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        metadata["download_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        metadata["model_repository"] = manifest.get("repository")
        metadata["model_revision"] = manifest.get("revision")
    return metadata


def validate_esm_metadata(saved, expected, label="feature cache"):
    if not saved:
        warnings.warn(
            f"Legacy {label} has no ESM provenance. Verify its model, precision and max length before reuse."
        )
        return
    differences = []
    for key in ("model_source", "max_length", "dtype", "pooling"):
        if saved.get(key) != expected.get(key):
            differences.append(f"{key}: cached={saved.get(key)!r}, requested={expected.get(key)!r}")
    for key in ("config_sha256", "model_repository", "model_revision", "download_manifest_sha256"):
        if saved.get(key) and expected.get(key) and saved[key] != expected[key]:
            differences.append(f"{key} differs")
    if differences:
        raise ValueError(
            f"Incompatible {label}: " + "; ".join(differences)
            + ". Select the original settings or rebuild this cache in a separate output directory."
        )


def _cache_identity(metadata):
    # Device is deliberately excluded: CPU/CUDA may reuse features computed with
    # the same model, truncation, precision and pooling settings.
    identity = {key: value for key, value in metadata.items() if key != "device"}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return identity, hashlib.sha256(encoded).hexdigest()


def _load_sequence_embedding(path, namespace, sequence_hash, hidden_size):
    if not path.is_file():
        return None
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict):
            raise ValueError("cache payload is not a dictionary")
        if saved.get("namespace") != namespace or saved.get("sequence_sha256") != sequence_hash:
            raise ValueError("cache identity does not match")
        tensor = saved.get("embedding")
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (hidden_size,):
            raise ValueError("cached embedding dimension does not match")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError("cached embedding is not finite float32")
        return tensor
    except (OSError, RuntimeError, ValueError, EOFError, TypeError, pickle.UnpicklingError) as exc:
        warnings.warn(f"Ignoring invalid ESM sequence cache {path}: {exc}")
        return None


def _save_sequence_embedding(path, namespace, sequence_hash, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(
            {"namespace": namespace, "sequence_sha256": sequence_hash, "embedding": tensor},
            temporary_name,
        )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def extract_esm_features(
    sequences, esm_path, device, batch_size, max_length, *,
    esm_device="auto", esm_dtype=None, expected_hidden_size=None,
):
    if batch_size < 1 or max_length < 3:
        raise ValueError("ESM batch size must be >= 1 and max length must be >= 3.")
    extraction_device, dtype = resolve_esm_runtime(esm_device, esm_dtype, device)
    config = AutoConfig.from_pretrained(esm_path)
    hidden_size = int(config.hidden_size)
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise ValueError(
            f"ESM hidden size {hidden_size} does not match required protein feature dimension "
            f"{expected_hidden_size}. Select the original ESM model; do not substitute a smaller model."
        )
    metadata = esm_metadata(esm_path, max_length, extraction_device, dtype)
    if getattr(config, "_commit_hash", None):
        metadata["resolved_config_revision"] = config._commit_hash
    identity, namespace = _cache_identity(metadata)
    cache_root = Path(os.environ.get("DTI_ESM_CACHE_DIR", str(Path.home() / ".cache" / "dti_esm"))).expanduser()
    cache_dir = cache_root / namespace
    sequences = list(sequences)
    ordered_hashes = []
    unique_sequences = {}
    for sequence in sequences:
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("ESM sequences must be nonempty strings.")
        sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        ordered_hashes.append(sequence_hash)
        unique_sequences.setdefault(sequence_hash, sequence)
    if not sequences:
        return torch.empty((0, hidden_size), dtype=torch.float32)
    embeddings = {}
    missing = []
    for sequence_hash, sequence in tqdm(unique_sequences.items(), desc="ESM cache"):
        path = cache_dir / sequence_hash[:2] / (sequence_hash + ".pt")
        tensor = _load_sequence_embedding(path, namespace, sequence_hash, hidden_size)
        if tensor is None:
            missing.append((sequence_hash, sequence))
        else:
            embeddings[sequence_hash] = tensor
    print(
        f"ESM sequence cache: {len(embeddings):,}/{len(unique_sequences):,} unique hits; "
        f"{len(missing):,} to compute. Cache: {cache_dir}", flush=True,
    )
    if not missing:
        return torch.stack([embeddings[key] for key in ordered_hashes], dim=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Human-readable provenance accompanies the hash-based namespace.
    (cache_dir / "metadata.json").write_text(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8",
    )
    print(f"Loading ESM from {esm_path}; device={extraction_device}, dtype={dtype}, batch={batch_size}")
    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    model = None
    try:
        model = AutoModel.from_pretrained(
            esm_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(extraction_device)
        model.eval()
        with torch.inference_mode():
            for start in tqdm(range(0, len(missing), batch_size), desc="ESM2"):
                batch = missing[start:start + batch_size]
                inputs = tokenizer(
                    [sequence for _, sequence in batch], return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length,
                ).to(extraction_device)
                outputs = model(**inputs)
                # Same residue mean as the original scripts; accumulate in FP32
                # when the model runs in FP16/BF16 to avoid overflow in the sum.
                last_hidden = (outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state).float()
                mask = inputs["attention_mask"].clone().to(last_hidden.dtype)
                mask[:, 0] = 0
                last_pos = (inputs["attention_mask"].sum(dim=1).long() - 1).clamp_min(0)
                rows = torch.arange(mask.size(0), device=mask.device)
                mask[rows, last_pos] = 0
                mask = mask.unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = pooled.cpu()
                for row, (sequence_hash, _) in enumerate(batch):
                    tensor = pooled[row].clone().contiguous()
                    if tensor.shape != (hidden_size,) or not torch.isfinite(tensor).all().item():
                        raise RuntimeError(f"ESM produced an invalid embedding for sequence SHA256 {sequence_hash}.")
                    path = cache_dir / sequence_hash[:2] / (sequence_hash + ".pt")
                    _save_sequence_embedding(path, namespace, sequence_hash, tensor)
                    embeddings[sequence_hash] = tensor
                del inputs, outputs, last_hidden, mask, pooled
        # Restore the exact caller order (including duplicate sequences). Outside
        # inference_mode, stack creates ordinary tensors usable in training.
        return torch.stack([embeddings[key] for key in ordered_hashes], dim=0)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "ESM exhausted GPU memory. Use --esm-batch-size 1 --esm-device cpu --esm-dtype float32, "
            "or reuse matching original protein embeddings. Keep the original ESM model and max length."
        ) from exc
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
