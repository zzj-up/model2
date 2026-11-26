"""Validated, pair-specific embeddings from the frozen PBCNet2.0 encoder."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem


SCHEMA = "pbcnet2_complex_embeddings_v1"
FEATURE_DIM = 96
GRAPH_PREPROCESSING_VERSION = "official_graph2pickle_pocket8_v1"


def encoder_identity(provenance):
    fields = ("official_commit", "checkpoint_sha256", "graph_preprocessing_version")
    identity = {field: provenance.get(field) for field in fields}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("PBCNet2.0 provenance lacks a pinned encoder/preprocessing identity.")
    for field, length in (("official_commit", 40), ("checkpoint_sha256", 64)):
        if len(identity[field]) != length or any(c not in "0123456789abcdef" for c in identity[field]):
            raise ValueError(f"Malformed PBCNet2.0 encoder identity field: {field}")
    if identity["graph_preprocessing_version"] != GRAPH_PREPROCESSING_VERSION:
        raise ValueError("The PBCNet2.0 graph preprocessing version is unsupported.")
    return identity


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sequence(sequence):
    if not isinstance(sequence, str):
        raise ValueError("A protein sequence must be a string of amino-acid letters.")
    sequence = "".join(sequence.split()).upper()
    if not sequence or not sequence.isalpha() or not sequence.isascii():
        raise ValueError("A protein sequence must contain amino-acid letters.")
    return sequence


def canonical_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("A compound SMILES must be a nonempty string.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid compound SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def structure_key(smiles, sequence):
    sequence_hash = hashlib.sha256(normalize_sequence(sequence).encode("ascii")).hexdigest()
    return canonical_smiles(smiles) + "\t" + sequence_hash


@dataclass
class PBCNetFeatures:
    embeddings: torch.Tensor
    lookup: dict
    provenance: dict

    @property
    def encoder_identity(self):
        return encoder_identity(self.provenance)

    def to(self, device):
        return PBCNetFeatures(self.embeddings.to(device), self.lookup, self.provenance)

    def validate_pairs(self, pairs):
        pairs = torch.as_tensor(pairs, dtype=torch.long, device="cpu")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("Expected compound-target pairs with shape [N, 2].")
        missing = [tuple(pair) for pair in pairs.tolist() if tuple(pair) not in self.lookup]
        if missing:
            sample = ", ".join(map(str, missing[:5]))
            raise ValueError(
                f"PBCNet2.0 structure features are missing for {len(missing)} requested pairs. "
                f"Example (drug_index, protein_index): {sample}. "
                "Provide prepared protein structures and docked poses for these pairs."
            )

    def select(self, edges):
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("Expected compound-target edges with shape [2, N].")
        pairs = edges.detach().to(device="cpu", dtype=torch.long).T
        self.validate_pairs(pairs)
        rows = torch.tensor(
            [self.lookup[tuple(pair)] for pair in pairs.tolist()],
            dtype=torch.long,
            device=self.embeddings.device,
        )
        return self.embeddings.index_select(0, rows)


def load_pbcnet_cache(path):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported PBCNet2.0 feature artifact: {path}")
    embeddings, keys = saved.get("embeddings"), saved.get("keys")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("PBCNet2.0 embeddings must be a two-dimensional tensor.")
    if embeddings.dtype != torch.float32 or embeddings.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected float32 PBCNet2.0 embeddings with {FEATURE_DIM} columns.")
    if saved.get("feature_dim") != FEATURE_DIM or not torch.isfinite(embeddings).all():
        raise ValueError("PBCNet2.0 feature dimension or finite-value validation failed.")
    if not isinstance(keys, list) or len(keys) != embeddings.shape[0] or not keys:
        raise ValueError("PBCNet2.0 keys must match the nonempty embedding tensor.")
    if not all(isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("PBCNet2.0 feature keys must be unique strings.")
    for key in keys:
        parts = key.rsplit("\t", 1)
        if len(parts) != 2 or len(parts[1]) != 64 or any(c not in "0123456789abcdef" for c in parts[1]):
            raise ValueError("Malformed canonical-SMILES/protein-sequence key.")
        if canonical_smiles(parts[0]) != parts[0]:
            raise ValueError("PBCNet2.0 artifact contains a noncanonical compound key.")
    if not isinstance(saved.get("provenance"), dict) or not isinstance(saved.get("records"), list):
        raise ValueError("PBCNet2.0 artifacts must include source provenance and per-pair records.")
    if len(saved["records"]) != len(keys):
        raise ValueError("PBCNet2.0 per-pair records do not match the feature rows.")
    encoder_identity(saved["provenance"])
    return saved


def load_pbcnet_features(path, smiles, sequences):
    saved = load_pbcnet_cache(path)
    drug_ids = {}
    for index, smiles_value in enumerate(smiles):
        drug_ids.setdefault(canonical_smiles(smiles_value), []).append(index)
    protein_ids = {}
    for index, sequence in enumerate(sequences):
        sequence_hash = hashlib.sha256(normalize_sequence(sequence).encode("ascii")).hexdigest()
        protein_ids.setdefault(sequence_hash, []).append(index)
    lookup = {}
    for row, key in enumerate(saved["keys"]):
        compound, sequence_hash = key.rsplit("\t", 1)
        for drug in drug_ids.get(compound, []):
            for protein in protein_ids.get(sequence_hash, []):
                lookup[(drug, protein)] = row
    provenance = dict(saved["provenance"])
    provenance["feature_file_sha256"] = sha256_file(path)
    provenance["feature_path"] = str(Path(path).resolve())
    provenance["feature_schema"] = SCHEMA
    return PBCNetFeatures(saved["embeddings"], lookup, provenance)
