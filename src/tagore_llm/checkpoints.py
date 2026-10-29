"""Full-state checkpoint sealing, corruption detection, compatibility checks, and recovery selection."""
from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import struct

from .common import canonical_hash, file_hash, now, read_json, write_json

REQUIRED = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
            "trainer_state.json", "rng_state.pth", "training_args.bin", "run_fingerprint.json")
MARKER = "COMPLETE.json"


class IncompatibleRunError(RuntimeError):
    pass


def check_safetensors(path: Path):
    """Check header and tensor byte ranges without allocating tensors or loading pickle."""
    size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", header)[0]
        if length > min(100_000_000, size - 8):
            raise ValueError("Invalid safetensors header length")
        tensors = json.loads(f.read(length))
    if not isinstance(tensors, dict) or not any(k != "__metadata__" for k in tensors):
        raise ValueError("No tensors in adapter")
    for key, value in tensors.items():
        if key == "__metadata__":
            continue
        a, b = value["data_offsets"]
        if not 0 <= a <= b <= size - 8 - length:
            raise ValueError(f"Tensor out of bounds: {key}")


def seal_checkpoint(path: Path, run_descriptor: dict, require_scaler=False):
    write_json(path / "run_fingerprint.json", run_descriptor)
    required = list(REQUIRED) + (["scaler.pt"] if require_scaler else [])
    for name in required:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise IOError(f"Checkpoint incomplete: {path / name}")
    state = read_json(path / "trainer_state.json")
    if int(path.name.split("-")[-1]) != state["global_step"]:
        raise ValueError("Checkpoint directory/global step mismatch")
    read_json(path / "adapter_config.json")
    check_safetensors(path / "adapter_model.safetensors")
    # Flush framework-written files before recording the completion marker.
    import os
    for name in required:
        with (path / name).open("rb") as f:
            os.fsync(f.fileno())
    files = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
             for p in path.iterdir() if p.is_file() and p.name != MARKER and not p.name.endswith(".partial")}
    write_json(path / MARKER, {"schema_version": 1, "sealed_at": now(), "global_step": state["global_step"],
                              "run_fingerprint": canonical_hash(run_descriptor), "require_scaler": require_scaler,
                              "files": files})


def inspect_checkpoint(path: Path, descriptor=None) -> dict:
    result = {"path": str(path), "status": "invalid", "reason": ""}
    try:
        if not re.fullmatch(r"checkpoint-\d+", path.name):
            raise ValueError("Not a checkpoint directory")
        if not (path / MARKER).is_file():
            raise ValueError("No completion marker (interrupted or unverified save)")
        marker = read_json(path / MARKER)
        required = list(REQUIRED) + (["scaler.pt"] if marker.get("require_scaler") else [])
        for name in required:
            if name not in marker["files"]:
                raise ValueError(f"Required file absent from seal: {name}")
        for name, spec in marker["files"].items():
            if Path(name).name != name:
                raise ValueError("Unsafe filename in checkpoint seal")
            p = path / name
            if not p.is_file() or p.stat().st_size != spec["bytes"] or file_hash(p) != spec["sha256"]:
                raise ValueError(f"Missing or corrupted file: {name}")
        saved = read_json(path / "run_fingerprint.json")
        if canonical_hash(saved) != marker["run_fingerprint"]:
            raise ValueError("Run fingerprint seal mismatch")
        state = read_json(path / "trainer_state.json")
        step = int(path.name.split("-")[-1])
        if state["global_step"] != step or marker["global_step"] != step:
            raise ValueError("Trainer state/global step mismatch")
        check_safetensors(path / "adapter_model.safetensors")
        if descriptor is not None and canonical_hash(saved) != canonical_hash(descriptor):
            return {**result, "status": "incompatible", "reason": "Model, dataset, training configuration, code, or package versions differ", "global_step": step}
        return {**result, "status": "complete", "reason": "All required files verified by SHA-256", "global_step": step}
    except (OSError, ValueError, KeyError, TypeError, struct.error) as e:
        return {**result, "reason": str(e)}


def checkpoint_inventory(run_dir: Path, descriptor=None):
    paths = [p for p in run_dir.glob("checkpoint-*") if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    paths.sort(key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    return [inspect_checkpoint(p, descriptor) for p in paths]


def choose_checkpoint(run_dir: Path, descriptor):
    inventory = checkpoint_inventory(run_dir, descriptor)
    valid = [r for r in inventory if r["status"] == "complete"]
    if valid:
        return Path(valid[0]["path"]), inventory
    if any(r["status"] == "incompatible" for r in inventory):
        raise IncompatibleRunError("No compatible checkpoint; use a new run_name or force_fresh=true. Existing experiments are preserved.")
    return None, inventory


def rotate_complete(run_dir: Path, keep: int, descriptor):
    if keep < 1:
        raise ValueError("Keep at least one complete checkpoint")
    valid = [r for r in checkpoint_inventory(run_dir, descriptor) if r["status"] == "complete"]
    # Incomplete folders are left for diagnosis; never count them against recoverable saves.
    for row in valid[keep:]:
        shutil.rmtree(row["path"])
