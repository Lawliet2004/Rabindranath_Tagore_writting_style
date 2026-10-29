from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from typing import Iterable


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def atomic_bytes(path: Path, data: bytes):
    """Write on the same filesystem, fsync file, replace, then fsync directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict]):
    # Corpus bounded to <=2M tokens; raw downloads remain individual files.
    atomic_bytes(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode())


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
                yield row
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {e}") from e


def root_path(start=None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        if (p / "configs/project.json").is_file() and (p / "src/tagore_llm").is_dir():
            return p
    raise FileNotFoundError("Start Jupyter in this project or its notebooks/ directory.")


def load_config(root: Path, name="project"):
    return read_json(root / "configs" / f"{name}.json")


def progress(items, **kwargs):
    try:
        from tqdm.auto import tqdm
        return tqdm(items, **kwargs)
    except ImportError:
        return items


def tokenizer_for(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("The pinned tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash the actual tokenizer graph and special token configuration."""
    return canonical_hash({"backend": tokenizer.backend_tokenizer.to_str(),
                           "special_tokens": tokenizer.special_tokens_map,
                           "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id})
