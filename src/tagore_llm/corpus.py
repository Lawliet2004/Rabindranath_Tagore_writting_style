"""Structure-preserving QC, disk-backed deduplication, exact sequence budgets, and leakage audits."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import random
import re
import sqlite3
import unicodedata

from .common import (atomic_bytes, canonical_hash, file_hash, now, progress, read_json, read_jsonl,
                     sha256, tokenizer_fingerprint, write_json, write_jsonl)


def clean_text(text: str) -> tuple[str, list[dict]]:
    """No NFC/NFKC, spelling changes, punctuation substitution, or speculative OCR repair."""
    changes = []
    if text.startswith("\ufeff"):
        text = text[1:]
        changes.append({"operation": "remove_initial_UTF8_BOM", "count": 1})
    count = text.count("\r")
    if count:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        changes.append({"operation": "CRLF_or_CR_to_LF", "count": count})
    # HTML entity NBSP is a layout separator; preserve all other Bengali spacing, ZWJ and ZWNJ.
    count = text.count("\xa0")
    if count:
        text = text.replace("\xa0", " ")
        changes.append({"operation": "NBSP_to_space", "count": count})
    lines = text.split("\n")
    count = sum(line != line.rstrip(" \t") for line in lines)
    if count:
        text = "\n".join(line.rstrip(" \t") for line in lines)
        changes.append({"operation": "remove_trailing_horizontal_whitespace", "lines": count})
    text, count = re.subn(r"\n{3,}", "\n\n", text)
    if count:
        changes.append({"operation": "collapse_3plus_blank_lines_to_stanza_boundary", "runs": count})
    stripped = text.strip("\n")
    if stripped != text:
        changes.append({"operation": "trim_document_outer_newlines", "characters": len(text) - len(stripped)})
    return stripped, changes


def quality(text: str, config: dict) -> dict:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    bengali = sum("\u0980" <= c <= "\u09ff" for c in letters)
    bn_ratio = bengali / max(1, len(letters))
    corrupt = sum(c == "\ufffd" or unicodedata.category(c) == "Cs" or
                  (unicodedata.category(c) == "Cc" and c not in "\n\t") for c in text)
    # These are review signals, never automatic rewrites.
    suspicious = len(re.findall(r"(?:[A-Za-z]{12,}|[|¦]{2,}|[?]{4,}|[০-৯0-9]{8,}|([\u0980-\u09ff])\1{7,})", text))
    mojibake = len(re.findall(r"(?:à¦|à§|Ã.|â€)", text))
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    numeric_ratio = sum(bool(re.fullmatch(r"[০-৯0-9 \-–—]+", x)) for x in lines) / max(1, len(lines))
    paragraphs = [re.sub(r"\s+", " ", x).strip() for x in text.split("\n\n") if len(x.strip()) >= 100]
    repeated_paragraphs = len(paragraphs) - len(set(paragraphs))
    reasons = []
    status = "accepted"
    if len(text) < config["min_chars"] or not letters:
        status = "rejected"
        reasons.append("minimum_content_length")
    if bn_ratio < config["min_bengali_ratio"]:
        status = "rejected"
        reasons.append("not_predominantly_Bengali")
    if corrupt or mojibake:
        status = "review" if status != "rejected" else status
        reasons.append("corruption_or_mojibake")
    if suspicious or numeric_ratio > 0.15:
        status = "review" if status != "rejected" else status
        reasons.append("suspicious_OCR_or_page_numbers")
    if len(paragraphs) >= 4 and repeated_paragraphs / len(paragraphs) > 0.35:
        status = "review" if status != "rejected" else status
        reasons.append("repeated_long_paragraphs_check_scan_vs_intentional_refrain")
    return {"quality_score": round(max(0, min(1, bn_ratio - 0.15 * bool(corrupt or mojibake) - 0.05 * bool(suspicious))), 4),
            "bengali_letter_ratio": round(bn_ratio, 5), "corrupt_characters": corrupt,
            "suspicious_patterns": suspicious, "mojibake_patterns": mojibake,
            "repeated_long_paragraphs": repeated_paragraphs,
            "numeric_line_ratio": numeric_ratio, "cleaning_status": status, "reasons": reasons}


def normalized_words(text: str) -> list[str]:
    # Only duplicate fingerprints use this representation. Training text is never normalized this way.
    return re.findall(r"[\w\u200c\u200d]+", text, flags=re.UNICODE)


def shingles(text: str, n: int) -> set[bytes]:
    words = normalized_words(text)
    return {hashlib.blake2b(" ".join(words[i:i+n]).encode(), digest_size=12).digest()
            for i in range(max(0, len(words) - n + 1))}


class UnionFind:
    def __init__(self):
        self.parents = {}

    def find(self, x):
        self.parents.setdefault(x, x)
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


class DiskDeduplicator:
    """SQLite inverted indexes: all candidates verified by exact shingle overlap, no approximate LSH misses."""
    def __init__(self, path: Path, passage_words: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("DROP TABLE IF EXISTS s; DROP TABLE IF EXISTS p; DROP TABLE IF EXISTS current; "
                              "CREATE TABLE s (h BLOB, doc TEXT, PRIMARY KEY(h,doc)); "
                              "CREATE TABLE p (h BLOB PRIMARY KEY, doc TEXT); "
                              "CREATE TEMP TABLE current (h BLOB PRIMARY KEY);")
        self.sizes = {}
        self.passage_words = passage_words

    def inspect(self, doc: str, text: str, threshold: float):
        s = shingles(text, 5)
        self.db.execute("DELETE FROM current")
        self.db.executemany("INSERT INTO current VALUES (?)", ((h,) for h in s))
        counts = self.db.execute("SELECT s.doc, COUNT(*) FROM s JOIN current USING(h) GROUP BY s.doc").fetchall()
        duplicates = []
        for other, shared in counts:
            jaccard = shared / max(1, len(s) + self.sizes[other] - shared)
            containment = shared / max(1, min(len(s), self.sizes[other]))
            if jaccard >= threshold or (containment >= 0.95 and shared >= 12):
                duplicates.append({"other": other, "jaccard": jaccard, "containment": containment})
        p = shingles(text, self.passage_words)
        self.db.execute("DELETE FROM current")
        self.db.executemany("INSERT INTO current VALUES (?)", ((h,) for h in p))
        linked = [r[0] for r in self.db.execute("SELECT DISTINCT p.doc FROM p JOIN current USING(h)")]
        return s, p, duplicates, linked

    def add(self, doc, s, p):
        self.db.executemany("INSERT OR IGNORE INTO s VALUES (?,?)", ((h, doc) for h in s))
        self.db.executemany("INSERT OR IGNORE INTO p VALUES (?,?)", ((h, doc) for h in p))
        self.sizes[doc] = len(s)
        self.db.commit()

    def close(self):
        self.db.close()


def sequence_texts(text: str, tokenizer, max_length: int) -> list[str]:
    """Split only at paragraph/stanza boundaries. An overlong paragraph is held for review.

    Each sequence is independently tokenized and gets exactly one EOS. No cross-work
    concatenation and no hidden overlap or repeated context; no poem is cut to fill a budget.
    """
    if max_length < 8:
        raise ValueError("sequence_length must be at least 8")
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(tokenizer.encode(paragraph, add_special_tokens=False)) + 1 > max_length:
            raise ValueError("paragraph_or_stanza_exceeds_sequence_length")
        candidate = current + "\n\n" + paragraph if current else paragraph
        if len(tokenizer.encode(candidate, add_special_tokens=False)) + 1 <= max_length:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def make_sequences(doc, tokenizer, max_length):
    chunks = sequence_texts(doc["text"], tokenizer, max_length)
    result = []
    for i, text in enumerate(chunks):
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Reject technical special tokens embedded in source text.
        if any(x in set(tokenizer.all_special_ids) for x in ids):
            raise ValueError("source_contains_model_special_token")
        ids.append(tokenizer.eos_token_id)
        if len(ids) < 2:
            raise ValueError("sequence_has_no_predictable_token")
        result.append({"sequence_id": f"{doc['doc_id']}:{i}", "doc_id": doc["doc_id"],
                       "work_id": doc["work_id"], "genre": doc["genre"], "text": text,
                       "input_ids": ids, "token_count": len(ids), "loss_token_count": len(ids) - 1})
    return result


def select_documents(candidates, max_tokens: int, genre_weights, seed=42):
    """Weighted round-robin by selected token mass; complete documents only, soft genre preferences."""
    if not 1 <= max_tokens <= 2_000_000:
        raise ValueError("max_tokens must be between 1 and 2,000,000")
    buckets = defaultdict(list)
    rng = random.Random(seed)
    for item in candidates:
        buckets[item["genre"]].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        bucket.sort(key=lambda x: -x["quality_score"])
    for genre in buckets:
        if float(genre_weights.get(genre, 1)) <= 0:
            raise ValueError(f"Genre weight must be positive: {genre}")
    used = Counter()
    selected, skipped, total = [], [], 0
    while buckets:
        genre = min(buckets, key=lambda g: (used[g] / float(genre_weights.get(g, 1)), g))
        item = buckets[genre].pop(0)
        if total + item["token_count"] <= max_tokens:
            selected.append(item)
            total += item["token_count"]
            used[genre] += item["token_count"]
        else:
            skipped.append({"doc_id": item["doc_id"], "reason": "whole_document_does_not_fit_budget", "tokens": item["token_count"]})
        if not buckets[genre]:
            del buckets[genre]
    return selected, skipped


def split_groups(docs, fraction=0.1, seed=42):
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be >0 and <0.5")
    groups = defaultdict(list)
    for d in docs:
        groups[d["split_group"]].append(d)
    if len(groups) < 2:
        raise ValueError("At least two independent work groups are required; add verified works. Do not split overlapping chunks.")
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    target = sum(d["token_count"] for d in docs) * fraction
    val, mass = set(), 0
    for group in ids:
        size = sum(d["token_count"] for d in groups[group])
        if abs(target - (mass + size)) < abs(target - mass):
            val.add(group)
            mass += size
    if not val:
        val.add(min(ids, key=lambda g: abs(sum(d["token_count"] for d in groups[g]) - target)))
    if len(val) == len(groups):
        val.remove(max(val, key=lambda g: sum(d["token_count"] for d in groups[g])))
    return {d["doc_id"]: "validation" if d["split_group"] in val else "train" for d in docs}


def prepare(root: Path, config: dict, tokenizer):
    """Run offline after collection/manual import; no model downloads in this function."""
    raw_paths = sorted((root / "data/raw/documents").glob("*.json"))
    if not raw_paths:
        raise FileNotFoundError("No source documents. Run collection or import an authorized manual manifest first.")
    decisions_path = root / "data/metadata/review_decisions.jsonl"
    decisions = {r["doc_id"]: r for r in read_jsonl(decisions_path)} if decisions_path.exists() else {}
    rejected, reviews, candidates, all_meta, edges = [], [], [], [], []
    counts = Counter()
    exact = {}
    uf = UnionFind()
    work_owner = {}
    dedup = DiskDeduplicator(root / "data/metadata/dedup.sqlite", config["shared_passage_words"])
    try:
        for path in progress(raw_paths, desc="QC / exact tokenization"):
            doc = read_json(path)
            text, transforms = clean_text(doc["text"])
            counts["collected_documents"] += 1
            counts["collected_characters"] += len(doc["text"])
            # The raw checksum must match its unmodified extracted text.
            if sha256(doc["text"]) != doc["content_checksum"]:
                raise ValueError(f"Raw document checksum mismatch: {path}")
            q = quality(text, config)
            reasons = q["reasons"][:]
            if doc.get("is_container"):
                q["cleaning_status"] = "rejected"
                reasons.append("container_or_table_of_contents")
            uncertain = doc.get("extraction_warnings", [])[:]
            if not doc.get("author_verified"):
                uncertain.append("authorship_unverified")
            if not doc.get("rights_verified"):
                uncertain.append("rights_or_edition_unverified")
            if uncertain and q["cleaning_status"] != "rejected":
                q["cleaning_status"] = "review"
                reasons += uncertain
            # Approval is bound to the exact cleaned content hash and requires evidence and original-text review.
            decision = decisions.get(doc["doc_id"], {})
            if decision.get("decision") == "reject":
                q["cleaning_status"] = "rejected"
                reasons.append("manual_rejection")
            elif decision.get("decision") == "approve" and q["cleaning_status"] == "review":
                if decision.get("content_checksum") != sha256(text) or not decision.get("rights_evidence_url") or not decision.get("reviewer") or decision.get("reviewed_original_text") is not True:
                    raise ValueError(f"Invalid/stale review approval: {doc['doc_id']}")
                # Broken encoding must be corrected through a new documented source import.
                if not q["corrupt_characters"] and not q["mojibake_patterns"]:
                    q["cleaning_status"] = "accepted"
            cleaned = {**doc, "text": text, **q, "reasons": reasons, "transformations": transforms,
                       "content_checksum": sha256(text), "raw_document_path": str(path.relative_to(root)),
                       "cleaned_path": f"data/cleaned/{doc['doc_id']}.json"}
            write_json(root / cleaned["cleaned_path"], cleaned)
            meta = {k: v for k, v in cleaned.items() if k != "text"}
            all_meta.append(meta)
            if q["cleaning_status"] != "accepted":
                (reviews if q["cleaning_status"] == "review" else rejected).append(meta)
                counts[q["cleaning_status"] + "_characters"] += len(text)
                continue
            did = doc["doc_id"]
            # Whitespace-only differences in an edition count as exact duplicate fingerprints.
            key = sha256(re.sub(r"\s+", " ", text).strip())
            if key in exact:
                anchor = exact[key]
                if doc["work_id"] in work_owner:
                    uf.union(anchor, work_owner[doc["work_id"]])
                work_owner.setdefault(doc["work_id"], anchor)
                rejected.append({**meta, "cleaning_status": "deduplicated", "reasons": ["exact_duplicate"], "duplicate_of": exact[key]})
                counts["deduplicated_characters"] += len(text)
                continue
            s, p, duplicates, linked = dedup.inspect(did, text, config["near_duplicate_threshold"])
            if duplicates:
                anchor = duplicates[0]["other"]
                if doc["work_id"] in work_owner:
                    uf.union(anchor, work_owner[doc["work_id"]])
                work_owner.setdefault(doc["work_id"], anchor)
                rejected.append({**meta, "cleaning_status": "deduplicated", "reasons": ["near_duplicate_or_contained_edition"], "matches": duplicates})
                counts["deduplicated_characters"] += len(text)
                # Preserve transitive leakage links even when this edition is removed.
                for other in linked + [x["other"] for x in duplicates]:
                    uf.union(other, duplicates[0]["other"])
                continue
            try:
                sequences = make_sequences(cleaned, tokenizer, config["sequence_length"])
            except ValueError as e:
                reviews.append({**meta, "cleaning_status": "review", "reasons": reasons + [str(e)]})
                counts["review_characters"] += len(text)
                continue
            token_count = sum(x["token_count"] for x in sequences)
            counts["eligible_tokens_before_budget"] += token_count
            uf.find(did)
            if doc["work_id"] in work_owner:
                uf.union(did, work_owner[doc["work_id"]])
            work_owner.setdefault(doc["work_id"], did)
            for other in linked:
                uf.union(did, other)
                edges.append({"a": did, "b": other, "reason": "shared_contiguous_passage"})
            exact[key] = did
            dedup.add(did, s, p)
            candidates.append({**meta, "token_count": token_count, "characters": len(text), "sequence_count": len(sequences)})
    finally:
        dedup.close()
    for d in candidates:
        d["split_group"] = uf.find(d["doc_id"])
    selected, skipped = select_documents(candidates, config["max_tokens"], config["genre_weights"], config["seed"])
    # Write all QC artifacts even when there is not enough text to form a safe split.
    output = root / "data/processed"
    write_jsonl(output / "rejected_documents.jsonl", rejected)
    write_jsonl(root / "data/metadata/manual_review_queue.jsonl", reviews)
    write_jsonl(root / "data/metadata/duplicate_passage_groups.jsonl", edges)
    write_jsonl(root / "data/metadata/budget_exclusions.jsonl", skipped)
    report = {**dict(counts), "eligible_documents": len(candidates), "retained_documents": len(selected),
              "retained_characters": sum(d["characters"] for d in selected), "review_documents": len(reviews),
              "rejected_or_deduplicated_documents": len(rejected), "budget_excluded_documents": len(skipped),
              "shared_passage_edges": len(edges),
              "duplicate_passage_policy": "Duplicate documents/editions removed; remaining shared passages grouped into one split, preserving legitimate quotations/refrains.",
              "unicode_normalization": "none; all transformations logged per document"}
    write_json(output / "quality_report.json", report)
    assignments = split_groups(selected, config["validation_fraction"], config["seed"])
    rows = {"train": [], "validation": []}
    metadata = []
    backup = []
    for d in selected:
        doc = read_json(root / d["cleaned_path"])
        split = assignments[d["doc_id"]]
        sequences = make_sequences(doc, tokenizer, config["sequence_length"])
        for row in sequences:
            row["split_group"] = d["split_group"]
        rows[split].extend(sequences)
        metadata.append({**d, "split": split})
        backup.append(doc["text"])
    for split in rows:
        write_jsonl(output / f"{split}.jsonl", rows[split])
    write_jsonl(output / "metadata.jsonl", metadata)
    atomic_bytes(output / "corpus.txt", "\n\n\n".join(backup).encode())
    stats = {}
    for split, seqs in rows.items():
        stats[split] = {"tokens": sum(r["token_count"] for r in seqs), "loss_tokens": sum(r["loss_token_count"] for r in seqs),
                        "sequences": len(seqs), "documents": len({r["doc_id"] for r in seqs}),
                        "genre_tokens": dict(Counter({g: sum(r["token_count"] for r in seqs if r["genre"] == g) for g in sorted({r["genre"] for r in seqs})}))}
    total = sum(s["tokens"] for s in stats.values())
    stats.update({"total_tokens": total, "eos_overhead": sum(len(r) for r in rows.values()),
                  "content_tokens": total - sum(len(r) for r in rows.values()), "padding_tokens_in_budget": 0,
                  "padding_note": "Padding is generated at collation, masked from attention and loss; it is not literary training data.",
                  "achieved_validation_token_fraction": stats["validation"]["tokens"] / max(1, total)})
    write_json(output / "token_statistics.json", stats)
    names = ["train.jsonl", "validation.jsonl", "metadata.jsonl", "corpus.txt", "quality_report.json", "rejected_documents.jsonl", "token_statistics.json"]
    manifest = {"schema_version": 1, "created_at": now(), "model_id": config["model_id"], "revision": config["revision"],
                "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer), "sequence_length": config["sequence_length"],
                "max_tokens": config["max_tokens"], "total_tokens": total, "preparation_config": config,
                "files": {n: file_hash(output / n) for n in names}, "sequence_policy": "paragraph/stanza packing within a single document + one EOS per sequence; no overlap",
                "split_policy": "whole canonical works, duplicate editions and shared passages connected before splitting"}
    manifest["dataset_id"] = canonical_hash({k: v for k, v in manifest.items() if k != "created_at"})
    write_json(output / "dataset_manifest.json", manifest)
    validate_dataset(root, config, tokenizer)
    return {"manifest": manifest, "statistics": stats, "quality": report}


def validate_dataset(root: Path, config, tokenizer):
    output = root / "data/processed"
    m = read_json(output / "dataset_manifest.json")
    if m["dataset_id"] != canonical_hash({k: v for k, v in m.items() if k not in {"created_at", "dataset_id"}}):
        raise ValueError("Dataset manifest identity does not match its contents")
    if m["model_id"] != config["model_id"] or m["revision"] != config["revision"]:
        raise ValueError("Dataset model/revision differs from the project configuration.")
    if m["tokenizer_fingerprint"] != tokenizer_fingerprint(tokenizer):
        raise ValueError("Tokenizer content changed; rebuild the dataset with the pinned tokenizer.")
    for name, checksum in m["files"].items():
        if file_hash(output / name) != checksum:
            raise ValueError(f"Dataset integrity check failed: {name}")
    metadata = list(read_jsonl(output / "metadata.jsonl"))
    by_doc = {d["doc_id"]: d for d in metadata}
    if len(by_doc) != len(metadata):
        raise ValueError("Duplicate document metadata")
    sets, docs, works, passages = {}, {}, {}, {}
    observed_counts, observed_sequences = Counter(), Counter()
    observed_texts = defaultdict(list)
    totals, per_split, seen_seq = 0, {}, set()
    for split in ("train", "validation"):
        sets[split], docs[split], works[split], passages[split] = set(), set(), set(), set()
        count = 0
        for row in read_jsonl(output / f"{split}.jsonl"):
            for required in ("sequence_id", "doc_id", "work_id", "split_group", "genre", "text", "input_ids", "token_count", "loss_token_count"):
                if required not in row:
                    raise ValueError(f"Missing {required} in {split}")
            ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if ids != row["input_ids"] or len(ids) != row["token_count"] or len(ids) - 1 != row["loss_token_count"]:
                raise ValueError(f"Exact tokenizer/count mismatch: {row['sequence_id']}")
            if not 2 <= len(ids) <= m["sequence_length"]:
                raise ValueError("Invalid sequence length")
            if row["sequence_id"] in seen_seq:
                raise ValueError("Repeated sequence ID")
            expected = by_doc.get(row["doc_id"])
            if expected is None or expected["split"] != split or any(row[k] != expected[k] for k in ("work_id", "genre", "split_group")):
                raise ValueError("Sequence provenance/split differs from document metadata")
            seen_seq.add(row["sequence_id"])
            sets[split].add(row["split_group"])
            docs[split].add(row["doc_id"])
            works[split].add(row["work_id"])
            observed_counts[row["doc_id"]] += len(ids)
            observed_sequences[row["doc_id"]] += 1
            observed_texts[row["doc_id"]].append(row["text"])
            count += len(ids)
        # Audit complete original documents, including passages crossing sequence boundaries.
        for d in metadata:
            if d["split"] == split:
                doc = read_json(root / d["cleaned_path"])
                if sha256(doc["text"]) != d["content_checksum"]:
                    raise ValueError("Cleaned document checksum changed after preparation")
                if "\n\n".join(observed_texts[d["doc_id"]]) != doc["text"]:
                    raise ValueError("Literary text omitted, reordered or duplicated during sequence construction")
                if observed_counts[d["doc_id"]] != d["token_count"] or observed_sequences[d["doc_id"]] != d["sequence_count"]:
                    raise ValueError("Document token/sequence count differs from metadata")
                passages[split].update(shingles(doc["text"], config["shared_passage_words"]))
        if not count:
            raise ValueError(f"Empty {split} split")
        per_split[split] = count
        totals += count
    for name, values in [("group", sets), ("document", docs), ("work", works), ("contiguous passage", passages)]:
        if values["train"] & values["validation"]:
            raise ValueError(f"Train/validation {name} leakage detected")
    if totals != m["total_tokens"] or totals > min(m["max_tokens"], config["max_tokens"], 2_000_000):
        raise ValueError("Dataset token ceiling or manifest mismatch")
    stats = read_json(output / "token_statistics.json")
    if totals != stats["total_tokens"] or any(per_split[s] != stats[s]["tokens"] for s in per_split):
        raise ValueError("Token statistics mismatch")
    return {"verified": True, "total_tokens": totals, "split_tokens": per_split, "dataset_id": m["dataset_id"]}
