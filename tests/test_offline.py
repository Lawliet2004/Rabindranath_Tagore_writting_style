"""Offline unit tests. Fixtures are synthetic test strings, never presented as Tagore works."""
from __future__ import annotations

import json
import contextlib
import io
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tagore_llm.common import atomic_bytes, canonical_hash, read_json, read_jsonl, sha256, write_json, write_jsonl
from tagore_llm.checkpoints import (IncompatibleRunError, choose_checkpoint, inspect_checkpoint,
                                    rotate_complete, seal_checkpoint)
from tagore_llm.corpus import (DiskDeduplicator, UnionFind, clean_text, make_sequences, prepare,
                               quality, select_documents, sequence_texts, split_groups, validate_dataset)
from tagore_llm.evaluation import GenerationSettings, repeated_ngram_fraction


class FixtureTokenizer:
    """Deterministic character tokenizer for boundary/ceiling tests, NOT the Qwen tokenizer."""
    eos_token_id = 2
    pad_token_id = 2
    all_special_ids = [2]
    special_tokens_map = {"eos_token": "<fixture-eos>", "pad_token": "<fixture-eos>"}
    @property
    def backend_tokenizer(self):
        return self
    def to_str(self):
        return "synthetic-character-tokenizer-v1"
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) + 10 for c in text]


CONFIG = {"model_id": "fixture/not-a-real-model", "revision": "fixture-revision", "max_tokens": 2000,
          "sequence_length": 128, "validation_fraction": 0.1, "seed": 42, "min_chars": 20,
          "min_bengali_ratio": 0.8, "near_duplicate_threshold": 0.85, "shared_passage_words": 16,
          "genre_weights": {"poems": 1, "essays": 1}}


def fake_checkpoint(run: Path, step, descriptor):
    """Structurally valid byte fixtures, not framework training results."""
    path = run / f"checkpoint-{step}"
    path.mkdir(parents=True)
    header = json.dumps({"test": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    atomic_bytes(path / "adapter_model.safetensors", struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
    write_json(path / "adapter_config.json", {"fixture": True})
    write_json(path / "trainer_state.json", {"global_step": step, "epoch": step / 100})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        atomic_bytes(path / name, b"STRUCTURE-ONLY TEST FIXTURE, NOT A TRAINING CHECKPOINT")
    seal_checkpoint(path, descriptor)
    return path


class CommonTests(unittest.TestCase):
    def test_atomic_jsonl_and_unicode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            rows = [{"text": "বাংলা\n\nপরীক্ষা"}]
            write_jsonl(path, rows)
            self.assertEqual(list(read_jsonl(path)), rows)
            self.assertFalse(list(Path(tmp).glob("*.partial")))

    def test_jsonl_line_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{}\nnot-json\n')
            with self.assertRaisesRegex(ValueError, ":2:"):
                list(read_jsonl(path))


class CorpusTests(unittest.TestCase):
    def test_cleaning_preserves_historical_spelling_joiners_and_stanzas(self):
        source = "\ufeffকরিয়া\r\nর্\u200cয\u200dা  \r\n\r\n  স্তবক॥\r\n"
        cleaned, changes = clean_text(source)
        self.assertEqual(cleaned, "করিয়া\nর্\u200cয\u200dা\n\n  স্তবক॥")
        self.assertTrue(changes)
        self.assertEqual(clean_text(cleaned), (cleaned, []))

    def test_corrupted_unicode_is_quarantined(self):
        q = quality("বাংলা পরীক্ষার লেখা। " * 6 + "\ufffd", CONFIG)
        self.assertEqual(q["cleaning_status"], "review")
        self.assertEqual(q["corrupt_characters"], 1)

    def test_first_line_indentation_is_preserved(self):
        self.assertEqual(clean_text("\n\n  প্রথম পঙ্‌ক্তি\n\n")[0], "  প্রথম পঙ্‌ক্তি")

    def test_wrong_language_is_rejected(self):
        self.assertEqual(quality("This fixture is English text only.", CONFIG)["cleaning_status"], "rejected")

    def test_paragraph_boundaries_and_exact_eos_overhead(self):
        tokenizer = FixtureTokenizer()
        text = "বাংলা অক্ষর\nদ্বিতীয় পঙ্‌ক্তি\n\nনূতন স্তবক\n\nশেষ স্তবক"
        doc = {"doc_id": "fixture", "work_id": "w1", "genre": "poems", "text": text}
        rows = make_sequences(doc, tokenizer, 35)
        self.assertEqual("\n\n".join(r["text"] for r in rows), text)
        for row in rows:
            self.assertLessEqual(row["token_count"], 35)
            self.assertEqual(row["input_ids"][-1], tokenizer.eos_token_id)
            self.assertEqual(row["token_count"], len(tokenizer.encode(row["text"])) + 1)
            self.assertEqual(row["loss_token_count"], row["token_count"] - 1)

    def test_overlong_stanza_is_reviewed_not_cut(self):
        with self.assertRaisesRegex(ValueError, "paragraph_or_stanza"):
            sequence_texts("অ" * 40, FixtureTokenizer(), 16)

    def test_budget_selects_whole_documents(self):
        docs = [{"doc_id": str(i), "genre": "poems", "token_count": n, "quality_score": 1} for i, n in enumerate([60, 40, 101])]
        selected, skipped = select_documents(docs, 100, {"poems": 1})
        self.assertEqual(sum(d["token_count"] for d in selected), 100)
        self.assertEqual(len(skipped), 1)
        with self.assertRaises(ValueError):
            select_documents(docs, 2_000_001, {"poems": 1})

    def test_transitive_groups_stay_together(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        docs = [{"doc_id": x, "split_group": uf.find(x), "token_count": 10} for x in "abcd"]
        assignment = split_groups(docs)
        self.assertEqual(assignment["a"], assignment["b"])
        self.assertEqual(assignment["b"], assignment["c"])
        self.assertNotEqual(assignment["c"], assignment["d"])

    def test_one_work_group_refused(self):
        with self.assertRaisesRegex(ValueError, "two independent"):
            split_groups([{"doc_id": "a", "split_group": "same", "token_count": 100}])

    def test_disk_dedup_detects_edition_and_shared_passage(self):
        with tempfile.TemporaryDirectory() as tmp:
            dedup = DiskDeduplicator(Path(tmp) / "dedup.sqlite", 16)
            words = " ".join(f"অক্ষর{i}" for i in range(70))
            s, p, duplicates, links = dedup.inspect("a", words, 0.85)
            self.assertEqual(duplicates, [])
            dedup.add("a", s, p)
            _, _, duplicates, links = dedup.inspect("b", words.replace("অক্ষর6 ", "পৃথক "), 0.85)
            self.assertEqual(duplicates[0]["other"], "a")
            self.assertIn("a", links)
            _, _, _, links = dedup.inspect("c", " ".join(words.split()[20:40]) + " বিশেষ লেখা", 0.85)
            self.assertIn("a", links)
            dedup.close()

    def test_end_to_end_synthetic_fixture_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, text in enumerate(["বাংলা অক্ষরের পরীক্ষামাত্র এই পঙ্‌ক্তি।", "স্বতন্ত্র বাক্যের নমুনাটুকু যাচাই করিবার জন্য।", "নানা শব্দে গঠিত পৃথক পরীক্ষার অনুচ্ছেদ।"]):
                doc = {"doc_id": str(i), "work_id": str(i), "title": "SYNTHETIC TEST FIXTURE", "genre": "poems",
                       "text": text, "author_verified": True, "rights_verified": True,
                       "is_container": False, "content_checksum": sha256(text)}
                write_json(root / f"data/raw/documents/{i}.json", doc)
            tokenizer = FixtureTokenizer()
            # Forbid network access in the entire preparation/audit path.
            with patch("socket.socket.connect", side_effect=AssertionError("Offline test attempted network access")):
                result = prepare(root, CONFIG, tokenizer)
                audit = validate_dataset(root, CONFIG, tokenizer)
            self.assertTrue(audit["verified"])
            self.assertEqual(audit["total_tokens"], result["statistics"]["total_tokens"])
            train_path = root / "data/processed/train.jsonl"
            train_path.write_text(train_path.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                validate_dataset(root, CONFIG, tokenizer)

    def test_removed_edition_bridges_work_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = " ".join("পরীক্ষাঅংশ" + str(i) for i in range(25))
            texts = [("a", original), ("b", original),
                     ("b", "অন্য অধ্যায়ের সম্পূর্ণ পৃথক পরীক্ষামাত্র বাক্য।"),
                     ("c", "স্বাধীন নমুনার নতুন অনুচ্ছেদের অক্ষরসমূহ যাচাই।")]
            for i, (work, text) in enumerate(texts):
                write_json(root / f"data/raw/documents/{i}.json", {"doc_id": str(i), "work_id": work,
                           "title": "SYNTHETIC FIXTURE", "genre": "poems", "text": text,
                           "content_checksum": sha256(text), "author_verified": True, "rights_verified": True})
            config = {**CONFIG, "sequence_length": 1024}
            prepare(root, config, FixtureTokenizer())
            metadata = {x["doc_id"]: x for x in read_jsonl(root / "data/processed/metadata.jsonl")}
            self.assertNotIn("1", metadata)
            self.assertEqual(metadata["0"]["split_group"], metadata["2"]["split_group"])
            self.assertEqual(metadata["0"]["split"], metadata["2"]["split"])


class CheckpointTests(unittest.TestCase):
    def test_latest_complete_selected_and_corrupt_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor = {"model": "fixture", "sequence_length": 32}
            first = fake_checkpoint(root, 25, descriptor)
            second = fake_checkpoint(root, 50, descriptor)
            (root / "checkpoint-75").mkdir()
            selected, _ = choose_checkpoint(root, descriptor)
            self.assertEqual(selected, second)
            (second / "optimizer.pt").write_bytes(b"damaged")
            selected, inventory = choose_checkpoint(root, descriptor)
            self.assertEqual(selected, first)
            self.assertEqual([r["status"] for r in inventory], ["invalid", "invalid", "complete"])

    def test_incompatible_experiment_never_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_checkpoint(root, 25, {"model": "a"})
            with self.assertRaises(IncompatibleRunError):
                choose_checkpoint(root, {"model": "b"})

    def test_rotation_keeps_valid_saves_not_incomplete_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in [25, 50, 75, 100]:
                fake_checkpoint(root, step, {"fixture": True})
            (root / "checkpoint-125").mkdir()
            rotate_complete(root, 3, {"fixture": True})
            self.assertFalse((root / "checkpoint-25").exists())
            self.assertTrue((root / "checkpoint-50").exists())
            self.assertTrue((root / "checkpoint-125").exists())

    def test_truncated_adapter_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fake_checkpoint(Path(tmp), 1, {})
            (path / "adapter_model.safetensors").write_bytes(b"truncated")
            self.assertEqual(inspect_checkpoint(path)["status"], "invalid")

    def test_fp16_checkpoint_requires_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fake_checkpoint(Path(tmp), 1, {})
            with self.assertRaisesRegex(IOError, "scaler.pt"):
                seal_checkpoint(path, {}, require_scaler=True)


class EvaluationTests(unittest.TestCase):
    def test_generation_settings_validation(self):
        GenerationSettings().validate()
        for value in [0, 1025]:
            with self.assertRaises(ValueError):
                GenerationSettings(max_new_tokens=value).validate()

    def test_excessive_repetition_signal(self):
        self.assertEqual(repeated_ngram_fraction(list(range(20))), 0)
        self.assertGreater(repeated_ngram_fraction([1, 2, 3, 4] * 20), 0.9)


class NotebookTests(unittest.TestCase):
    def test_all_default_notebooks_execute_without_download_or_training(self):
        root = Path(__file__).resolve().parents[1]
        with contextlib.chdir(root), contextlib.redirect_stdout(io.StringIO()), patch(
                "socket.socket.connect", side_effect=AssertionError("Default notebook attempted network access")):
            for path in sorted((root / "notebooks").glob("*.ipynb")):
                with self.subTest(notebook=path.name):
                    namespace = {"__name__": "__notebook_offline_test__"}
                    for cell in read_json(path)["cells"]:
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), str(path) + ":" + cell["id"], "exec"), namespace)
                    for name, value in namespace.items():
                        if name.startswith("RUN_") or name in {"LOAD_MODEL", "CHECK_MODEL_AND_TOKENIZER", "SHOW_WIDGET"}:
                            self.assertIs(value, False, f"Unsafe default: {name}")


if __name__ == "__main__":
    unittest.main()
