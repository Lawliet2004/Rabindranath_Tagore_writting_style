"""One-model base/adapter comparisons, human ratings, and conservative memorization signals."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from .common import (canonical_hash, file_hash, now, read_json, read_jsonl, tokenizer_for,
                     write_json, write_jsonl)

PROMPTS = {
    "Bengali poetry": "রাত্রির নীরব আকাশে একটি তারা জেগে আছে,\nনদীর জলে",
    "Nature imagery": "বর্ষার প্রথম মেঘ মাঠের উপর ছায়া ফেলিয়াছে। দূরের তালবনে",
    "Emotional prose": "অনেক দিন পরে পুরাতন চিঠিখানি হাতে লইয়া সে জানালার কাছে বসিল।",
    "Short-story continuation": "গ্রামের শেষ প্রান্তে ছোটো ডাকঘরটির সামনে আজ একটি অপরিচিত বালক দাঁড়াইয়া ছিল।",
    "Dialogue writing": "— তুমি আবার কবে আসিবে?\n— সে কথা আজ কেমন করিয়া বলি?\n",
    "Reflective philosophical prose": "মানুষের স্বাধীনতা কেবল বাহিরের বন্ধন কাটিয়া পাওয়া যায় না।",
    "Free-form literary generation": "সন্ধ্যার আলো নিবিয়া আসিতেছে। পথের ধারে",
}
RATING_FIELDS = ("bengali_fluency", "grammar", "coherence", "vocabulary", "poetic_quality",
                 "stylistic_resemblance", "originality", "unwanted_repetition")


@dataclass
class GenerationSettings:
    max_new_tokens: int = 160
    temperature: float = 0.85
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.08
    seed: int = 42

    def validate(self):
        if not 1 <= self.max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be 1..1024")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Use temperature 0..2 and top_p >0..1")
        if self.top_k < 0 or not 0.5 <= self.repetition_penalty <= 2:
            raise ValueError("Use top_k>=0 and repetition_penalty 0.5..2")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be 0..2^32-1")


def validate_adapter(path: Path, project):
    metadata = read_json(path / "export.json")
    adapter = read_json(path / "adapter_config.json")
    if metadata["model_id"] != project["model_id"] or metadata["revision"] != project["revision"]:
        raise ValueError("Adapter base repository/revision mismatch")
    if adapter["base_model_name_or_path"] != project["model_id"] or adapter.get("revision") != project["revision"]:
        raise ValueError("PEFT adapter configuration does not match the pinned base")
    for name, expected in metadata["files"].items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError(f"Adapter export integrity check failed: {name}")
    return metadata


def repeated_ngram_fraction(ids, n=4):
    grams = [tuple(ids[i:i+n]) for i in range(max(0, len(ids) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


class MemorizationIndex:
    """Exact token overlap index. A review aid, not a proof of originality or copyright status."""
    def __init__(self, root: Path, n=24):
        self.root, self.n = root, n
        path = root / "evaluations/memorization.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        m = read_json(root / "data/processed/dataset_manifest.json")
        signature = canonical_hash({"dataset_id": m["dataset_id"], "n": n})
        old = self.db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if not old or old[0] != signature:
            self.db.executescript("DROP TABLE IF EXISTS grams; DROP TABLE IF EXISTS sequences; "
                                  "CREATE TABLE grams (h BLOB, seq TEXT, PRIMARY KEY(h,seq)); "
                                  "CREATE TABLE sequences (seq TEXT PRIMARY KEY, doc TEXT, ids TEXT);")
            for row in read_jsonl(root / "data/processed/train.jsonl"):
                ids = row["input_ids"][:-1]  # EOS is structural, not a literary match.
                self.db.execute("INSERT INTO sequences VALUES (?,?,?)", (row["sequence_id"], row["doc_id"], json.dumps(ids)))
                self.db.executemany("INSERT OR IGNORE INTO grams VALUES (?,?)", ((h, row["sequence_id"]) for h in self._grams(ids)))
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('signature',?)", (signature,))
            self.db.commit()

    def _grams(self, ids):
        return {hashlib.sha256(",".join(map(str, ids[i:i+self.n])).encode()).digest() for i in range(max(0, len(ids) - self.n + 1))}

    def check(self, ids, tokenizer):
        candidates = Counter()
        for h in self._grams(ids):
            candidates.update(r[0] for r in self.db.execute("SELECT seq FROM grams WHERE h=?", (h,)))
        matches = []
        for seq, hits in candidates.most_common(8):
            doc, raw = self.db.execute("SELECT doc, ids FROM sequences WHERE seq=?", (seq,)).fetchone()
            source = json.loads(raw)
            match = SequenceMatcher(None, source, ids, autojunk=False).find_longest_match()
            if match.size >= self.n:
                matches.append({"sequence_id": seq, "doc_id": doc, "longest_exact_tokens": match.size,
                                "generation_start_token": match.b,
                                "matching_excerpt": tokenizer.decode(ids[match.b:match.b+match.size])})
        return {"review_recommended": bool(matches), "threshold_tokens": self.n, "matches": matches,
                "note": "Only exact >=24-token sequence-local matches are indexed. Short, paraphrased or cross-sequence reproduction may be missed; inspect source metadata."}

    def close(self):
        self.db.close()


class ComparisonSession:
    """One text backbone with PEFT toggled off/on, identical prompts and seeds."""
    def __init__(self, root, project, adapter_path=None, device=None):
        import torch
        from peft import PeftModel
        from .modeling import load_text_model
        self.root, self.project = root, project
        if adapter_path is None:
            adapter_path = Path(read_json(root / "models/latest_adapter.json")["adapter_path"])
        self.adapter_path = Path(adapter_path).expanduser().resolve()
        self.export = validate_adapter(self.adapter_path, project)
        self.tokenizer = tokenizer_for(project)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, self.loading_report = load_text_model(project, self.device, training=False)
        self.model = PeftModel.from_pretrained(model, self.adapter_path, is_trainable=False)
        self.model.eval()
        self.model.config.use_cache = True
        self.index = None
        if (root / "data/processed/dataset_manifest.json").exists():
            # Verify source corpus before using it as a memorization reference.
            from .corpus import validate_dataset
            audit = validate_dataset(root, project, self.tokenizer)
            if audit["dataset_id"] != self.export["descriptor"]["dataset_id"]:
                raise ValueError("Evaluation corpus differs from the adapter's training corpus")
            self.index = MemorizationIndex(root)

    def _generate(self, prompt, settings):
        import torch
        from transformers import set_seed
        set_seed(settings.seed)
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        if batch.input_ids.shape[1] > 1024:
            raise ValueError("Keep prompts <=1024 tokens on the laptop configuration")
        params = {"max_new_tokens": settings.max_new_tokens, "do_sample": settings.temperature > 0,
                  "repetition_penalty": settings.repetition_penalty, "pad_token_id": self.tokenizer.pad_token_id,
                  "eos_token_id": self.tokenizer.eos_token_id, "use_cache": True}
        if settings.temperature > 0:
            params.update(temperature=settings.temperature, top_p=settings.top_p, top_k=settings.top_k)
        with torch.inference_mode():
            ids = self.model.generate(**batch, **params)[0, batch.input_ids.shape[1]:].tolist()
        if ids and ids[-1] == self.tokenizer.eos_token_id:
            ids = ids[:-1]
        return {"text": self.tokenizer.decode(ids, skip_special_tokens=True), "generated_tokens": len(ids),
                "repeated_4gram_fraction": repeated_ngram_fraction(ids),
                "memorization": self.index.check(ids, self.tokenizer) if self.index else {"status": "not_checked", "reason": "Training corpus unavailable"}}

    def compare(self, prompt: str, settings=None):
        from filelock import FileLock
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a nonempty Bengali prompt")
        settings = settings or GenerationSettings()
        settings.validate()
        with self.model.disable_adapter():
            original = self._generate(prompt, settings)
        adapted = self._generate(prompt, settings)
        record = {"comparison_id": uuid.uuid4().hex, "created_at": now(), "prompt": prompt,
                  "settings": asdict(settings), "model_id": self.project["model_id"], "revision": self.project["revision"],
                  "adapter_path": str(self.adapter_path), "device": self.device,
                  "base": original, "fine_tuned": adapted}
        path = self.root / "evaluations/generations.jsonl"
        with FileLock(str(path) + ".lock"):
            rows = list(read_jsonl(path)) if path.exists() else []
            write_jsonl(path, rows + [record])
        return record

    def heldout_losses(self):
        from datasets import Dataset
        from .training import CausalCollator, target_weighted_loss
        rows = Dataset.from_list([{"input_ids": r["input_ids"]} for r in read_jsonl(self.root / "data/processed/validation.jsonl")])
        collator = CausalCollator(self.tokenizer.pad_token_id)
        with self.model.disable_adapter():
            base = target_weighted_loss(self.model, rows, collator)
        adapted = target_weighted_loss(self.model, rows, collator)
        result = {"created_at": now(), "base": base, "fine_tuned": adapted,
                  "adapter_path": str(self.adapter_path), "dataset_id": self.export["descriptor"]["dataset_id"]}
        write_json(self.root / "evaluations/heldout_losses.json", result)
        return result

    def close(self):
        if self.index:
            self.index.close()
        del self.model
        from .modeling import release_model
        release_model()


def save_rating(root: Path, comparison_id: str, model: str, ratings: dict, notes="", reviewer="local"):
    from filelock import FileLock
    path = root / "evaluations/generations.jsonl"
    if not path.exists() or not any(r["comparison_id"] == comparison_id for r in read_jsonl(path)):
        raise ValueError("Unknown saved comparison_id")
    if model not in {"base", "fine_tuned"}:
        raise ValueError("Rate base or fine_tuned")
    if set(ratings) != set(RATING_FIELDS) or any(type(v) is not int or not 1 <= v <= 5 for v in ratings.values()):
        raise ValueError("Every rating must be an integer from 1 to 5")
    row = {"comparison_id": comparison_id, "model": model, "created_at": now(),
           "reviewer": reviewer, "notes": notes, **ratings}
    target = root / "evaluations/ratings.jsonl"
    with FileLock(str(target) + ".lock"):
        rows = list(read_jsonl(target)) if target.exists() else []
        write_jsonl(target, rows + [row])
    return row


def display_comparison(record):
    from IPython.display import HTML, display
    from html import escape
    # Escape generated HTML; model output is untrusted text.
    blocks = [f"<h4>{name}</h4><pre style='white-space:pre-wrap;font:18px/1.8 serif'>{escape(record[key]['text'])}</pre>"
              for name, key in [("Original base", "base"), ("Fine-tuned adapter", "fine_tuned")]]
    display(HTML("".join(blocks)))


def evaluation_widget(session):
    import ipywidgets as w
    from IPython.display import display
    prompt = w.Textarea(value=PROMPTS["Nature imagery"], description="Prompt", layout=w.Layout(width="95%", height="100px"))
    preset = w.Dropdown(options=list(PROMPTS), description="Example")
    preset.observe(lambda event: setattr(prompt, "value", PROMPTS[event["new"]]), names="value")
    controls = {
        "max_new_tokens": w.IntSlider(value=160, min=8, max=512, step=8, description="New tokens"),
        "temperature": w.FloatSlider(value=0.85, min=0, max=1.5, step=0.05, description="Temperature"),
        "top_p": w.FloatSlider(value=0.9, min=0.05, max=1, step=0.05, description="Top-p"),
        "top_k": w.IntSlider(value=50, min=0, max=100, description="Top-k"),
        "repetition_penalty": w.FloatSlider(value=1.08, min=0.8, max=1.5, step=0.02, description="Repetition"),
        "seed": w.IntText(value=42, description="Seed")}
    output, status = w.Output(), w.HTML()
    generate = w.Button(description="Compare continuations", button_style="primary")
    rating_model = w.Dropdown(options=["base", "fine_tuned"], description="Rate output")
    ratings = {key: w.IntSlider(value=3, min=1, max=5, description=key.replace("_", " "),
                                style={"description_width": "180px"}, layout=w.Layout(width="450px")) for key in RATING_FIELDS}
    notes = w.Textarea(description="Notes")
    save = w.Button(description="Save ratings")
    current = {}

    def run(_):
        generate.disabled = True
        with output:
            output.clear_output(wait=True)
            try:
                record = session.compare(prompt.value, GenerationSettings(**{k: c.value for k, c in controls.items()}))
                current["record"] = record
                display_comparison(record)
                print("Saved comparison:", record["comparison_id"])
                print("Memorization review:", {k: record[k]["memorization"] for k in ("base", "fine_tuned")})
            except Exception as e:
                print(type(e).__name__ + ": " + str(e))
            finally:
                generate.disabled = False

    def rate(_):
        if "record" not in current:
            status.value = "Generate a comparison first."
            return
        try:
            save_rating(session.root, current["record"]["comparison_id"], rating_model.value,
                        {k: c.value for k, c in ratings.items()}, notes.value)
            status.value = "Ratings saved. For unwanted repetition, 5 means severe repetition; other criteria use 5 = best."
        except Exception as e:
            from html import escape
            status.value = escape(str(e))

    generate.on_click(run)
    save.on_click(rate)
    display(w.VBox([preset, prompt, *controls.values(), generate, output, rating_model, *ratings.values(), notes, save, status]))
