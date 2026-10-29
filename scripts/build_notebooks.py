#!/usr/bin/env python3
"""Rebuild the three notebooks deterministically with no notebook-authoring dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    text = textwrap.dedent(text).strip() + "\n"
    result = {"cell_type": kind, "id": hashlib.sha256(text.encode()).hexdigest()[:12], "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def md(text):
    return cell("markdown", text)


def code(text):
    return cell("code", text)


BOOTSTRAP = """
from pathlib import Path
import sys, json, logging

here = Path.cwd().resolve()
ROOT = next((p for p in [here, *here.parents] if (p / "src/tagore_llm").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Open this notebook from the repository root or notebooks/ directory.")
sys.path.insert(0, str(ROOT / "src"))
from tagore_llm.common import load_config, read_json, read_jsonl, write_json
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
PROJECT = load_config(ROOT)
print("Project:", ROOT)
print("Pinned model:", PROJECT["model_id"], "revision:", PROJECT["revision"])
"""


def write(name, cells):
    book = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (tagore-llm)", "language": "python", "name": "tagore-llm"},
                                           "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    write("01_collect_and_prepare_tagore_dataset.ipynb", [
        md("""
        # 01 · Authentic Bengali corpus, quality review, and exact token budget

        Build a corpus of **up to 2,000,000 tokens** from Tagore's original Bengali writing.
        This notebook orchestrates the actual implementation in `src/tagore_llm/collection.py`
        and `corpus.py`. Raw downloads and authorized imports remain intact. No artificial
        Q&A, paraphrases, spelling modernization, or invented literature enter the corpus.

        **Repository-only defaults:** all network and preparation actions are off. Read the
        README, install the environment, then explicitly enable the operations you want.
        Running the default notebook does not download anything or start training.
        """),
        code(BOOTSTRAP),
        md("""
        ## Configuration and source policy

        Wikisource's public-domain notices and transcription license are recorded separately.
        Its current robots rules disallow `/w/` and `/api/` for this collector, so the default
        implementation checks robots live and uses permitted article pages, serially, with a
        minimum three-second delay. It stops on policy/redirect uncertainty. Author catalogs
        provide discovery evidence; individual texts still need authorship, rights and QC checks.

        SNLTR's FAQ permits individual-page downloads; its live footer also asserts rights.
        Bichitra/Jadavpur is an authoritative edition reference, but no bulk reuse permission
        was established. These sources use the authorized manual-import workflow instead
        of an unverified crawler. See `docs/SOURCES_AND_COMPATIBILITY.md` for primary sources.
        """),
        code("""
        RUN_COLLECTION = False
        RUN_MANUAL_IMPORT = False
        RUN_PREPARATION = False
        # Set to an actual authorized manifest, never the unfilled example.
        MANUAL_MANIFEST = ROOT / "configs/manual_import.local.jsonl"

        # Edit configs/project.json for persistent settings. This copy can be changed for this session.
        # PROJECT["collection"]["contact"] = "your actual contact email or project URL"
        assert 1 <= PROJECT["max_tokens"] <= 2_000_000
        assert PROJECT["sequence_length"] >= 8
        print(json.dumps(PROJECT, indent=2, ensure_ascii=False))
        """),
        md("""
        ## Resumable collection and authorized imports

        Collection saves each response and SHA-256 checksum before extracting text. Repeated
        runs reuse verified responses and document IDs; they do not append duplicate copies.
        A real contact value is required in the user agent. 429/503 responses trigger backoff;
        authentication, paywalls, denied paths and anti-bot controls are never bypassed.

        For manual imports, fill every metadata field, set `authorized=true` and
        `reviewed_original_text=true` only after checking the actual material and permission.
        The example's false flags deliberately prevent accidental import.
        """),
        code("""
        if RUN_COLLECTION:
            from tagore_llm.collection import collect
            print(collect(ROOT, PROJECT))
        else:
            print("Collection disabled; no network requests.")

        if RUN_MANUAL_IMPORT:
            from tagore_llm.collection import import_manual
            print("Imported document IDs:", import_manual(ROOT, MANUAL_MANIFEST))
        else:
            print("Manual import disabled.")
        """),
        md("""
        ## Quality control and the manual review queue

        Cleaning preserves Bengali characters, historical wording, joiners, punctuation,
        paragraph boundaries, line breaks and stanza boundaries. It does not apply NFC/NFKC.
        Only the initial BOM, newline conventions, HTML NBSP, trailing layout whitespace and
        excessive blank lines are normalized; each change is logged. Known site UI is removed
        structurally. Suspected OCR, corrupted text, tables, front matter and uncertain rights
        enter review, rather than speculative correction. Containers/TOCs are excluded.

        Exact and near duplicate editions are removed. Remaining works that share a contiguous
        16-word passage are connected before splitting, preserving legitimate refrains/quotations.
        SQLite indexes keep duplicate fingerprints off the GPU and out of a giant Python map.

        A paragraph or stanza that exceeds the configured sequence length is queued for review.
        The pipeline does not truncate it. Import verified natural sections if needed, with the
        **same canonical work_id** for all sections and editions. Do not approve text simply
        to fill the target budget. Entire selected documents are retained or excluded by budget.
        """),
        code("""
        if RUN_PREPARATION:
            from tagore_llm.common import tokenizer_for
            from tagore_llm.corpus import prepare
            # This explicit action downloads the pinned TOKENIZER if it is not already cached.
            tokenizer = tokenizer_for(PROJECT)
            preparation = prepare(ROOT, PROJECT, tokenizer)
            print(json.dumps(preparation["quality"], indent=2, ensure_ascii=False))
            print(json.dumps(preparation["statistics"], indent=2, ensure_ascii=False))
        else:
            print("Preparation disabled; tokenizer is not downloaded.")
        """),
        code("""
        queue_path = ROOT / "data/metadata/manual_review_queue.jsonl"
        rejected_path = ROOT / "data/processed/rejected_documents.jsonl"
        if queue_path.exists():
            review_rows = list(read_jsonl(queue_path))
            print("Review queue:", len(review_rows))
            for row in review_rows[:10]:
                print(row["doc_id"], row.get("title"), row["reasons"], row["content_checksum"])
        else:
            print("No review queue yet.")
        if rejected_path.exists():
            from collections import Counter
            print("Rejection reasons:", Counter(reason for row in read_jsonl(rejected_path) for reason in row.get("reasons", [])))
        """),
        md("""
        To review a flagged work, compare `data/cleaned/<doc_id>.json` with the original source
        and its saved HTML. Put one decision per document in `data/metadata/review_decisions.jsonl`:

        ```json
        {"doc_id":"actual ID", "decision":"approve", "content_checksum":"actual cleaned checksum", "reviewer":"your name", "reviewed_original_text":true, "rights_evidence_url":"actual permission or license URL"}
        ```

        Use `decision="reject"` for exclusions. Corrupted Unicode or mojibake cannot be approved
        away: obtain a corrected authorized source and import it with provenance. Re-run
        preparation after review. Never edit frozen processed files to make an audit pass.
        """),
        md("""
        ## Budget and leakage audit

        The sequence constructor tokenizes each actual training passage with the pinned Qwen
        tokenizer, appends one EOS, and counts that exact array. No BOS, chat template or prompt
        wrapper is added. Padding positions are masked later and are reported separately from
        literary data. Independent work groups target a 90/10 token split; large works may
        make the achieved fraction differ. Fewer authentic tokens are a valid result.
        """),
        code("""
        if RUN_PREPARATION:
            from tagore_llm.corpus import validate_dataset
            audit = validate_dataset(ROOT, PROJECT, tokenizer)
            print(json.dumps(audit, indent=2))
            assert audit["total_tokens"] <= 2_000_000
        stats_path = ROOT / "data/processed/token_statistics.json"
        if stats_path.exists():
            print(json.dumps(read_json(stats_path), indent=2, ensure_ascii=False))
        else:
            print("No dataset has been prepared. No corpus or statistics are fabricated.")
        """),
        md("""
        ## Output contract

        A successful preparation produces `train.jsonl`, `validation.jsonl`, `metadata.jsonl`,
        `dataset_manifest.json`, `quality_report.json`, `rejected_documents.jsonl`,
        `token_statistics.json`, and `corpus.txt` under `data/processed/`. Review queues,
        deduplication groups, source records and budget exclusions remain under `data/metadata/`.
        No URLs, licenses, IDs or technical metadata are inserted into the `text` passages.
        Continue to notebook 02 only after auditing these artifacts.
        """),
    ])
    write("02_finetune_qwen35_tagore.ipynb", [
        md("""
        # 02 · Recoverable Qwen3.5 Bengali LoRA continued pretraining

        The objective is next-token prediction on authentic literary passages. The official
        `Qwen/Qwen3.5-0.8B-Base` checkpoint includes a vision tower; the loader extracts and
        verifies the text backbone using `Qwen3_5ForCausalLM`, never silently initializes missing
        language weights, freezes base parameters, and checks every requested LoRA target.

        **All training and model downloads are disabled by default.** This repository was
        requested without executing training. The memory probe, full-model passes and recovery
        demonstration below are real implementations, with no fabricated results.
        """),
        code(BOOTSTRAP),
        code("""
        TRAINING = load_config(ROOT, "training")
        CHECK_MODEL_AND_TOKENIZER = False
        RUN_DATASET_AUDIT = False
        RUN_TRAINING = False
        RUN_RECOVERY_DEMO = False
        print(json.dumps(TRAINING, indent=2))
        """),
        md("""
        ## Hardware and dependencies

        Use Python 3.11 or 3.12 on Linux/WSL2. Install the CUDA-enabled PyTorch wheel before
        `requirements.txt`. `nvidia-smi` reports driver-supported CUDA; `torch.version.cuda`
        reports the installed wheel's runtime. Verify the actual laptop's power limit instead
        of assuming 95 W. GPU availability alone does not prove a 512-token training run fits.
        """),
        code("""
        from tagore_llm.modeling import environment_report
        try:
            hardware = environment_report(ROOT)
            print(json.dumps(hardware, indent=2))
        except ModuleNotFoundError as exc:
            print("Dependencies are not installed:", exc)
            print("Follow README installation instructions before enabling training.")
        """),
        md("""
        ## Conservative configuration

        Start with one epoch, length 512, batch 1, accumulation 8, rank 8/alpha 16/dropout 0,
        learning rate 2e-5, cosine schedule, warmup 0.03, weight decay 0.01 and clipping 1.0.
        BF16 is selected only if PyTorch confirms support; otherwise FP16 with framework scaling.
        Gradient checkpointing uses the non-reentrant implementation; cache is disabled for training.

        The baseline uses Transformers Trainer plus PEFT, with the pinned TRL package available
        for controlled experiments. Direct pretokenized causal labels avoid additional SFT
        formatting or packing transformations. Unsloth officially supports Qwen3.5, but is not
        mixed into this baseline: its patched stack needs a separate verified environment.
        Unsloth currently discourages 4-bit QLoRA for Qwen3.5. The 8-bit AdamW **optimizer**
        is separate from quantizing base weights.
        """),
        code("""
        if CHECK_MODEL_AND_TOKENIZER:
            from tagore_llm.modeling import verify_model
            from tagore_llm.common import tokenizer_for, tokenizer_fingerprint
            _, model_report = verify_model(PROJECT)
            tokenizer = tokenizer_for(PROJECT)
            print(json.dumps(model_report, indent=2))
            print("Tokenizer fingerprint:", tokenizer_fingerprint(tokenizer))
        else:
            print("Model/tokenizer download and inspection disabled.")

        if RUN_DATASET_AUDIT:
            from tagore_llm.common import tokenizer_for
            from tagore_llm.corpus import validate_dataset
            tokenizer = tokenizer_for(PROJECT)
            print(validate_dataset(ROOT, PROJECT, tokenizer))
        """),
        md("""
        ## Full-state checkpoint recovery

        Every 25 optimizer steps, Trainer saves adapter weights, optimizer, scheduler, global
        step, epoch, trainer configuration and RNG states (plus scaler state when used). After
        file verification and fsync, this project writes `COMPLETE.json` with SHA-256 checksums.
        Rotation keeps three **complete** saves. A partial newest folder cannot displace a
        recoverable earlier save. Old incomplete folders remain visible for diagnosis.

        Relaunch this same notebook or CLI with the same configs and run_name: the latest
        complete compatible checkpoint is selected and passed to `Trainer.train` through
        `resume_from_checkpoint`. Incompatible model/tokenizer/dataset/config/code/package
        fingerprints raise an error. `force_fresh=true` creates a new run directory.

        Save to a persistent SSD; `checkpoint_root` accepts an absolute path. Unsaved steps
        since the latest successful save can be lost after power failure or a hard crash.
        One SIGINT/SIGTERM requests a save after the current optimizer step. A second interrupt,
        SIGKILL, closing an unprotected terminal, or disk failure can interrupt the current save.
        """),
        code("""
        from tagore_llm.checkpoints import checkpoint_inventory
        checkpoint_root = Path(TRAINING["checkpoint_root"]).expanduser()
        if not checkpoint_root.is_absolute():
            checkpoint_root = ROOT / checkpoint_root
        run_dir = checkpoint_root / TRAINING["run_name"]
        if (run_dir / "RUN.json").exists():
            inventory = checkpoint_inventory(run_dir, read_json(run_dir / "RUN.json"))
            print(json.dumps(inventory, indent=2))
        else:
            print("No existing training run at", run_dir)
        """),
        md("""
        ## Train with a measured forward/backward memory probe

        Enabling the cell below loads the real model, rechecks data hashes/token counts/leakage,
        applies LoRA, and probes the longest prepared sequence before training. The probe
        reports measured peak allocated/reserved memory and estimates optimizer headroom.
        It is a gate, not a guarantee for all subsequent steps. CPU Arrow data stays off CUDA.

        Loss, learning rate, gradient norm, actual non-padding training tokens/s, elapsed session
        time, epoch, optimizer step, GPU memory and measured ETA are logged in `logs/<run>/`.
        Held-out NLL is also evaluated with exact target-token weighting. Validation-gap and
        rising-loss warnings inform the decision to stop; a second epoch requires a new run.
        Optional early stopping is configured by `early_stopping_patience` (>0).
        """),
        code("""
        if RUN_TRAINING:
            from tagore_llm.training import train
            result = train(ROOT, PROJECT, TRAINING)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Training disabled. No weights loaded and no optimizer steps executed.")
        """),
        md("""
        ## Real stop / process exit / resume demonstration (explicit opt-in)

        This runs **four real optimizer steps** on your prepared corpus in two separate Python
        processes, with one fixed four-step scheduler horizon. The first stops at step 2 and
        saves; the second restores full state and reaches step 4. It verifies adapter changes,
        optimizer step counts, scheduler progression and checkpoint seals, then loads the
        exported adapter and saves a short Bengali-prompt base/adapter comparison. It does not
        create fake literary examples or infer success from folder names alone.

        The demonstration uses a unique experiment name. It is not the main training run.
        **It has not been executed as part of creating this repository.**
        """),
        code("""
        if RUN_RECOVERY_DEMO:
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "scripts/recovery_demo.py"), "--execute"], cwd=ROOT, check=True)
        else:
            print("Recovery demonstration not executed.")
        """),
        md("""
        ## OOM, disk errors, and the next epoch decision

        On OOM, restart the kernel before loading the model again. For an unchanged configuration,
        resume the latest valid save. To reduce sequence length, set 256 in **both** configs,
        rebuild/audit the dataset, and choose a new run_name; try 128 if still necessary. A changed
        dataset/scheduler/batch layout must not reuse incompatible optimizer state. Keep batch 1
        and dataloader workers 0. If bitsandbytes fails, choose `optimizer="adamw_torch"` in a new
        experiment and rerun the memory probe. Do not assume 4-bit quantization is a safe fix.

        A short run may not reach eval_steps=50; final evaluation and a full checkpoint still run.
        Inspect held-out loss, generation comparisons, originality flags and human ratings before
        planning another epoch. A lower training loss alone is not evidence of better writing.
        See README for terminal persistence, relaunch commands, export paths and disk planning.
        """),
    ])
    write("03_test_tagore_model.ipynb", [
        md("""
        # 03 · Bengali continuations, base-model comparison, and literary evaluation

        Load one text backbone and toggle its LoRA adapter off/on to compare the original and
        adapted model with identical prompts and seeds. This avoids keeping two model copies
        in 6 GB VRAM. GPU inference is preferred; CPU FP32 fallback is implemented and needs
        several GiB of available system RAM. Generation speed is measured locally, not promised.

        Style adaptation does not establish factual literary knowledge or knowledge of every
        Tagore work. The example prompts below were written for testing, not attributed to Tagore.
        Token limits control generation length, not an exact Bengali word count.
        """),
        code(BOOTSTRAP),
        code("""
        LOAD_MODEL = False
        RUN_COMPARISON = False
        RUN_HELDOUT_EVALUATION = False
        SHOW_WIDGET = False
        ADAPTER_PATH = None  # None reads models/latest_adapter.json; otherwise use an explicit export directory.
        DEVICE = None        # None selects CUDA if available, else CPU. Set "cpu" explicitly if desired.
        session = None
        """),
        md("""
        ## Verify adapter provenance and load

        The export carries model/revision identity, run fingerprint and file checksums. Mismatches
        are errors. When the training corpus is present, its hash/token audit runs before building
        a disk-backed exact-token memorization index. The source metadata remains separate
        from the prompts and generated passages.
        """),
        code("""
        from tagore_llm.evaluation import ComparisonSession, GenerationSettings, PROMPTS, display_comparison
        if LOAD_MODEL:
            session = ComparisonSession(ROOT, PROJECT, ADAPTER_PATH, DEVICE)
            print("Device:", session.device)
            print("Adapter:", session.adapter_path)
            print(json.dumps(session.loading_report, indent=2))
        else:
            print("Model loading disabled; no downloads or weight allocation.")
        """),
        md("""
        ## Seven kinds of Bengali literary prompts

        Test several unrelated prompts and more than one seed. A completion can resemble a
        remembered source even when the prompt is novel. Examine both models' outputs,
        source-overlap flags and repeated n-gram fraction, alongside your literary judgment.
        """),
        code("""
        for name, text in PROMPTS.items():
            print(name + ":\\n" + text + "\\n")

        PROMPT = PROMPTS["Nature imagery"]
        SETTINGS = GenerationSettings(max_new_tokens=160, temperature=0.85, top_p=0.9,
                                      top_k=50, repetition_penalty=1.08, seed=42)
        SETTINGS.validate()
        """),
        code("""
        if RUN_COMPARISON:
            if session is None:
                raise RuntimeError("Set LOAD_MODEL=True and run the loading cell first.")
            comparison = session.compare(PROMPT, SETTINGS)
            display_comparison(comparison)
            print("Saved ID:", comparison["comparison_id"])
            for key in ["base", "fine_tuned"]:
                print(key, "generated tokens:", comparison[key]["generated_tokens"],
                      "repeated 4-gram fraction:", comparison[key]["repeated_4gram_fraction"])
                print(json.dumps(comparison[key]["memorization"], indent=2, ensure_ascii=False))
        else:
            print("Generation disabled; no results fabricated.")
        """),
        md("""
        ## Adjustable generation and saved human ratings

        The widget provides prompt, token limit, temperature (0 = greedy), top-p, top-k,
        repetition penalty and seed controls. Both models use identical settings. Seeds make
        comparisons more controlled, but do not guarantee bit-identical results across hardware
        or software versions. Generated text is HTML-escaped before display.

        Rate fluency, grammar, coherence, vocabulary, poetic quality, stylistic resemblance and
        originality from 1 (poor) to 5 (best). **Unwanted repetition uses 1 = none, 5 = severe.**
        Ratings and outputs are saved as JSONL under `evaluations/`; nothing is sent externally.
        """),
        code("""
        if SHOW_WIDGET:
            if session is None:
                raise RuntimeError("Load the model before displaying the evaluation widget.")
            from tagore_llm.evaluation import evaluation_widget
            evaluation_widget(session)
        else:
            print("Widget disabled.")
        """),
        md("""
        ## Held-out NLL and memorization limits

        Evaluate both models on the same held-out literary groups. Target-token-weighted NLL
        and perplexity measure predictive fit, not poetic merit. The exact-overlap index flags
        contiguous matches of at least 24 tokens against stored training sequences and returns
        source document IDs for inspection. It may miss shorter, paraphrased or cross-sequence
        reproduction. An unflagged result is not proof of originality.
        """),
        code("""
        if RUN_HELDOUT_EVALUATION:
            if session is None:
                raise RuntimeError("Load the model first.")
            losses = session.heldout_losses()
            print(json.dumps(losses, indent=2))
        else:
            print("Held-out model evaluation disabled.")
        """),
        code("""
        output_path = ROOT / "evaluations/generations.jsonl"
        ratings_path = ROOT / "evaluations/ratings.jsonl"
        print("Saved generations:", sum(1 for _ in read_jsonl(output_path)) if output_path.exists() else 0)
        print("Saved ratings:", sum(1 for _ in read_jsonl(ratings_path)) if ratings_path.exists() else 0)
        # After finishing interactive evaluation, explicitly release GPU memory:
        # if session is not None:
        #     session.close()
        #     session = None
        """),
    ])


if __name__ == "__main__":
    main()
