# Tagore Bengali LLM · Qwen3.5-0.8B

A notebook project for collecting authentic Bengali works by Rabindranath Tagore, preparing **at most 2,000,000 exact tokenizer tokens**, and running recoverable LoRA continued pretraining on `Qwen/Qwen3.5-0.8B-Base`.

**Repository only:** no corpus, model weights, trained adapter, or generated evaluation results are included. Notebook actions that download, prepare data, or train are disabled by default. No training was run while creating this repository. Dependencies are pinned for a reproducible candidate environment; the complete ML stack has **not been installed or validated together**. See [validation status](docs/VALIDATION.md) before treating this as a tested training configuration.

The original request specifies an RTX 3050 Laptop with 6 GB VRAM and 16 GB RAM. Read-only inspection of this development host instead reported an **RTX 2050, 4 GB VRAM, 45 W reported power limit**, with an NVIDIA 595.99.02 driver. These observations do not establish the target laptop's specifications. No VRAM-fit or throughput claim is made for either GPU.

## Files

```text
.
├── notebooks/
│   ├── 01_collect_and_prepare_tagore_dataset.ipynb
│   ├── 02_finetune_qwen35_tagore.ipynb
│   └── 03_test_tagore_model.ipynb
├── src/tagore_llm/       # Collection, QC, token budgeting, training, checkpoints, evaluation
├── scripts/             # Offline validation, opt-in model check and real recovery demo
├── tests/               # Standard-library offline tests with clearly synthetic fixtures
├── configs/
│   ├── project.json
│   ├── training.json
│   └── manual_import.example.jsonl
├── docs/                # Source research, compatibility decisions, validation status
├── data/{raw,cleaned,processed,metadata}/
├── checkpoints/         # Persistent full-state saves; configurable location
├── models/              # Exported adapters and provenance
├── logs/                # Training metrics, hardware/probe reports and recovery scans
├── evaluations/         # Generated comparisons, ratings, loss reports and overlap index
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Install later on Linux or WSL2

These commands are provided for your later use. They have not been executed to completion as part of delivering the repository.

1. Use a 64-bit Ubuntu environment and **Python 3.11 or 3.12**. Python 3.14 is deliberately outside the supported project range.
2. Install an NVIDIA driver compatible with the chosen PyTorch CUDA wheel. In WSL2, install the NVIDIA Windows host driver and verify GPU passthrough; do not install a second Linux GPU driver inside WSL. Keep the project/checkpoints on the WSL Linux filesystem for better small-file performance, backed by a persistent SSD.
3. Check the driver and real GPU:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version --format=csv
```

The `CUDA Version` in `nvidia-smi` is the driver's supported runtime level, not proof that a Python CUDA-enabled wheel is installed. A separate system CUDA toolkit is not required for the prebuilt baseline. Optional compiled kernels would add toolkit/compiler requirements.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 12.8 wheel candidate. Confirm support for your driver against official PyTorch instructions.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
python -m ipykernel install --user --name tagore-llm --display-name "Python (tagore-llm)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m tagore_llm.cli environment
jupyter lab notebooks/
```

If `venv` is missing on Ubuntu, install the matching `python3.12-venv` system package. Select **Python (tagore-llm)** in Jupyter. No CUDA extensions, FlashAttention, `mamba-ssm`, `causal-conv1d`, or Unsloth are required by the baseline. Qwen's fallback DeltaNet implementation can be slower and use more memory; test it with the real memory probe. Do not install a newer framework into an active resumable experiment without creating a new run.

After verifying your installation, capture exact transitive versions:

```bash
python -m pip freeze > logs/environment-lock.txt
```

The top-level pins prevent automatic major-version changes; they are not represented as a tested transitive lock. Transformers v5 is required for Qwen3.5. The exact model commit is pinned in `configs/project.json`; no `trust_remote_code` or automatically executed remote kernels are used.

## Notebook 01 · Collection and dataset preparation

Open [01_collect_and_prepare_tagore_dataset.ipynb](notebooks/01_collect_and_prepare_tagore_dataset.ipynb). Enable its operation flags only when you want to perform those actions.

### Sources and permissions

The default collector discovers works through Bengali Wikisource's Tagore author catalog, then follows mainspace work pages and bounded subpages. It saves source bytes, URLs, response metadata, checksums, discovery evidence, extraction decisions, and rights evidence. It checks robots live, uses a real identifying contact, runs serially at a minimum three-second delay, respects 429/503 backoff, and stops on forbidden paths or unchecked redirects.

The researched robots rules block `/w/` and `/api/` for this collector. Thus, despite preferring APIs, this implementation uses permitted article HTML. It does not override robots to use the Action API. Containers and tables of contents are retained raw for auditing and excluded from training; child works are collected separately. The default 200-page/depth-2 cap is a cautious starting point, not a promise of a complete collection. Increase it deliberately after reviewing source policy and results.

For automatic acceptance, a literary page needs explicit authorship evidence, a public-domain notice and transcription-license evidence, as well as acceptable text quality. Missing evidence enters review. The underlying work's public-domain status and a site's transcription license are distinct. In India, lifetime-published Tagore works are generally beyond the life-plus-60-year term; this does not establish rights in every later edition, posthumous work, editorial addition, scan or jurisdiction. Relevant primary law, page licenses and source permissions are linked in [source research](docs/SOURCES_AND_COMPATIBILITY.md).

SNLTR's Rabindra Rachanabali FAQ permits page-by-page downloading and copying, but a current site footer also asserts rights. Bichitra/Jadavpur University is an authoritative reference for editions and variants, with no verified bulk reuse authorization in this research. Use **authorized manual import** for these and other archives. An accessible page alone is not an import authorization. Modern translations, editorial commentary, introductions by others, and unverified OCR are excluded.

Set your own `collection.contact` in `configs/project.json` before collecting:

```bash
tagore collect
```

No placeholder contact is supplied to a server. Interrupted collection reuses cached responses only when their checksums match. Delete a specific cache entry only if you intentionally want to refetch it; do not edit raw source bytes.

### Authorized manual import

Copy `configs/manual_import.example.jsonl` to `configs/manual_import.local.jsonl`, fill it with real metadata, and point `path` at a UTF-8 text file you are authorized to use. Fill `source_url`, `rights_status`, `rights_evidence_url`, title, genre, and canonical `work_id`. Set the two authorization/review flags to true only after checking the material. Keep the same work ID for all chapters/sections and editions of one work.

```bash
tagore import-manual configs/manual_import.local.jsonl
```

Original file bytes are copied intact into `data/raw/manual/`. Invalid UTF-8 is rejected rather than silently replaced. The example manifest is intentionally unusable until populated; it contains no literary material.

### Cleaning, review, deduplication and diversity

The cleaner keeps historical spelling, punctuation, Bengali code points, joiners, line breaks, dialogue, paragraph/stanza boundaries, and meaningful leading indentation. It applies no NFC/NFKC, modern spelling or language-model rewriting. It logs initial BOM removal, CR/CRLF-to-LF, HTML NBSP-to-space, trailing horizontal-whitespace removal, outer-blank-line trimming, and excessive-blank-line collapse. DOM extraction logs removal of identified navigation, page labels, references and license/UI elements. Inspect manual queues against raw sources; selectors do not establish literary authenticity by themselves.

Language ratio, minimum length, Unicode corruption, mojibake, suspicious OCR patterns, page-number-like lines, front matter and tables produce scored decisions. Suspected OCR text is reviewed, never automatically guessed. Legitimate short songs can be considered by adjusting the documented minimum length. Each retained work has title, genre, source, author/language, edition/publication date when known, rights status, content checksum, cleaning status, quality score and unique IDs. Unknown bibliographic fields remain null.

Exact duplicates and near-identical/contained editions are removed with an audit trail. For remaining partial overlaps, a conservative 16-word contiguous-passage index connects works into one split. This preserves legitimate literary refrains and quotations instead of deleting them from otherwise distinct works. `duplicate_passage_groups.jsonl` documents those links. Metadata, source boilerplate and machine IDs do not enter literary `text` passages.

Genre preferences use weighted token-mass selection among poems, songs, stories, novels, plays, essays, criticism and other prose/letters. They are soft preferences; no category is invented or duplicated to fill a quota. Genre discovery is a heading heuristic; verify uncertain categories manually.

### Exact token ceiling and non-overlapping groups

```bash
python scripts/verify_model.py --download-config-and-tokenizer
tagore prepare
tagore audit
```

These explicit commands may download the pinned tokenizer/config; `prepare` does not load model weights. Training rows contain both clean `text` and exact `input_ids`. One EOS is appended to each independently tokenized sequence; the dataset budget counts **these actual arrays**, including EOS, across both splits. No chat template or hidden BOS is added. Padding is created later, excluded from attention and loss, and not counted as literary data. No source text is repeated as overlapping context.

Packing combines adjacent paragraphs/stanzas **within the same document** up to the configured sequence length. Poems/stories that fit remain whole. Overlong works are divided only at these natural boundaries. An indivisible long paragraph/stanza is queued rather than cut mid-sentence; obtain verified natural sections or deliberately revise the data configuration and rebuild. Selection includes whole prepared documents only. A document that does not fit the remaining budget is logged, and the pipeline continues to consider smaller works.

Entire canonical works, related editions and transitive shared-passage groups are assigned together. A token-aware greedy split targets 90%/10%; actual percentages may differ for large works. Fewer than two independent groups causes a clear failure. The final audit re-tokenizes every sequence, verifies hashes/counts and checks work/document/group/passage leakage. This is a conservative exact/near-overlap control, not a claim to detect every paraphrase.

Successful preparation writes:

| Artifact in `data/processed/` | Purpose |
| --- | --- |
| `train.jsonl`, `validation.jsonl` | Literary text and exact sequence IDs/token arrays, separate metadata fields |
| `metadata.jsonl` | Selected work provenance, scores, token totals and split groups |
| `dataset_manifest.json` | Model/tokenizer identity, configuration, dataset ID and file hashes |
| `quality_report.json` | Collected/retained/rejected/review/dedup counts and character totals |
| `rejected_documents.jsonl` | Per-document exclusions and duplicate reasons |
| `token_statistics.json` | Exact split totals, EOS overhead, loss tokens and genre distribution |
| `corpus.txt` | Readable clean corpus backup with paragraph/stanza structure |

`data/metadata/` holds source discovery, collection errors, review queues, passage groups and budget exclusions. If too little text passes QC, reports are still written before the safe-split failure. Do not pad, fabricate or duplicate works to reach 2M tokens. Rerunning preparation is deterministic for the same frozen inputs/config/tokenizer; a new manifest is written last so mixed generations fail integrity checks.

Review decisions are explicit JSONL, as shown in notebook 01. Approval must name a reviewer, match the exact cleaned checksum, affirm original-text review and include rights evidence. Corrupted Unicode/mojibake must be corrected from an authorized source rather than approved away. Regenerate the processed dataset after review; preserve raw files.

## Notebook 02 · LoRA training and memory checks

Open [02_finetune_qwen35_tagore.ipynb](notebooks/02_finetune_qwen35_tagore.ipynb), or later run:

```bash
tagore train
```

This command **does start training** when dependencies and a validated dataset are present. It is never called by repository validation or notebook defaults.

The loader checks the pinned model config and safetensors key index, extracts the text backbone from the multimodal repository, verifies loaded text weights, and checks all configured LoRA targets are actual Linear layers. It freezes base parameters. The corpus remains CPU-side in Arrow buffers, with batch transfers to CUDA; it never places the full corpus on the GPU.

The baseline uses PyTorch + Transformers Trainer + PEFT for explicit causal labels and full-state recovery. TRL is pinned and reported in the environment, but no additional SFT formatting/packing is introduced. Unsloth's official guide confirms Qwen3.5 training support; its patched backend is deliberately not combined with the baseline. The guide currently discourages 4-bit QLoRA on Qwen3.5. **An 8-bit AdamW optimizer is not 4-bit model quantization.** See the compatibility notes before experimenting with another backend.

Default settings are in `configs/training.json`:

| Setting | Default |
| --- | --- |
| Objective / epochs | Causal next-token prediction / 1 |
| Sequence / micro-batch / accumulation | 512 / 1 / 8 |
| Effective batch | 8 sequences, variable number of non-padding tokens |
| LoRA | rank 8, alpha 16, dropout 0, attention/DeltaNet/MLP projection targets |
| Learning rate / schedule / warmup | 2e-5 / cosine / 0.03 |
| Weight decay / clipping | 0.01 / 1.0 |
| Optimizer | 8-bit AdamW |
| Precision | BF16 if supported, otherwise FP16 with framework scaler |
| Gradient checkpointing | Enabled, non-reentrant |
| Save / retain / evaluate | Every 25 optimizer steps / 3 complete saves / every 50 steps |
| Seed / dataloader workers | 42 / 0 |

Before training, the implementation checks CUDA, available RAM/disk, actual model identity, data hashes and exact token totals. It performs a **real longest-sequence forward/backward probe**, checks finite loss/adapter gradients, and records measured peak memory plus estimated optimizer reserve in `logs/<run>/memory_probe.json`. The probe does not update weights. No configuration is claimed to fit 6 GB without running this and the subsequent optimizer steps. The reference DeltaNet path and the model's large vocabulary make measurement necessary.

### Monitoring and overfitting

`logs/<run>/metrics.jsonl` records loss, learning rate, gradient norm when reported by Trainer, optimizer step, epoch, elapsed session time, actual non-padding training tokens/s, memory and measured ETA. ETA is based on completed local optimizer steps; it is not a benchmark. Use:

```bash
tail -f logs/tagore-qwen35-v1/metrics.jsonl
```

Training and validation losses are compared; a wide gap and three rising validation evaluations emit warnings. Optional patience-based early stopping can be enabled. Final held-out evaluation runs even if a short run does not reach the regular evaluation interval. A separate held-out NLL/perplexity calculation weights by actual causal target counts, so short sequences do not receive the same weight as long ones. It uses only loss, avoiding storage of the huge vocabulary logits across validation.

Start with one epoch. To plan a second, inspect held-out fit, human ratings and verbatim-overlap flags, then choose a new experiment configuration. Increasing epochs changes the scheduler/run fingerprint and cannot silently resume an old optimizer trajectory. More epochs are not assumed to improve style.

### Full-state automatic checkpoints

`checkpoints/<run_name>/checkpoint-<optimizer_step>/` includes adapter weights, optimizer, scheduler, trainer state, RNG, training arguments, run fingerprint and scaler when used. `save_only_model=False` is explicit. After framework saving, required files are checked, fsynced and sealed with `COMPLETE.json` containing hashes and sizes. A save without a valid seal is ineligible for recovery. Rotation happens **after sealing** and retains three complete checkpoints; incomplete folders remain for diagnosis.

The run fingerprint includes model/revision, actual tokenizer, dataset ID, training hyperparameters, precision, key package versions and helper-source hashes. Incompatible runs raise an error. There is a process lock to prevent concurrent writers to one experiment. Checkpoint state files are local trusted framework pickles; do not resume arbitrary third-party checkpoint directories.

To save elsewhere, set an absolute persistent SSD directory in `checkpoint_root`. Logs remain under the project. Keep checkpoints on durable storage rather than `/tmp`, a temporary cloud volume or a volatile container layer. A crash can lose work since the latest completed checkpoint. Filesystem/SSD failure can still destroy data; hashes detect corruption, they do not replace backups.

### Resume after interruption

Activate the same environment, keep the same source/config/dataset, and relaunch:

```bash
source .venv/bin/activate
tagore checkpoints checkpoints/tagore-qwen35-v1
tagore train
```

The scan verifies candidate files and hashes, skips incomplete/corrupt folders, and passes the newest compatible save to `Trainer.train(resume_from_checkpoint=...)`. Earlier valid saves are used if necessary. With no valid saves, a new run starts from zero and the scan report explains why; a conflicting experiment cannot be silently adopted. To deliberately start separately, set `force_fresh=true` or use a new `run_name`. Fresh mode creates a unique directory rather than overwriting a previous run.

One Ctrl+C/SIGTERM requests completion of the current optimizer step followed by a full save. A second interrupt stops immediately. SIGKILL, a power loss, or a forced Jupyter restart cannot run cleanup; the latest completed save remains the recovery point. A disk error during saving is reported and leaves prior sealed saves eligible. The final step always gets a full save, including short runs that end between regular intervals.

For terminal persistence, run later inside `tmux` if available:

```bash
tmux new -s tagore
source .venv/bin/activate
tagore train
# Detach with Ctrl+B, then D. Reattach later with: tmux attach -t tagore
```

An adapter export alone cannot restore optimizer/RNG progress. Keep full checkpoints for recovery.

### Real recovery demonstration (not run during delivery)

After preparing authentic data and checking the environment:

```bash
python scripts/recovery_demo.py --execute
```

This uses a unique run, four optimizer steps, a fixed four-step scheduler horizon, and two separate training processes. The first saves at step 2 and exits; the second resumes to step 4. It checks actual optimizer step states, scheduler progress, changed adapter bytes and checkpoint seals, then loads the exported adapter and saves a short base/adapter Bengali-prompt comparison. No literary smoke corpus is invented. Its measured result is written only if the checks actually succeed. Running the script **without** `--execute` performs no training or downloads.

This is a graceful process-restart test. It does not claim to simulate power-loss guarantees or prove bit-for-bit equivalence to uninterrupted CUDA training. Offline tests separately exercise missing files, damaged bytes, wrong experiment fingerprints and corrupt-newest fallback.

### CUDA OOM and optimizer fallback

1. Stop and restart the notebook kernel so failed CUDA allocations/model references are released. Close other GPU applications.
2. If retrying unchanged settings, relaunch and let the checkpoint scan restore the latest complete state.
3. For a smaller configuration, change `sequence_length` to **256 in both** configs, rebuild/audit the dataset, and choose a **new run_name**. Try 128 if necessary. Whole long stanzas/paragraphs may enter review at the smaller length.
4. Keep micro-batch 1, gradient accumulation 8 and dataloader workers 0. Do not increase the corpus's GPU residency.
5. If bitsandbytes cannot load, set `optimizer="adamw_torch"` for a new run and rerun the probe. AdamW uses more optimizer-state memory, although only LoRA parameters are optimized.

Changing sequence length, optimizer, dataset or batch layout is not a transparent continuation of an existing optimizer trajectory. The strict fingerprint rejects it. If you intentionally want to warm-start a new experiment from an adapter, implement/review that distinct workflow rather than editing fingerprints to force a resume. Do not silently switch to 4-bit QLoRA.

## Export and notebook 03 · Inference and evaluation

A successful training invocation exports to:

```text
models/tagore-qwen35-0.8b-lora/<actual-run-name>/step-<global-step>/
```

The export contains the adapter, tokenizer, training arguments and `export.json` with model revision, dataset/run provenance and file hashes. Exports are immutable per run/step. `models/latest_adapter.json` points to the latest export. Base weights remain in the Hugging Face cache; merging a second full model is not required and is not done automatically.

Open [03_test_tagore_model.ipynb](notebooks/03_test_tagore_model.ipynb), set `LOAD_MODEL=True`, and optionally enable comparisons, held-out evaluation and the widget. It supports explicit adapter paths and `DEVICE="cpu"`. CPU FP32 inference uses several GiB of RAM and is expected to be slower; no latency claim is made.

The interface supports Bengali prompts, max new tokens, temperature, top-p, top-k, repetition penalty and seed. It offers seven testing themes: poetry, nature, emotional prose, story continuation, dialogue, philosophical reflection and free-form writing. These prompts are original testing prompts, not quotations attributed to Tagore. Both original-base and adapter-enabled outputs use one backbone, identical prompts/settings and reset seeds. GPU kernels and version changes can still affect reproducibility.

Outputs and ratings are saved locally to `evaluations/generations.jsonl` and `ratings.jsonl`. Rate fluency, grammar, coherence, vocabulary, poetic quality, stylistic resemblance and originality from 1 to 5, with 5 best. **Unwanted repetition uses 1=none and 5=severe.** No human rating is filled in automatically.

The memorization check indexes exact 24-token windows from actual training sequences in SQLite and flags matching generated continuations with document/sequence references and a longest-match excerpt. It checks both base and adapted outputs; a base model may already contain some public-domain literature. It can miss short, paraphrased, or cross-sequence matches. An unflagged result is not proof of originality. Review flagged outputs against source metadata and avoid deliberately prompting reproduction of a known passage.

Held-out NLL/perplexity helps compare predictive fit; it does not measure literary merit or prove factual knowledge. Original Bengali style, continuity and authenticity require human evaluation across multiple prompts and seeds. Finish with `session.close()` when using the interface to release the model's GPU memory.

## Offline verification now; runtime verification later

No installation or network access is required for:

```bash
python3.12 scripts/validate_repository.py
```

This validates Python syntax, notebook JSON structure and code-cell syntax, config consistency, and offline unit tests. If `nbformat` is already installed it also validates the full notebook schema. Synthetic fixture tokenizers test budgeting/boundary invariants; they are **not** evidence of Qwen tokenizer compatibility. Structurally valid checkpoint byte fixtures test seals/fallback; they are **not** trained checkpoints. Nothing in offline validation downloads, trains or generates model output.

Later, the actual tokenizer check and recovery demonstration cover the model-backed path. Package installation, actual model weight loading, LoRA target matching, CUDA forward/backward/optimizer steps, optimizer restoration, generation and measured VRAM/time remain to be verified on your environment. See [the validation matrix](docs/VALIDATION.md) for exact status.

## Disk space and time planning

Use a persistent SSD and plan **roughly 15–25 GB of free space** for the environment, PyTorch/CUDA libraries, download caches, model weights, raw sources, checkpoints and logs. This is a planning allowance, not a measured project size or guaranteed upper bound. An 0.8B-scale text backbone has approximately 1.6 GB of 16-bit parameter storage before architecture-specific extras; the official download also includes vision weights. Framework caches may keep multiple copies/versions. LoRA full-state saves are smaller than full-model saves, but optimizer/scaler/RNG files add overhead beyond the adapter alone.

Training refuses to start with less than the configured 8 GiB free at the checkpoint location, checks a minimum reserve before each save, and reports write failures. Choose a higher threshold for your storage needs. Review raw HTML size and archived/incomplete runs before deleting anything; keep at least one verified full checkpoint and an external backup if recovery matters.

Training time depends on the GPU, thermal/power limits, fallback kernels, sequence utilization, evaluation and checkpoint I/O. No unmeasured hours estimate or tokens/s figure is supplied. At most 2M tokens and length 512 imply at least about 3,907 sequences if nearly full; short literary units increase that count. With accumulation 8, one epoch is roughly 489 optimizer steps in that idealized occupancy case. The actual manifest/Trainer step count and measured early-run ETA are the figures to use.

Large model files, corpora, checkpoints, logs, evaluations, virtual environments and `.env` files are ignored by Git. Do not add credentials to configs. Source licenses and attribution remain separate from literary training passages and must accompany any redistribution as required.
