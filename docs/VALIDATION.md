# Validation status

Delivery mode: **repository creation only**. After the user clarified the scope, dependency fetching was stopped. No model weights or literary corpus were collected, no training or inference was started, and no generated outputs/ratings/benchmarks are claimed.

Run `python3.12 scripts/validate_repository.py` for the repeatable offline checks. See the final repository validation report for the exact test count.

| Requirement | Status at delivery | Actual check / later command |
| --- | --- | --- |
| Python syntax | Checked offline | AST parse every helper/script/test |
| Notebook JSON and code-cell syntax | Checked offline | Three nbformat-4.5 JSON documents, unique IDs, empty outputs, AST parse all code cells |
| Notebook top-to-bottom safe defaults | Checked offline | All code cells executed with default flags and network connections blocked; no model loads or training |
| Full nbformat schema | Not run unless nbformat already present | Validator automatically enables schema checks when installed |
| Dependency installation and resolution | Not completed | Later install pinned requirements and run `python -m pip check` |
| Official repository/config/tokenizer identity | Official source inspection only | `python scripts/verify_model.py --download-config-and-tokenizer` for actual local Qwen round-trip/fingerprint |
| Dataset JSONL and tamper detection | Checked with explicit synthetic fixtures | End-to-end local preparation/audit and modified-file failure |
| Actual Qwen exact token totals | Not run | `tagore prepare` and `tagore audit` on authorized text |
| Sequence boundaries / EOS budget | Checked offline with a fixture tokenizer | Whole paragraphs/stanzas, no truncation, exact array length plus EOS, whole-document token ceiling |
| Work/group leakage | Checked offline | Transitive group partition tests; final corpus audit also checks contiguous-passage intersections |
| Bengali preservation / corruption rejection | Checked offline | Historical forms/joiners/stanzas preserved; broken Unicode quarantined |
| Duplicate/near-edition detection | Checked offline | SQLite shingle intersection and containment fixture cases |
| LoRA target matching and full weight loading | Not run | Actual loader checks all requested targets and missing text weights before training |
| CUDA forward/backward and optimizer | Not run | Real longest-sequence probe, then `recovery_demo.py --execute` |
| Checkpoint files, seals, fallback, compatibility | Checked with structural byte fixtures | Missing scaler, truncated weights, corrupt newest optimizer, incompatible experiment and retention tests |
| Real Trainer state saving/restoration | Not run | `python scripts/recovery_demo.py --execute` verifies optimizer steps 2→4 and scheduler progress across processes |
| Inference / Bengali generation | Not run | Demo generation stage or notebook 03 with explicit model load/generation flags |
| Saved evaluation results / human ratings | Implemented, model-backed workflow not run | Demo saves actual comparisons; notebook widget saves ratings only on explicit human input |
| 6 GB fit / throughput / ETA | Not measured | Probe and early runtime logs on the actual target GPU |

Synthetic test documents/tokenizers and structural checkpoint bytes are confined to temporary test directories. They are not attributed to Tagore, shipped as a corpus, substituted for the official tokenizer, or described as successful model training. There are no populated training/evaluation artifacts in the delivered repo.

Model-backed verification remains a required acceptance step before a full training run. The repository's failure checks intentionally stop on missing model modules, architecture changes, stale tokenizers, unsafe source evidence, insufficient independent validation groups, insufficient memory/disk or incompatible checkpoints.
