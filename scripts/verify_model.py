#!/usr/bin/env python3
"""Explicit opt-in compatibility inspection; may download pinned config/tokenizer, never trains."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-config-and-tokenizer", action="store_true")
    args = parser.parse_args()
    if not args.download_config_and_tokenizer:
        parser.exit(message="No downloads performed. Use --download-config-and-tokenizer for explicit model inspection.\n")
    from tagore_llm.common import load_config, tokenizer_for, tokenizer_fingerprint, write_json
    from tagore_llm.modeling import verify_model
    config = load_config(ROOT)
    _, report = verify_model(config)
    tokenizer = tokenizer_for(config)
    sample = "বাংলা ভাষার সাহিত্য।"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(ids) == sample
    report.update({"tokenizer_fingerprint": tokenizer_fingerprint(tokenizer), "bengali_roundtrip": True,
                   "sample_exact_tokens": len(ids), "pad_token_id": tokenizer.pad_token_id,
                   "tokenizer_eos_token_id": tokenizer.eos_token_id})
    write_json(ROOT / "logs/model_verification.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
