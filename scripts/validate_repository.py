#!/usr/bin/env python3
"""Offline syntax/structure validation. Uses only the Python standard library."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate():
    report = {"python_files": 0, "notebooks": 0, "code_cells": 0, "json_files": 0, "full_nbformat_schema": False}
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            report["python_files"] += 1
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        book = json.loads(path.read_text(encoding="utf-8"))
        assert book["nbformat"] == 4 and book["nbformat_minor"] >= 5, path
        assert isinstance(book["metadata"], dict) and book["cells"], path
        ids = set()
        for cell in book["cells"]:
            assert cell["id"] not in ids, "Duplicate cell ID"
            ids.add(cell["id"])
            assert cell["cell_type"] in ("code", "markdown"), path
            assert isinstance(cell["metadata"], dict) and isinstance(cell["source"], list), path
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and cell["outputs"] == [], path
                ast.parse("".join(cell["source"]), filename=f"{path}:{cell['id']}")
                report["code_cells"] += 1
        # Use full nbformat schema validation if the dependency is already installed.
        try:
            import nbformat
        except ImportError:
            pass
        else:
            nbformat.validate(nbformat.from_dict(book))
            report["full_nbformat_schema"] = True
        report["notebooks"] += 1
    assert report["notebooks"] == 3, "Expected exactly three delivered notebooks"
    for path in (ROOT / "configs").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
        report["json_files"] += 1
    for path in (ROOT / "configs").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)
    project = json.loads((ROOT / "configs/project.json").read_text())
    train = json.loads((ROOT / "configs/training.json").read_text())
    assert project["sequence_length"] == train["sequence_length"]
    assert project["max_tokens"] <= 2_000_000
    assert len(project["revision"]) == 40
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report.update({"tests_run": result.testsRun, "tests_skipped": len(result.skipped), "tests_passed": result.wasSuccessful(),
                   "model_weight_downloads": "not attempted", "training": "not executed",
                   "real_qwen_tokenizer": "not executed", "gpu_training_tests": "not executed",
                   "default_notebook_execution": "covered by passing offline test" if result.wasSuccessful() else "see test failures"})
    print(json.dumps(report, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    from tagore_llm.common import now, write_json
    result = validate()
    write_json(ROOT / "docs/offline_validation.json", {"checked_at_utc": now(), "python": sys.version, **result})
