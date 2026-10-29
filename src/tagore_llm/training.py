"""Single-GPU continued pretraining, measured memory probe, and full-state automatic recovery."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from importlib.metadata import version
import logging
import math
from pathlib import Path
import shutil
import signal
import time
import uuid

from .common import (canonical_hash, file_hash, load_config, now, read_json, read_jsonl,
                     tokenizer_for, write_json)
from .checkpoints import (IncompatibleRunError, checkpoint_inventory, choose_checkpoint,
                          inspect_checkpoint, rotate_complete, seal_checkpoint)

LOG = logging.getLogger(__name__)


class CausalCollator:
    """Pad by position, not token value: EOS remains a target even when pad_token == eos_token."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        size = max(len(f["input_ids"]) for f in features)
        ids, masks, labels = [], [], []
        for f in features:
            tokens = list(f["input_ids"])
            padding = size - len(tokens)
            ids.append(tokens + [self.pad_token_id] * padding)
            masks.append([1] * len(tokens) + [0] * padding)
            # The model shifts labels internally. Do not pre-shift them.
            labels.append(tokens + [-100] * padding)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def training_descriptor(root, project, train, manifest):
    packages = {p: version(p) for p in ("torch", "transformers", "peft", "accelerate", "trl", "bitsandbytes")}
    important = {k: v for k, v in train.items() if k not in {"force_fresh", "probe_before_training", "checkpoint_root", "min_free_disk_gb"}}
    code = {p.name: file_hash(p) for p in sorted((root / "src/tagore_llm").glob("*.py"))}
    return {"model_id": project["model_id"], "revision": project["revision"],
            "dataset_id": manifest["dataset_id"], "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
            "training": important, "packages": packages, "source_code": code,
            "precision": "bf16" if __import__("torch").cuda.is_bf16_supported() else "fp16"}


def resolve_run(root, train, descriptor):
    root_dir = Path(train["checkpoint_root"]).expanduser()
    if not root_dir.is_absolute():
        root_dir = root / root_dir
    name = train["run_name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("run_name must be one directory name")
    if train.get("force_fresh"):
        name += "-fresh-" + uuid.uuid4().hex[:10]
    run_dir = root_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = run_dir / "RUN.json"
    if descriptor_path.exists() and canonical_hash(read_json(descriptor_path)) != canonical_hash(descriptor):
        raise IncompatibleRunError(f"{run_dir} belongs to a different experiment. Choose a new run_name or force_fresh=true.")
    if not descriptor_path.exists() and any(run_dir.iterdir()):
        raise IncompatibleRunError(f"Refusing to adopt non-empty directory without RUN.json: {run_dir}")
    write_json(descriptor_path, descriptor)
    return run_dir


def memory_probe(model, sample, collator, train):
    """Real forward/backward on longest sequence; does not update weights or create a Trainer.

    A probe cannot prove the entire run fits. Include a conservative optimizer-state estimate,
    then monitor the actual first training steps. FP16 probe uses GradScaler to avoid underflow.
    """
    import torch
    from .modeling import precision
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Training probe requires CUDA")
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = {k: v.to(device) for k, v in collator([sample] * train["micro_batch_size"]).items()}
    dtype = precision()
    try:
        with torch.autocast("cuda", dtype=dtype):
            output = model(**batch)
            loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite probe loss")
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        scaler.scale(loss).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("No finite LoRA gradients in memory probe")
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Two FP32 Adam moments, deliberately conservative even with the 8-bit optimizer.
        optimizer_reserve = trainable * 8
        result = {"measured_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "measured_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "free_after_probe_gib": free / 1024**3, "vram_total_gib": total / 1024**3,
                  "optimizer_reserve_gib": optimizer_reserve / 1024**3,
                  "sequence_length": len(sample["input_ids"]), "loss": float(loss.detach()),
                  "note": "Measured forward/backward; optimizer steps and complete training still require runtime validation."}
        if free < optimizer_reserve + 384 * 1024**2:
            raise RuntimeError(f"Insufficient probe headroom: {result}. Prepare a new dataset at 256 (then 128) and use a new run_name.")
        return result
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError("CUDA OOM during forward/backward probe. Restart the kernel, set sequence_length=256 in BOTH configs, rebuild dataset and choose a new run_name. Try 128 if needed; close other GPU apps. No automatic training retry was performed.") from e
    finally:
        model.zero_grad(set_to_none=True)
        del batch
        torch.cuda.empty_cache()


def trainer_classes():
    """Import heavy dependencies lazily so corpus and checkpoint audits remain usable offline."""
    import torch
    from transformers import Trainer, TrainerCallback

    class RecoverableTrainer(Trainer):
        def __init__(self, *args, descriptor, retention=3, **kwargs):
            self.descriptor = descriptor
            self.retention = retention
            super().__init__(*args, **kwargs)

        def _rotate_checkpoints(self, *args, **kwargs):
            # HF calls rotation inside save, before our completion marker exists.
            # Rotate only after a fully sealed checkpoint.
            return None

        def _save_checkpoint(self, model, trial):
            path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            if inspect_checkpoint(path, self.descriptor)["status"] == "complete":
                return
            free = shutil.disk_usage(self.args.output_dir).free
            if free < 512 * 1024**2:
                raise OSError("Less than 512 MiB free at checkpoint location; free disk space before saving. Earlier sealed checkpoints remain available.")
            # Unlink any stale seal before a retry of the same interrupted step.
            (path / "COMPLETE.json").unlink(missing_ok=True)
            super()._save_checkpoint(model, trial)
            seal_checkpoint(path, self.descriptor, require_scaler=self.accelerator.scaler is not None)
            rotate_complete(Path(self.args.output_dir), self.retention, self.descriptor)

        def evaluate(self, *args, **kwargs):
            # Trainer's averaged batch losses can bias short variable-length sequences.
            # Report a separate exact target-token-weighted held-out NLL without retaining logits.
            metrics = super().evaluate(*args, **kwargs)
            dataset = kwargs.get("eval_dataset") or (args[0] if args else None) or self.eval_dataset
            weighted = target_weighted_loss(self.model, dataset, self.data_collator, self.args.per_device_eval_batch_size, self.compute_loss_context_manager)
            metrics.update(weighted)
            self.log(weighted)
            return metrics

    class Monitor(TrainerCallback):
        def __init__(self, log_path, train, stop_after=None):
            self.log_path, self.train, self.stop_after = log_path, train, stop_after
            self.started = None
            self.start_step = 0
            self.interval_tokens = 0
            self.total_tokens = 0
            self.last_log_time = None
            self.train_losses = deque(maxlen=8)
            self.eval_losses = deque(maxlen=3)
            self.stop_requested = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = self.last_log_time = time.monotonic()
            self.start_step = state.global_step

        def on_step_end(self, args, state, control, **kwargs):
            if self.stop_requested or (self.stop_after is not None and state.global_step >= self.stop_after):
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            import json
            logs = dict(logs or {})
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
            elapsed = time.monotonic() - (self.started or time.monotonic())
            step_delta = state.global_step - self.start_step
            logs.update({"step": state.global_step, "epoch": state.epoch, "session_elapsed_seconds": elapsed,
                         "session_tokens": self.total_tokens, "session_tokens_per_second": self.total_tokens / max(elapsed, 1e-9),
                         "eta_seconds": max(0, state.max_steps - state.global_step) * elapsed / max(1, step_delta),
                         "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                         "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3})
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": now(), **logs}, ensure_ascii=False, allow_nan=False) + "\n")
            LOG.info("Step %s epoch %s loss %s tokens/s %.2f ETA %.0fs", state.global_step, state.epoch,
                     logs.get("loss", logs.get("eval_loss")), logs["session_tokens_per_second"], logs["eta_seconds"])

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            value = (metrics or {}).get("eval_loss")
            if value is not None:
                self.eval_losses.append(value)
                if self.train_losses and value - sum(self.train_losses) / len(self.train_losses) > self.train["overfit_gap"]:
                    LOG.warning("Validation/training loss gap exceeds %.2f. Inspect losses and originality before another epoch.", self.train["overfit_gap"])
                if len(self.eval_losses) == 3 and all(a < b for a, b in zip(self.eval_losses, list(self.eval_losses)[1:])):
                    LOG.warning("Validation loss increased at three evaluations; do not increase epochs without review.")

    return RecoverableTrainer, Monitor


def target_weighted_loss(model, dataset, collator, batch_size=1, context=None):
    import torch
    from contextlib import nullcontext
    total, count = 0.0, 0
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = collator([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
            targets = int((batch["labels"][:, 1:] != -100).sum())
            batch = {k: v.to(device) for k, v in batch.items()}
            with context() if context else nullcontext():
                loss = model(**batch).loss
            total += float(loss) * targets
            count += targets
    model.train(was_training)
    value = total / max(1, count)
    if not math.isfinite(value):
        raise ValueError("Non-finite validation NLL")
    return {"heldout_token_weighted_nll": value, "heldout_perplexity": math.exp(min(value, 80)), "heldout_loss_tokens": count}


@contextmanager
def graceful_signals(monitor):
    """SIGINT/SIGTERM request a full checkpoint at the next completed optimizer step."""
    previous = {}
    def handler(signum, frame):
        if monitor.stop_requested:
            raise KeyboardInterrupt("Second interrupt: stop immediately; recover from the latest sealed checkpoint.")
        LOG.warning("Signal %s: finishing current optimizer step, then saving full state. A second interrupt stops immediately.", signum)
        monitor.stop_requested = True
    try:
        for s in (signal.SIGINT, signal.SIGTERM):
            previous[s] = signal.signal(s, handler)
        yield
    finally:
        for s, old in previous.items():
            signal.signal(s, old)


def build_trainer(root, project, train, stop_after=None):
    import torch
    from datasets import Dataset
    from filelock import FileLock
    from transformers import TrainingArguments, set_seed
    from .corpus import validate_dataset
    from .modeling import apply_lora, environment_report, load_text_model, precision

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU. CPU inference is available in notebook 03.")
    if train["epochs"] not in (1, 2):
        raise ValueError("Start with one epoch. Only set epochs=2 after reviewing validation and originality; larger values are intentionally refused.")
    if train["micro_batch_size"] < 1 or train["gradient_accumulation_steps"] < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    tokenizer = tokenizer_for(project)
    audit = validate_dataset(root, project, tokenizer)
    manifest = read_json(root / "data/processed/dataset_manifest.json")
    if train["sequence_length"] != manifest["sequence_length"]:
        raise ValueError("Training sequence_length must match the prepared dataset. Rebuild it to use a smaller length.")
    descriptor = training_descriptor(root, project, train, manifest)
    # A short initialization lock protects RUN.json; caller holds a full-run lock below.
    checkpoint_root = Path(train["checkpoint_root"]).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = root / checkpoint_root
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(checkpoint_root / ".initialization.lock"), timeout=0):
        run_dir = resolve_run(root, train, descriptor)
    run_lock = FileLock(str(run_dir / ".training.lock"), timeout=0)
    run_lock.acquire()
    try:
        report = environment_report(run_dir)
        if report["disk_free_gib"] < train["min_free_disk_gb"]:
            raise OSError(f"Only {report['disk_free_gib']:.1f} GiB free at checkpoint location; require {train['min_free_disk_gb']} GiB.")
        if report["ram_available_gib"] < 3:
            raise RuntimeError("Less than 3 GiB available RAM. Close applications before loading the model.")
        log_dir = root / "logs" / run_dir.name
        log_dir.mkdir(parents=True, exist_ok=True)
        write_json(log_dir / "environment.json", report)
        resume, inventory = choose_checkpoint(run_dir, descriptor)
        write_json(log_dir / "checkpoint_scan.json", inventory)
        for item in inventory:
            LOG.info("Checkpoint %s: %s (%s)", item["path"], item["status"], item["reason"])
        set_seed(train["seed"])
        data = {}
        for split in ("train", "validation"):
            # A <=2M-token corpus in CPU Arrow buffers. No dataset is moved wholesale to CUDA.
            data[split] = Dataset.from_list([{"input_ids": row["input_ids"]} for row in read_jsonl(root / f"data/processed/{split}.jsonl")])
        collator = CausalCollator(tokenizer.pad_token_id)
        model, loading = load_text_model(project, "cuda", training=True)
        model, lora = apply_lora(model, train, project)
        write_json(log_dir / "model_loading.json", {"model": loading, "lora": lora})
        if train.get("probe_before_training", True):
            longest = max(data["train"], key=lambda x: len(x["input_ids"]))
            write_json(log_dir / "memory_probe.json", memory_probe(model, longest, collator, train))
            set_seed(train["seed"])  # The probe must not perturb training's initial RNG state.
        RecoverableTrainer, Monitor = trainer_classes()
        monitor = Monitor(log_dir / "metrics.jsonl", train, stop_after)
        # Count actual non-padding input tokens that pass through training_step (not an estimated seq_length).
        class CountingTrainer(RecoverableTrainer):
            def training_step(self, model, inputs, num_items_in_batch=None):
                tokens = int(inputs["attention_mask"].sum())
                result = super().training_step(model, inputs, num_items_in_batch)
                monitor.total_tokens += tokens
                return result

        bf16 = precision() == torch.bfloat16
        args = TrainingArguments(
            output_dir=str(run_dir), logging_dir=str(log_dir / "trainer"), run_name=run_dir.name,
            num_train_epochs=train["epochs"], max_steps=train["max_steps"],
            per_device_train_batch_size=train["micro_batch_size"], per_device_eval_batch_size=1,
            gradient_accumulation_steps=train["gradient_accumulation_steps"],
            learning_rate=train["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=train["warmup_ratio"],
            weight_decay=train["weight_decay"], max_grad_norm=train["max_grad_norm"],
            optim=train["optimizer"], bf16=bf16, fp16=not bf16,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="steps", save_steps=train["save_steps"], save_total_limit=None,
            save_only_model=False, save_safetensors=True, eval_strategy="steps", eval_steps=train["eval_steps"],
            logging_strategy="steps", logging_steps=train["logging_steps"], logging_first_step=True,
            prediction_loss_only=True, dataloader_num_workers=0, dataloader_pin_memory=True,
            seed=train["seed"], data_seed=train["seed"], report_to="none", remove_unused_columns=False,
            include_num_input_tokens_seen=False, ignore_data_skip=False, load_best_model_at_end=False,
            restore_callback_states_from_checkpoint=True,
            label_names=["labels"], use_cpu=False, full_determinism=False)
        callbacks = [monitor]
        if train.get("early_stopping_patience", 0):
            # Supported custom callback avoids loading a 'best' adapter without its matching optimizer state.
            from transformers import TrainerCallback
            from transformers.trainer_callback import ExportableState
            class StopOnWorsening(TrainerCallback, ExportableState):
                best, bad = float("inf"), 0
                def state(self):
                    return {"args": {}, "attributes": {"best": self.best, "bad": self.bad}}
                def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                    val = (metrics or {}).get("eval_loss", float("inf"))
                    if val < self.best:
                        self.best, self.bad = val, 0
                    else:
                        self.bad += 1
                    if self.bad >= train["early_stopping_patience"]:
                        control.should_save = control.should_training_stop = True
                    return control
            callbacks.append(StopOnWorsening())
        trainer = CountingTrainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["validation"],
                                  data_collator=collator, processing_class=tokenizer, callbacks=callbacks,
                                  descriptor=descriptor, retention=train["save_total_limit"])
        return {"trainer": trainer, "tokenizer": tokenizer, "resume": resume, "run_dir": run_dir,
                "monitor": monitor, "lock": run_lock, "audit": audit, "project": project, "train": train}
    except BaseException:
        run_lock.release()
        raise


def export_adapter(root, bundle):
    """Export one new immutable step directory; adapters are not full resumable checkpoints."""
    import os
    trainer = bundle["trainer"]
    parent = root / "models/tagore-qwen35-0.8b-lora" / bundle["run_dir"].name
    target = parent / f"step-{trainer.state.global_step}"
    if target.exists():
        metadata = read_json(target / "export.json")
        if metadata["run_fingerprint"] != canonical_hash(trainer.descriptor):
            raise IncompatibleRunError("Existing export belongs to another experiment")
        write_json(root / "models/latest_adapter.json", {"adapter_path": str(target.resolve())})
        return target
    temp = parent / (".export-" + uuid.uuid4().hex)
    temp.mkdir(parents=True)
    trainer.save_model(str(temp))
    bundle["tokenizer"].save_pretrained(temp)
    write_json(temp / "export.json", {"model_id": bundle["project"]["model_id"], "revision": bundle["project"]["revision"],
                                     "run_fingerprint": canonical_hash(trainer.descriptor), "descriptor": trainer.descriptor,
                                     "global_step": trainer.state.global_step, "created_at": now(),
                                     "files": {p.name: file_hash(p) for p in temp.iterdir() if p.is_file()}})
    os.replace(temp, target)
    write_json(root / "models/latest_adapter.json", {"adapter_path": str(target.resolve())})
    return target


def train(root: Path, project=None, training_config=None, stop_after=None, export=True):
    """Explicit training entry point. Importing this module never starts training."""
    import torch
    project = project or load_config(root)
    training_config = training_config or load_config(root, "training")
    bundle = build_trainer(root, project, training_config, stop_after)
    trainer = bundle["trainer"]
    try:
        resume = bundle["resume"]
        LOG.info("%s", f"Resuming {resume}" if resume else "Starting a new run; no complete compatible checkpoint")
        # Rerunning a finished notebook must not accidentally add another optimizer step.
        saved_state = read_json(resume / "trainer_state.json") if resume is not None else {}
        if resume is not None and saved_state.get("max_steps", 0) > 0 and saved_state["global_step"] >= saved_state["max_steps"]:
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            from transformers.trainer_callback import TrainerState
            weights = load_peft_weights(str(resume), device="cpu")
            set_peft_model_state_dict(trainer.model, weights)
            trainer.state = TrainerState.load_from_json(str(resume / "trainer_state.json"))
            metrics = trainer.evaluate()
            adapter = export_adapter(root, bundle) if export else None
            return {"run_dir": str(bundle["run_dir"]), "global_step": trainer.state.global_step,
                    "resumed_from": str(resume), "adapter_path": str(adapter) if adapter else None,
                    "already_complete": True, "metrics": metrics}
        if resume is None:
            before = trainer.evaluate()
            write_json(root / "logs" / bundle["run_dir"].name / "session_start_evaluation.json", before)
        # On resume, Trainer restores adapter + optimizer + scheduler + RNG before the next step.
        # Do not label an evaluation of the newly initialized base as a resumed-model evaluation.
        with graceful_signals(bundle["monitor"]):
            result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        # Always save full training state, even when the last step is not divisible by save_steps.
        trainer._save_checkpoint(trainer.model, trial=None)
        metrics = trainer.evaluate()
        trainer.save_metrics("train", result.metrics)
        trainer.save_metrics("eval", metrics)
        write_json(root / "logs" / bundle["run_dir"].name / "session_end.json", {
            "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
            "max_steps": trainer.state.max_steps, "stopped_early": trainer.state.global_step < trainer.state.max_steps,
            "session_tokens": bundle["monitor"].total_tokens, "metrics": metrics})
        adapter = export_adapter(root, bundle) if export else None
        return {"run_dir": str(bundle["run_dir"]), "global_step": trainer.state.global_step,
                "resumed_from": str(resume) if resume else None, "adapter_path": str(adapter) if adapter else None,
                "metrics": metrics}
    except torch.cuda.OutOfMemoryError as e:
        LOG.error("CUDA OOM at step %s. Restart the kernel and resume the latest sealed checkpoint with the SAME config. For sequence_length=256/128 or another optimizer, rebuild the dataset and choose a NEW run_name. Do not reuse incompatible optimizer state.", trainer.state.global_step)
        raise RuntimeError("CUDA OOM; see logs and README recovery instructions. Existing sealed checkpoints are retained.") from e
    finally:
        bundle["lock"].release()
        # No persistent reference to the model escapes this function.
        del trainer, bundle
        import gc
        gc.collect()
        torch.cuda.empty_cache()
