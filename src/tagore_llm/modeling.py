"""Pinned Qwen3.5 text backbone loading with explicit checkpoint-key and LoRA validation."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import gc
import shutil
import subprocess
import warnings

from .common import read_json


def environment_report(root: Path):
    import torch
    import psutil
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes"):
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = "not installed"
    report = {"packages": packages, "torch_cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(),
              "ram_total_gib": psutil.virtual_memory().total / 1024**3,
              "ram_available_gib": psutil.virtual_memory().available / 1024**3,
              "disk_free_gib": shutil.disk_usage(root).free / 1024**3}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        report.update({"gpu": p.name, "vram_total_gib": total / 1024**3, "vram_free_gib": free / 1024**3,
                       "compute_capability": f"{p.major}.{p.minor}", "bf16_supported": torch.cuda.is_bf16_supported()})
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,power.limit,driver_version", "--format=csv"],
                           capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = s.stdout.strip() or s.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        report["nvidia_smi"] = str(e)
    return report


def precision(device="cuda"):
    import torch
    if device == "cpu":
        return torch.float32
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install the matching CUDA PyTorch wheel and verify nvidia-smi.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def verify_model(config):
    from transformers import AutoConfig
    full = AutoConfig.from_pretrained(config["model_id"], revision=config["revision"], trust_remote_code=False)
    if full.model_type != "qwen3_5" or not hasattr(full, "text_config"):
        raise ValueError("Expected the pinned Qwen3.5 multimodal checkpoint with a text backbone")
    text = full.text_config
    if text.model_type != "qwen3_5_text" or text.num_hidden_layers != 24 or text.hidden_size != 1024:
        raise ValueError("Unexpected model architecture; inspect the repository before proceeding")
    return full, {"repository": config["model_id"], "revision": config["revision"],
                  "architecture": full.architectures, "text_architecture": text.model_type,
                  "vocabulary": text.vocab_size, "layers": text.num_hidden_layers,
                  "hidden_size": text.hidden_size, "layer_types": text.layer_types,
                  "eos_token_id": text.eos_token_id}


def load_text_model(config, device="cuda", training=False):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3_5ForCausalLM
    full, _ = verify_model(config)
    index = read_json(Path(hf_hub_download(config["model_id"], "model.safetensors.index.json", revision=config["revision"])))
    keys = list(index["weight_map"])
    # The official repository includes vision weights; remap only the language backbone.
    if any(k.startswith("model.language_model.") for k in keys):
        mapping = {r"^model\.language_model\.": "model."}
    elif any(k.startswith("language_model.") for k in keys):
        mapping = {r"^language_model\.": "model."}
    elif any(k.startswith("model.layers.") for k in keys):
        mapping = {}
    else:
        raise ValueError("Unrecognized official checkpoint keys; refusing to initialize missing text weights")
    dtype = precision(device)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], config=full.text_config,
        key_mapping=mapping, dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=False, output_loading_info=True, low_cpu_mem_usage=True)
    missing = [k for k in info.get("missing_keys", []) if k != "lm_head.weight"]
    if missing or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Text backbone not completely loaded: {info}")
    if "lm_head.weight" in info.get("missing_keys", []) and not model.config.tie_word_embeddings:
        raise RuntimeError("Missing output head without tied embeddings")
    unexpected_text = [k for k in info.get("unexpected_keys", [])
                       if ("language_model." in k or k.startswith("model.layers.") or k.startswith("model.embed_tokens."))
                       and ".mtp." not in k]
    if unexpected_text:
        raise RuntimeError(f"Checkpoint text weights were not consumed: {unexpected_text[:10]}")
    model.tie_weights()
    model.config.use_cache = not training
    model.to(device)
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, {"loading_info": info, "dtype": str(dtype), "device": device,
                   "text_parameters": sum(p.numel() for p in model.parameters()), "key_mapping": mapping}


def apply_lora(model, training_config, project_config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    requested = training_config["target_modules"]
    found = {target: [name for name, module in model.named_modules()
                      if name.endswith("." + target) and isinstance(module, torch.nn.Linear)] for target in requested}
    if any(not matches for matches in found.values()):
        raise ValueError(f"LoRA target modules absent/not Linear: {[k for k,v in found.items() if not v]}. Inspect the pinned architecture.")
    config = LoraConfig(r=training_config["lora_rank"], lora_alpha=training_config["lora_alpha"],
                        lora_dropout=training_config["lora_dropout"], target_modules=requested,
                        bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, config)
    model.peft_config["default"].base_model_name_or_path = project_config["model_id"]
    model.peft_config["default"].revision = project_config["revision"]
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise ValueError("Expected trainable LoRA parameters only; embeddings and base weights must stay frozen")
    return model, {"matched_modules": found, "trainable_parameters": sum(p.numel() for _, p in trainable)}


def release_model(model=None):
    import torch
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
