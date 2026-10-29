# Source and compatibility research

Researched 29 October 2026 using official model repositories, framework documentation, archive pages and primary copyright law. This records source inspection, not successful local training. Network activity stopped after the user's repository-only instruction. No source corpus or model weights were downloaded for this deliverable.

## Official model identity

- [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base): pretraining/base repository, Apache-2.0 model license. The instruct/post-trained `Qwen/Qwen3.5-0.8B` is a separate repository and is not silently substituted.
- [Pinned official configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json): inspected `model_type=qwen3_5`, `Qwen3_5ForConditionalGeneration`; text config `qwen3_5_text`, 24 layers, width 1024, vocabulary 248320, tied embeddings, EOS 248044. The stack has three linear-attention layers per full-attention layer. This small checkpoint is a dense hybrid model; the family also includes distinct MoE variants.
- [Official Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5): distinguishes text-only `Qwen3_5ForCausalLM`/text config from multimodal conditional generation. It describes optional fast DeltaNet kernels and slower/more-memory-intensive fallback paths.
- [Transformers v5.3.0 Qwen3.5 API](https://huggingface.co/docs/transformers/v5.3.0/model_doc/qwen3_5): the baseline pins this model-supporting v5 release. Source-level/runtime checks still need execution; a documentation page is not a compatibility test of all project dependencies.
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora): adapter-based training, with actual target modules checked dynamically. No guessed module is silently skipped.
- [TRL SFTTrainer v0.29.0](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer): language-modeling datasets are supported. The project chooses Transformers Trainer's explicit pretokenized causal path to keep formatting, EOS budget and loss masking under one implementation. TRL remains pinned and included in version/fingerprint reports.
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune): confirms support for the small Qwen3.5 models and Transformers v5, and currently discourages 4-bit QLoRA because of quantization differences. Unsloth's published memory figures depend on its own backend and were not treated as measurements of this repository. No unverified Unsloth API is included.
- [PyTorch installation selector](https://pytorch.org/get-started/locally/): confirm wheel/driver compatibility for your environment before installation. The project offers a pinned torch 2.10.0/CUDA 12.8 candidate, not an assertion that every NVIDIA/WSL configuration can use it.

The official checkpoint's complete weight-key layout is verified at runtime using its pinned index. The loader handles a language backbone nested under `model.language_model` or `language_model`, rejects unknown layouts and missing text weights, and avoids keeping a vision model on CUDA. Actual weight loading has not been executed.

## Bengali sources and permitted access

| Source | Evidence and treatment |
| --- | --- |
| [Bengali Wikisource Tagore author catalog](https://bn.wikisource.org/wiki/লেখক:রবীন্দ্রনাথ_ঠাকুর) | Trustworthy discovery list for original Bengali editions. Work-level author/rights/QC evidence is still required. Catalog pages and links are not literary training text. |
| [Bengali Wikisource robots](https://bn.wikisource.org/robots.txt) | Inspected rules disallow `/w/` and `/api/` for the collector. Live rules are fetched and enforced. No robots override or API-path bypass. Permitted article pages are the implemented fallback. |
| [Wikimedia User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy) | A descriptive bot name with a real operator contact is required by the implementation. |
| [Wikimedia API usage guidelines](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines) | Respect throttling, licenses and robot policy; no concurrency/spikes or identity rotation. |
| [SNLTR Rabindra Rachanabali FAQ](https://rabindra-rachanabali.nltr.org/index.php?q=node%2F8) | FAQ states page-by-page download/copy permissions. Current site footer also asserts rights. Use a documented authorized manual import; no unverified site-wide crawler. Exclude modern site prose and musical recordings. |
| [SNLTR collection](https://rabindra-rachanabali.nltr.org/) | Government-backed Unicode collection covering several literary genres. Site introduction identifies Visva-Bharati editions as the Bengali text basis. Bibliographic metadata must be verified for the individual import. |
| [Bichitra, Jadavpur University](https://bichitra.jdvu.ac.in/index.php) | Institutional Tagore variorum with manuscripts, transcriptions, editions and bibliographic aids, in collaboration with Rabindra-Bhavana/Visva-Bharati. No bulk automated reuse permission established; authorized manual import only. |

No general archive crawler is added on the assumption that public access implies permission. Additional institutional collections can be incorporated through the same evidence-bearing manual manifest after verifying their terms, original authorship and edition rights. This avoids converting modern editorial material or OCR guesses into purported original literature.

## Copyright distinction

[India Copyright Act, Chapter V, sections 22–24](https://copyright.gov.in/Copyright_Act_1957/chapter_v.html) distinguishes lifetime-published works from anonymous/pseudonymous and posthumous cases. Applying section 22's life-plus-60 rule to Tagore's 1941 death supports expiry in India from 1 January 2002 for the covered lifetime-published works. This is an inference from the statutory rule and the recorded death year, not a blanket rights determination for every edition, translation, scan, unpublished letter, editorial contribution or country.

The pipeline records the underlying-work rights evidence and site/transcription terms separately. Missing or uncertain notices enter manual review. Source URLs, licenses, author/edition fields and reviewers remain metadata; the training text contains only literary material. Redistributors must preserve attribution and applicable licensing obligations. No blanket license is assigned to a future gathered corpus or trained adapter by this repository.
