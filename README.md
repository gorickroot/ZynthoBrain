# 🧠 ZynthoBrain — LLM Fine-tuning with LoRA + PEFT
 
<div align="center">
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-PEFT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Progress-a855f7?style=for-the-badge)
 
**Fine-tuning LLaMA-3 on domain-specific data using parameter-efficient LoRA adapters.**  
A deep-learning project pushing the boundaries of modern NLP — trained with QLoRA on a single GPU.
 
</div>
---
 
## What is ZynthoBrain?
 
ZynthoBrain is a clean, production-ready pipeline for fine-tuning large language models using **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized LoRA). Instead of retraining billions of parameters from scratch, LoRA injects tiny trainable adapter matrices into the model's attention layers — achieving great results with a fraction of the compute.
 
```
Base Model (frozen) + LoRA Adapters (trained) = Fine-tuned Model
       ~8B params          ~0.1% of params
```
 
---
 
## Features
 
- **QLoRA** — 4-bit NF4 quantization via `bitsandbytes` — fits LLaMA-3 8B on a single 10GB GPU
- **LoRA** — adapter injection into `q_proj`, `v_proj`, `k_proj`, `o_proj` attention layers
- **SFTTrainer** — supervised fine-tuning with sequence packing for maximum efficiency
- **Cosine LR schedule** — with warmup for stable convergence
- **Auto train/val split** — configurable validation set from your corpus
- **Inference mode** — load saved adapter and generate text in one command
- **Full CLI** — every hyperparameter is configurable via flags
---
 
## Tech Stack
 
| Tool | Purpose |
|------|---------|
| `transformers` | Base model loading, tokenizer, training loop |
| `peft` | LoRA adapter injection and management |
| `trl` | SFTTrainer for supervised fine-tuning |
| `bitsandbytes` | 4-bit NF4 quantization (QLoRA) |
| `datasets` | Dataset loading and preprocessing |
| `accelerate` | Multi-GPU / mixed precision support |
| `tensorboard` | Training metrics and loss curves |
 
---
 
## Installation
 
```bash
git clone https://github.com/gorickroot/ZynthoBrain.git
cd ZynthoBrain
pip install -r requirements.txt
```
 
> Requires Python 3.10+, CUDA-compatible GPU (10GB+ VRAM recommended)
 
---
 
## Dataset Format
 
Create a `.jsonl` file where each line is a JSON object with a `text` field:
 
```jsonl
{"text": "Your first training sample goes here."}
{"text": "Another domain-specific example."}
{"text": "Keep going — one sample per line."}
```
 
Place it at `data/corpus.jsonl` or pass `--dataset_path` to point elsewhere.
 
---
 
## Training
 
```bash
python zynthobrain_finetune.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset_path data/corpus.jsonl \
  --output_dir ./checkpoints \
  --epochs 5 \
  --batch_size 4 \
  --lora_r 16 \
  --lora_alpha 32
```
 
Monitor training live:
 
```bash
tensorboard --logdir ./checkpoints/logs
```
 
---
 
## Inference
 
```bash
python zynthobrain_finetune.py \
  --infer \
  --model_name meta-llama/Meta-Llama-3-8B \
  --adapter_path ./checkpoints/final_adapter \
  --prompt "Explain attention mechanisms in transformers." \
  --max_new_tokens 256
```
 
---
 
## CLI Reference
 
| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `meta-llama/Meta-Llama-3-8B` | HuggingFace model ID or local path |
| `--dataset_path` | `data/corpus.jsonl` | Path to JSONL file or HF dataset name |
| `--output_dir` | `./checkpoints` | Where to save adapters and logs |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--grad_accum` | `4` | Gradient accumulation steps |
| `--lora_r` | `16` | LoRA rank (higher = more trainable params) |
| `--lora_alpha` | `32` | LoRA alpha scaling (keep at 2x rank) |
| `--learning_rate` | `2e-4` | Peak learning rate |
| `--max_seq_length` | `2048` | Maximum tokens per sequence |
| `--no_4bit` | off | Disable QLoRA (run in full precision) |
| `--seed` | `42` | Random seed for reproducibility |
| `--infer` | off | Switch to inference mode |
| `--adapter_path` | None | Path to saved adapter (required for --infer) |
| `--prompt` | — | Input prompt for inference |
| `--max_new_tokens` | `256` | Max tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
 
---
 
## How LoRA Works
 
Instead of updating all 8 billion weights of LLaMA-3, LoRA freezes the original model and inserts small low-rank matrices into each attention layer:
 
```
Output = W_frozen x  +  (B x A) x
          (frozen)         (trained)
 
Where: A is R^(r x d),  B is R^(d x r),  rank r << d
```
 
With r=16 across 4 attention modules, ZynthoBrain trains roughly **0.1% of total parameters** — making fine-tuning feasible on consumer hardware in hours.
 
---
 
## GPU Memory Guide
 
| Model | Mode | Min VRAM |
|-------|------|----------|
| LLaMA-3 8B | QLoRA 4-bit | ~10 GB |
| LLaMA-3 8B | LoRA fp16 | ~18 GB |
| LLaMA-3 70B | QLoRA 4-bit | ~40 GB |
 
---
 
## Project Structure
 
```
ZynthoBrain/
├── zynthobrain_finetune.py   # Main training + inference script
├── requirements.txt          # All dependencies with versions
├── LICENSE                   # MIT License
├── data/                     # Put your corpus.jsonl here
└── checkpoints/              # Saved LoRA adapters + TensorBoard logs
```
 
---
 
## Roadmap
 
- [x] LoRA + QLoRA training pipeline
- [x] Inference mode with adapter merging
- [x] CLI with full hyperparameter control
- [ ] W&B logging integration
- [ ] Data preprocessing script
- [ ] Gradio inference UI
- [ ] Multi-GPU training support
---

 ## 👨‍💻 Author

**Gorick Nath**
BSc Computing Science — Griffith College Dublin, Ireland 🇮🇪

Building AI agents & automations 🤖 · Founding ZynthoAI

> *"Every line of code is a step forward."*

[![Portfolio](https://img.shields.io/badge/Portfolio-gorickroot.github.io-2d6a4f?style=flat&logo=google-chrome&logoColor=white)](https://gorickroot.github.io)
[![GitHub](https://img.shields.io/badge/GitHub-gorickroot-181717?style=flat&logo=github)](https://github.com/gorickroot)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-gorick--nath--aigeek-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/gorick-nath-aigeek)

---
## License
 
MIT — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.
 
---
 
<div align="center">
  <i>Built as part of a deep-learning portfolio project exploring modern NLP fine-tuning techniques.</i>
</div>
 
