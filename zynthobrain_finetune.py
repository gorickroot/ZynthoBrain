"""
ZynthoBrain — LLM Fine-tuning with LoRA + PEFT
================================================
Fine-tunes LLaMA-3 (or any causal LM) on domain-specific data
using parameter-efficient LoRA adapters via the PEFT library.

Requirements:
    pip install -r requirements.txt

Usage (training):
    python zynthobrain_finetune.py \
        --model_name meta-llama/Meta-Llama-3-8B \
        --dataset_path data/my_corpus.jsonl \
        --output_dir ./checkpoints \
        --epochs 5 \
        --batch_size 4

Usage (inference):
    python zynthobrain_finetune.py \
        --infer \
        --model_name meta-llama/Meta-Llama-3-8B \
        --adapter_path ./checkpoints/final_adapter \
        --prompt "Explain LoRA in simple terms."
"""

import os
import sys
import argparse
import logging
from dataclasses import dataclass, field
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer


# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ZynthoBrain")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ZynthoConfig:
    # Model
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    use_4bit: bool = True
    # FIX: bfloat16 is stabler than float16 for QLoRA (less overflow)
    bnb_4bit_compute_dtype: str = "bfloat16"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    dataset_path: str = "data/corpus.jsonl"
    output_dir: str = "./checkpoints"
    seed: int = 42
    epochs: int = 5
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    max_seq_length: int = 2048
    logging_steps: int = 20
    save_steps: int = 200
    eval_steps: int = 200

    # Dataset
    text_column: str = "text"
    val_split: float = 0.05


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_and_prepare_dataset(cfg: ZynthoConfig):
    """
    Load a JSONL/JSON file or HuggingFace Hub dataset.
    Each sample must have a field matching cfg.text_column (default: 'text').
    """
    logger.info(f"Loading dataset: {cfg.dataset_path}")

    ext = os.path.splitext(cfg.dataset_path)[-1].lower()

    # FIX: validate file exists before loading
    if ext in (".jsonl", ".json"):
        if not os.path.isfile(cfg.dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {cfg.dataset_path}\n"
                "Create a JSONL file where each line is: {\"text\": \"your sample here\"}"
            )
        dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")
    else:
        dataset = load_dataset(cfg.dataset_path, split="train")

    # FIX: validate text column exists
    if cfg.text_column not in dataset.column_names:
        raise ValueError(
            f"Column '{cfg.text_column}' not found. "
            f"Available: {dataset.column_names}"
        )

    logger.info(f"Loaded {len(dataset):,} samples")

    # Keep only what we need, rename to 'text' if different
    dataset = dataset.select_columns([cfg.text_column])
    if cfg.text_column != "text":
        dataset = dataset.rename_column(cfg.text_column, "text")

    split = dataset.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    logger.info(f"Train: {len(split['train']):,} | Val: {len(split['test']):,}")
    return split["train"], split["test"]


# ─── Model ────────────────────────────────────────────────────────────────────

def load_base_model(cfg: ZynthoConfig):
    """Load base LM with optional 4-bit QLoRA quantization."""
    logger.info(f"Loading base model: {cfg.model_name}")

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    # FIX: 4-bit quant requires CUDA — gracefully fall back to full precision
    if cfg.use_4bit and not torch.cuda.is_available():
        logger.warning(
            "CUDA not available — disabling 4-bit quantization. "
            "Training on CPU is slow; use a GPU machine for real runs."
        )
        cfg.use_4bit = False

    quant_config = None
    if cfg.use_4bit:
        logger.info("QLoRA: 4-bit NF4 quantization active")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        # FIX: do NOT set torch_dtype when using 4-bit — bnb manages it internally
        torch_dtype=compute_dtype if not cfg.use_4bit else None,
    )

    # FIX: disable KV-cache — incompatible with gradient checkpointing
    model.config.use_cache = False
    # FIX: needed for LLaMA-3 tensor parallelism config
    model.config.pretraining_tp = 1

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    return model


def apply_lora(model, cfg: ZynthoConfig):
    """Inject LoRA adapter matrices into attention projection layers."""
    logger.info(
        f"Injecting LoRA → r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
        f"dropout={cfg.lora_dropout}, targets={cfg.lora_target_modules}"
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,  # must be False during training
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─── Training args ────────────────────────────────────────────────────────────

def build_training_args(cfg: ZynthoConfig) -> TrainingArguments:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    cuda_available = torch.cuda.is_available()
    # FIX: prefer bf16 for QLoRA — less likely to cause loss spikes than fp16
    use_bf16 = cuda_available and cfg.bnb_4bit_compute_dtype == "bfloat16"
    use_fp16 = cuda_available and not use_bf16

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        gradient_checkpointing=True,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        # FIX: 'evaluation_strategy' was renamed to 'eval_strategy' in transformers>=4.41
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        # FIX: hardcoded 4 workers breaks on machines with fewer CPUs
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        # FIX: paged_adamw_8bit requires CUDA + bitsandbytes
        optim="paged_adamw_8bit" if (cfg.use_4bit and cuda_available) else "adamw_torch",
        seed=cfg.seed,
        # FIX: pin_memory causes errors on CPU-only runs
        dataloader_pin_memory=cuda_available,
        remove_unused_columns=False,
    )


# ─── Train ────────────────────────────────────────────────────────────────────

def train(cfg: ZynthoConfig):
    set_seed(cfg.seed)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=False,  # use_fast can silently misbehave on some models
    )

    # FIX: LLaMA-3 has no pad token by default — add one explicitly
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"  # required for SFTTrainer sequence packing

    train_ds, val_ds = load_and_prepare_dataset(cfg)

    model = load_base_model(cfg)

    # FIX: resize embeddings AFTER model load if we added a new pad token
    if tokenizer.pad_token == "<pad>":
        model.resize_token_embeddings(len(tokenizer))

    model = apply_lora(model, cfg)

    training_args = build_training_args(cfg)

    # FIX: use processing_class= instead of deprecated tokenizer= (trl >= 0.9)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=True,
    )

    logger.info("=" * 60)
    logger.info("  ZynthoBrain — training started")
    logger.info(f"  Model        : {cfg.model_name}")
    logger.info(f"  QLoRA 4-bit  : {cfg.use_4bit}")
    logger.info(f"  LoRA r / α   : {cfg.lora_r} / {cfg.lora_alpha}")
    logger.info(f"  Epochs       : {cfg.epochs}")
    logger.info(f"  Eff. batch   : {cfg.batch_size * cfg.grad_accum_steps}")
    logger.info(f"  LR / sched   : {cfg.learning_rate} ({cfg.lr_scheduler})")
    logger.info(f"  Max seq len  : {cfg.max_seq_length}")
    logger.info(f"  Output dir   : {cfg.output_dir}")
    logger.info("=" * 60)

    trainer.train()

    final_path = os.path.join(cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Adapter saved → {final_path}")
    logger.info("Done. Use --infer --adapter_path to test your model.")


# ─── Inference ────────────────────────────────────────────────────────────────

def load_trained_model(base_model_name: str, adapter_path: str):
    """Load base model + LoRA adapter, merge weights for fast inference."""
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    logger.info(f"Loading tokenizer from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)

    logger.info(f"Loading base model: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # FIX: resize before loading adapter in case pad token was added at train time
    base.resize_token_embeddings(len(tokenizer))

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    # Merge adapter weights into base for faster inference (removes adapter overhead)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            # FIX: pass attention_mask explicitly to silence pad-token warnings
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            # FIX: set these explicitly to avoid GenerationConfig warnings
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # FIX: decode only the new tokens, not the entire prompt + response
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ZynthoBrain — LoRA/QLoRA fine-tuning for causal LMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--infer", action="store_true",
                   help="Run inference mode instead of training")

    # Model
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B",
                   help="HuggingFace model ID or local path")
    p.add_argument("--no_4bit", action="store_true",
                   help="Disable QLoRA 4-bit quantization (uses full precision)")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--dataset_path", default="data/corpus.jsonl",
                   help="Path to .jsonl file or HuggingFace dataset name")
    p.add_argument("--text_column", default="text",
                   help="Column name containing training text")
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    # Inference
    p.add_argument("--adapter_path", default=None,
                   help="Path to saved adapter directory (required for --infer)")
    p.add_argument("--prompt", default="Tell me about large language models.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.infer:
        if not args.adapter_path:
            logger.error("--adapter_path is required for inference mode.")
            sys.exit(1)
        model, tokenizer = load_trained_model(args.model_name, args.adapter_path)
        logger.info(f'Prompt: "{args.prompt}"')
        response = generate(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n" + "─" * 60)
        print(response)
        print("─" * 60)

    else:
        cfg = ZynthoConfig(
            model_name=args.model_name,
            use_4bit=not args.no_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dataset_path=args.dataset_path,
            text_column=args.text_column,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )
        train(cfg)
