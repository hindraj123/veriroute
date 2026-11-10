"""Supervised LoRA fine-tuning for normalized routing demonstrations."""

import argparse
from typing import Any, Dict

from .dataset import filter_rows, read_jsonl


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install optional dependencies with: pip install -e '.[train]'") from exc

    rows, counts = filter_rows(read_jsonl(args.data), require_legal_route=not args.skip_route_validation)
    print("SFT filter counts:", dict(counts))
    if not rows:
        raise SystemExit("No training examples remain after filtering")
    eval_rows = None
    if args.eval_data:
        eval_rows, eval_counts = filter_rows(
            read_jsonl(args.eval_data), require_legal_route=not args.skip_route_validation
        )
        print("SFT eval filter counts:", dict(eval_counts))
        if not eval_rows:
            raise SystemExit("No evaluation examples remain after filtering")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        trust_remote_code=args.trust_remote_code,
    )
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=args.target_modules.split(","), task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    max_length = args.max_length

    def tokenize(row: Dict[str, Any]) -> Dict[str, Any]:
        prefix = row["prompt"] + "\n"
        full = prefix + row["completion"] + tokenizer.eos_token
        encoded = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
        prefix_tokens = tokenizer(prefix, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
        labels = list(encoded["input_ids"])
        prefix_len = min(len(prefix_tokens), len(labels))
        labels[:prefix_len] = [-100] * prefix_len
        labels = [(-100 if mask == 0 else token)
                  for token, mask in zip(labels, encoded["attention_mask"])]
        encoded["labels"] = labels
        return encoded

    train_dataset = Dataset.from_list(rows)
    train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    eval_dataset = None
    if eval_rows is not None:
        eval_dataset = Dataset.from_list(eval_rows)
        eval_dataset = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)
    training_args = TrainingArguments(
        output_dir=args.output, learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        logging_steps=10, save_strategy="steps", save_steps=args.save_steps,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, report_to="none", remove_unused_columns=False,
    )
    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=eval_dataset).train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-data", default=None,
                        help="Optional held-out JSONL; CircuitNet prep writes eval.jsonl")
    parser.add_argument("--output", default="runs/veriroute-sft-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-route-validation", action="store_true",
                        help="Use only schema/length/dedup filters for non-grid datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    train(parser.parse_args())
