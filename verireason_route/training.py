"""TRL GRPO training with a PEFT LoRA policy and verifier rewards."""

from __future__ import annotations

import argparse

from .data import RoutingTask
from .dataset import filter_rows, read_jsonl
from .rewards import CircuitNetCongestionReward, RewardWeights, RoutingReward, completion_text


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies with: pip install -e '.[train]'") from exc

    records, filter_counts = filter_rows(
        read_jsonl(args.data), require_legal_route=not args.skip_route_validation
    )
    print("GRPO filter counts:", dict(filter_counts))
    if not records:
        raise SystemExit("No GRPO prompts remain after filtering")
    tasks = {}
    for i, row in enumerate(records):
        task_data = row.get("task")
        if task_data is None:
            if not args.skip_route_validation:
                raise ValueError("grid GRPO requires task metadata for verifier rewards")
            # Explicit non-grid mode supports only format reward until a task-specific
            # evaluator is connected; keep a neutral dummy task out of reward calculation.
            task_id = str(i)
            tasks[task_id] = None
        elif task_data.get("task_type") == "circuitnet_congestion":
            task_id = str(task_data.get("task_id", i))
            tasks[task_id] = task_data
        else:
            task_id = str(task_data.get("task_id", i))
            tasks[task_id] = RoutingTask.from_dict(task_data)
        row["task_id"] = task_id
    rows = [{"prompt": row["prompt"], "task_id": row["task_id"]} for row in records]
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        trust_remote_code=args.trust_remote_code,
    )
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        lora = None
    else:
        lora = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(","), task_type="CAUSAL_LM",
        )
    weights = RewardWeights(format=args.reward_format, connectivity=args.reward_connectivity,
                            legality=args.reward_legality, wirelength=args.reward_wirelength,
                            drc=args.reward_drc)
    reward_model = RoutingReward(weights)
    circuitnet_reward = CircuitNetCongestionReward(
        format_weight=args.congestion_reward_format,
        map_weight=args.congestion_reward_map,
        hotspot_weight=args.congestion_reward_hotspot,
    )

    def reward_func(prompts, completions, task_id, **kwargs):
        rewards = []
        for completion, key in zip(completions, task_id):
            completion = completion_text(completion)
            task = tasks[str(key)]
            if task is None:
                from .verifier import extract_json
                parsed = extract_json(completion)
                rewards.append(1.0 if parsed is not None else -1.0)
            elif isinstance(task, dict) and task.get("task_type") == "circuitnet_congestion":
                rewards.append(circuitnet_reward.score(task, completion))
            else:
                external = records_by_task.get(str(key), {}).get("metrics", {})
                rewards.append(reward_model.score(task, completion, external))
        return rewards

    records_by_task = {row["task_id"]: row for row in records}

    config = GRPOConfig(
        output_dir=args.output,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        remove_unused_columns=False,
        beta=args.beta,
        gradient_checkpointing=True,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--init-adapter", default=None,
                        help="Continue GRPO from an SFT PEFT adapter; base model still comes from --model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/veriroute-lora")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-route-validation", action="store_true",
                        help="Skip grid route checks; CircuitNet congestion records still use their map reward")
    parser.add_argument("--reward-format", type=float, default=0.10)
    parser.add_argument("--reward-connectivity", type=float, default=0.20)
    parser.add_argument("--reward-legality", type=float, default=0.40)
    parser.add_argument("--reward-wirelength", type=float, default=0.20)
    parser.add_argument("--reward-drc", type=float, default=0.10)
    parser.add_argument("--congestion-reward-format", type=float, default=0.10)
    parser.add_argument("--congestion-reward-map", type=float, default=0.65)
    parser.add_argument("--congestion-reward-hotspot", type=float, default=0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    train(parser.parse_args())
