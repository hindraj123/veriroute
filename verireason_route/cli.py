"""Command line entry points."""

import argparse
import json
import random
from pathlib import Path

from .data import generate_task
from .data_gen import main as make_data
from .verifier import RoutingVerifier


def demo():
    task = generate_task(random.Random(21), size=8, index=0)
    verifier = RoutingVerifier()
    candidate = verifier.repair_candidate(task)
    print(task.prompt())
    print("Candidate:", candidate)
    print("Evaluation:", json.dumps(verifier.evaluate(task, candidate).__dict__, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="veriroute")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo")
    data_parser = subparsers.add_parser("make-data")
    data_parser.add_argument("--count", type=int, default=200)
    data_parser.add_argument("--output", default="data/routes.jsonl")
    data_parser.add_argument("--seed", type=int, default=17)
    data_parser.add_argument("--size", type=int, default=8)
    train_parser = subparsers.add_parser("train")
    from .training import add_train_args
    add_train_args(train_parser)
    sft_parser = subparsers.add_parser("sft")
    from .sft import add_args as add_sft_args
    add_sft_args(sft_parser)
    filter_parser = subparsers.add_parser("filter-data")
    filter_parser.add_argument("--input", required=True)
    filter_parser.add_argument("--output", required=True)
    filter_parser.add_argument("--skip-route-validation", action="store_true")
    circuitnet_parser = subparsers.add_parser("circuitnet-prepare")
    from .circuitnet import add_args as add_circuitnet_args
    add_circuitnet_args(circuitnet_parser)
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "make-data":
        # Inline generation keeps the installed entry point independent of argv parsing.
        from .verifier import RoutingVerifier
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        rng, verifier = random.Random(args.seed), RoutingVerifier()
        with open(args.output, "w", encoding="utf-8") as out:
            for i in range(args.count):
                task = generate_task(rng, size=args.size, index=i)
                row = {"prompt": task.prompt(), "completion": verifier.repair_candidate(task), "task": task.to_dict()}
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        print("wrote %d examples to %s" % (args.count, args.output))
    elif args.command == "train":
        from .training import train
        train(args)
    elif args.command == "sft":
        from .sft import train
        train(args)
    elif args.command == "filter-data":
        from .dataset import filter_rows, read_jsonl
        rows, counts = filter_rows(read_jsonl(args.input),
                                  require_legal_route=not args.skip_route_validation)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        print("filter counts:", json.dumps(dict(counts), sort_keys=True))
        print("wrote %d examples to %s" % (len(rows), args.output))
    elif args.command == "circuitnet-prepare":
        from .circuitnet import prepare
        train_count, eval_count = prepare(args.features, args.labels, args.output,
                                          args.patch_size, args.stride, args.bins,
                                          args.eval_fraction, args.seed, args.max_samples)
        print("wrote %d train and %d eval patches under %s" %
              (train_count, eval_count, args.output))


if __name__ == "__main__":
    main()
