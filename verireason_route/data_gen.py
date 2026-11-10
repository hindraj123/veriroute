"""Generate tiny legal demonstrations for SFT/GRPO prompts."""

import argparse
import json
import random

from .data import generate_task
from .verifier import RoutingVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output", default="data/routes.jsonl")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    verifier = RoutingVerifier()
    with open(args.output, "w", encoding="utf-8") as out:
        for i in range(args.count):
            task = generate_task(rng, args.size, index=i)
            answer = verifier.repair_candidate(task)
            out.write(json.dumps({"prompt": task.prompt(), "completion": answer,
                                  "task": task.to_dict()}, separators=(",", ":")) + "\n")
    print("wrote %d examples to %s" % (args.count, args.output))


if __name__ == "__main__":
    main()
