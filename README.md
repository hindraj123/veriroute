# VeriRoute

VeriRoute is a research prototype adapting VeriReason's LoRA SFT and
verifier-guided GRPO recipe to chip-design data. Its current CircuitNet path
fine-tunes a language model to predict quantized congestion-map patches, then
can apply GRPO to hidden-label map rewards. A separate toy grid-routing path
prototypes route legality rewards. Neither path is yet a physical-design router.

This starter uses a compact textual board representation and a JSON action
format, so it can be tested without a proprietary EDA tool. It is an
educational prototype, not a signoff router. For real designs, replace the
reference grid parser/router with a LEF/DEF or OpenROAD adapter and preserve the
same verifier interface.

## Quick start

The environment/verifier requires only Python 3.9+:

```sh
python3 -m verireason_route.cli demo
python3 -m verireason_route.cli make-data --count 200 --output data/routes.jsonl
```

## Dataset and training workflow

Training data is JSONL with `prompt`/`completion` and routing `task` metadata.
Common `instruction`/`response`, `input`/`output`, and chat `messages` records
are normalized too. The current strict filter verifies the target route against
the included grid task, drops malformed/duplicate examples, and prints rejection
counts. For another dataset that is not represented in the grid schema, use
`--skip-route-validation`; only schema, length, and duplicate checks then apply.
You must connect a dataset-specific evaluator before treating its GRPO reward as
physical routing feedback.

```sh
python3 -m verireason_route.cli filter-data --input data/raw.jsonl --output data/filtered.jsonl
python3 -m verireason_route.cli sft --model Qwen/Qwen2.5-0.5B-Instruct --data data/filtered.jsonl --output runs/veriroute-sft-lora
python3 -m verireason_route.cli train --model Qwen/Qwen2.5-0.5B-Instruct --data data/filtered.jsonl --output runs/veriroute-grpo-lora
```

The default reward has configurable weights for output format, endpoint
connectivity, route legality, wirelength, and DRC. DRC is neutral until supplied
as `task`-adjacent `metrics.drc_violations` produced by a real backend. Adjust
weights with `--reward-format`, `--reward-connectivity`, `--reward-legality`,
`--reward-wirelength`, and `--reward-drc`. For non-grid records, the current
GRPO fallback is format-only and is intended only to validate plumbing.

## CircuitNet Stage A: congestion SFT

The first CircuitNet experiment is congestion-map prediction. CircuitNet's
official congestion inputs are macro-region, RUDY, and pin-RUDY maps; its target
is the global-routing congestion map. The adapter below consumes the
`feature/` and `label/` `.npy` files produced by CircuitNet's official
`generate_training_set.py --task congestion` preprocessing. It splits whole
layout samples into train/eval **before** extracting patches, then serializes
quantized 8×8 patches as compact text for the causal LM. This is an exploratory
LLM baseline for a spatial prediction task; a CNN/FCN remains a more natural
baseline for map-to-map prediction.

This adapter currently targets the packed array format produced by CircuitNet's
official N28-style preprocessing; other releases must first be checked against
the same H×W×3 feature / H×W×1 label contract. CircuitNet data is not downloaded
by this project. Follow the official
[CircuitNet download/preprocessing guide](https://circuitnet.github.io/intro/download.html)
and make sure the prepared feature and label maps share filenames and have
three feature channels and one congestion label channel.

```sh
python3 -m pip install -e '.[train,circuitnet]'
python3 -m verireason_route.cli circuitnet-prepare \
  --features /data/CircuitNet/congestion/feature \
  --labels /data/CircuitNet/congestion/label \
  --output data/circuitnet-congestion --patch-size 8 --stride 8 \
  --max-samples 20000
python3 -m verireason_route.cli sft \
  --model models/Qwen2.5-0.5B-Instruct-checkpoint \
  --data data/circuitnet-congestion/train.jsonl \
  --eval-data data/circuitnet-congestion/eval.jsonl \
  --skip-route-validation \
  --output runs/circuitnet-congestion-lora
python3 -m verireason_route.cli train \
  --model models/Qwen2.5-0.5B-Instruct-checkpoint \
  --init-adapter runs/circuitnet-congestion-lora \
  --data data/circuitnet-congestion/train.jsonl \
  --skip-route-validation \
  --output runs/circuitnet-congestion-grpo
```

This establishes a two-stage experiment on **congestion prediction**: SFT learns
the reference map, then GRPO optimizes a hidden-label reward combining output
format, normalized map MAE, and congestion-hotspot F1. It does not yet train a
routing policy or claim physical-design improvement. For routing-policy GRPO,
we still need a router/evaluator that runs each candidate and returns legality,
DRC, congestion, timing, and wirelength metrics; CircuitNet's static
feature/label pairs do not provide candidate-action outcomes.

To process the entire dataset, pass `--max-samples 0`; the official CircuitNet
download/preprocessing documentation warns that decompression can need about
200 GB of storage. Start with the capped pilot first.

Install only the optional training dependencies in a GPU-enabled environment:

```sh
python3 -m pip install -e '.[train]'
```

Training uses a small instruct model by default in examples; use a larger coder
model and tune batch size/sequence length for available memory. It saves a PEFT
LoRA adapter and tokenizer under the output directory. `--max-steps` is useful
for smoke runs. The GRPO implementation relies on the installed TRL API.

## Local training setup

This workspace has a project-local `.venv` with the training packages installed
and the Qwen2.5-0.5B-Instruct base checkpoint under
`models/Qwen2.5-0.5B-Instruct-checkpoint`. Activate the environment with
`source .venv/bin/activate`. The current Python/PyTorch runtime reports MPS as
unavailable, so training will fall back to CPU until an MPS-enabled runtime or
CUDA machine is available. CircuitNet feature/label arrays are still needed
before either CircuitNet training stage can start.

## Task/action format

The prompt contains a grid (`#` blocked, `.` free), source, target, and existing
routed nets. The model returns JSON such as:

```json
{"reasoning":"Go around the obstacle on the north side.","path":[[0,0],[0,1],[1,1]]}
```

Coordinates are `[row, column]`; paths must be orthogonally adjacent. Other
nets occupy their cells and are kept separate (no multi-terminal nets yet).
The verifier checks JSON/schema, endpoints, bounds, obstacles, adjacency,
shortest-path excess, and conflicts against the existing routing. It also
computes a deterministic shortest path to provide a useful verifier-backed
repair candidate and supervision data.

## Extending to physical design

The core seam is `RoutingVerifier.evaluate()` and the prompt/data schema in
`data.py`. Connect a real backend there to run detailed route plus design-rule
checks, and return structured metrics (opens, shorts, spacing/width violations,
wirelength, via count, overflow). Keep hard validity signals separate from
soft PPA/congestion terms. Dataset examples should represent real routing
windows/nets with a legal reference route and reproducible verifier setup.

## Method mapping

The paper's SFT examples become legal route demonstrations; testbench feedback
becomes routing/DRC feedback; syntax/structural rewards become JSON, geometry,
connectivity, and design-rule rewards; and GRPO group-relative optimization is
applied to several route candidates per prompt. This repository includes the
verifier and LoRA GRPO scaffold; quality depends on the realism of the supplied
routing tasks and verifier.
