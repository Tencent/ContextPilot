# ContextPilot Inference

Evaluate ContextPilot checkpoints on InfBench, NovelQA, LongMemEval, and
BrowseComp+.

## Setup

### 1. Environment

```bash
bash infer/scripts/setup_environment.sh
source infer/.venv/bin/activate
```

The setup script creates `infer/.venv`, installs its Python dependencies, and
downloads Elasticsearch.

### 2. Data

InfBench's `longbook_choice_eng` split is downloaded from Hugging Face
automatically.

NovelQA answer annotations cannot be redistributed. Request full access from
the [NovelQA dataset page](https://huggingface.co/datasets/NovelQA/NovelQA#request-full-accesss),
then set the local data paths:

```bash
export NOVELQA_DATA=/path/to/CopyrightProtected.jsonl
export NOVELQA_CONTEXT_ROOT=/path/to/NovelQA/Books/CopyrightProtected
```

LongMemEval is included with the repository. BrowseComp+ includes the original
obfuscated parquet data; decrypt it locally before evaluation:

```bash
export LONGMEMEVAL_DATA="$PWD/infer/data/LongMemEval/longmemeval_s_cleaned.json"
python infer/scripts/prepare_browsecomp_plus.py \
  infer/data/BrowseCompPlus/data \
  infer/data/BrowseCompPlus/decrypted.jsonl
export BROWSECOMP_PLUS_DATA="$PWD/infer/data/BrowseCompPlus/decrypted.jsonl"
```

LongMemEval and BrowseComp+ use an LLM judge. Store its OpenAI-compatible
endpoint configuration in a JSON file and set:

```bash
export JUDGE_OPENAI_FILE=/path/to/judge_endpoint.json
```

## Evaluation

### Run all tasks

```bash
bash infer/scripts/run_full_pipeline.sh /path/to/checkpoint RUN_ID
```

This starts Elasticsearch and vLLM, evaluates all four benchmarks, scores the
outputs, and shuts down the vLLM server.

### Run one task

```bash
bash infer/scripts/eval_infbench.sh /path/to/checkpoint RUN_ID
bash infer/scripts/eval_novelqa.sh /path/to/checkpoint RUN_ID
bash infer/scripts/eval_longmemeval.sh /path/to/checkpoint RUN_ID
bash infer/scripts/eval_browsecomp_plus.sh /path/to/checkpoint RUN_ID
```

### Reuse an existing server

Start Elasticsearch and vLLM:

```bash
bash infer/scripts/start_elasticsearch.sh
bash infer/scripts/serve_vllm.sh /path/to/checkpoint
```

Then run a task in another terminal:

```bash
export TOKENIZER_PATH=/path/to/checkpoint
export RUN_ID=my-run
bash infer/scripts/eval_task.sh infbench
```

Valid task names are `infbench`, `novelqa`, `longmemeval`, and `bc_plus`.

## Results

Predictions, trajectories, and scores are saved under:

```text
infer/results/<task>/<RUN_ID>/<timestamp>/
```
