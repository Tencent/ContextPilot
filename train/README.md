# ContextPilot Training

This directory contains the reinforcement-learning pipeline used to train
ContextPilot-8B and ContextPilot-14B on LongBench-v2. The implementation builds
on verl and uses vLLM for asynchronous agent rollouts.

## Installation

Create the provided Conda environment and install ContextPilot in editable
mode:

```bash
cd train
conda env create -f agent_rl_env.yml
conda activate agent_rl
pip install --no-deps -e .
```

## Data and models

Download the transformed LongBench-v2 RL dataset:

```bash
cd train
hf download lindsay21/longbench_v2_transformed_rl \
  --repo-type dataset \
  --local-dir data/rl_data/longbench_v2_transformed_rl
```

The default paths are:

```text
data/rl_data/longbench_v2_transformed_rl/data/train-00000-of-00001.parquet
data/rl_data/longbench_v2_transformed_rl/data/val-00000-of-00001.parquet
```

Set `MODEL_PATH` to the SFT checkpoint before launching training.

## Judge configuration

Open-ended LongBench-v2 examples are scored by an OpenAI-compatible judge.
Copy the repository-level environment template and configure the endpoint:

```bash
cp ../.env.example ../.env
```

```dotenv
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://your-judge-endpoint.example/v1
OPENAI_MODEL=your-judge-model
```

## Elasticsearch

ContextPilot uses Elasticsearch 7.17.29 for document retrieval. Install it
once under `train/.runtime`:

```bash
cd train
mkdir -p .runtime
curl -fL -o .runtime/elasticsearch-7.17.29-linux-x86_64.tar.gz \
  https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.29-linux-x86_64.tar.gz
curl -fL -o .runtime/elasticsearch-7.17.29-linux-x86_64.tar.gz.sha512 \
  https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.29-linux-x86_64.tar.gz.sha512
(
  cd .runtime
  sha512sum -c elasticsearch-7.17.29-linux-x86_64.tar.gz.sha512
  tar -xzf elasticsearch-7.17.29-linux-x86_64.tar.gz
)
```

Start the service before training:

```bash
bash sh/start_elasticsearch.sh
curl http://localhost:9200
```

Set `ES_HOME`, `ES_DATA_DIR`, or `ES_PORT` to use an existing installation or
a non-default location.

## Training

Run the 8B recipe:

```bash
cd train
MODEL_PATH=/path/to/model bash sh/run_qwen3-8b_longbenchv2.sh
```

Run the 14B recipe:

```bash
cd train
MODEL_PATH=/path/to/model bash sh/run_qwen3-14b_longbenchv2.sh
```

Configuration can be overridden through environment variables:

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
TRAIN_BATCH_SIZE=4 \
TOTAL_TRAINING_STEPS=4 \
WANDB_MODE=offline \
bash sh/run_qwen3-8b_longbenchv2.sh
```

### Main defaults

| Setting | Default |
| --- | ---: |
| GPUs per node | 8 |
| Training batch size | 16 queries |
| Initial rollouts per query | 8 |
| Retained samples per initial rollout | at most 8 |
| Query-global training-sample budget | at most 128 |
| Maximum concurrent partial branches | 64 |
| Maximum assistant turns | 100 |
| Model context length | 40,960 tokens |
| Maximum generation per assistant turn | 2,048 tokens |
| Proactive context cleanup threshold | 24,000 input tokens |
| Rollout penalty/termination threshold | 26,000 input tokens |
| Sampling temperature / top-p / top-k | 0.7 / 0.8 / 20 |
| vLLM topology | TP=1, DP=8 |

The dataset system message is replaced at runtime by
`configs/contextpilot_system_prompt.txt`, which matches the inference prompt.
Set `CONTEXTPILOT_SYSTEM_PROMPT_FILE` to run with another prompt.

## Checkpoints and logs

Outputs are written to:

```text
runs/contextpilot_longbenchv2/
├── checkpoints/<experiment>/
└── monitor/
    ├── valid_logs/
    ├── trajectories/
    └── validation_data/
```

Set `SAVE_DIR` or `MONITOR_DIR` to change these locations.

The default checkpoint stores model weights and trainer metadata but omits the
optimizer. To save resumable optimizer state, set both
`CHECKPOINT_SAVE_CONTENTS` and `CHECKPOINT_LOAD_CONTENTS` to
`[model,optimizer,extra]`.
