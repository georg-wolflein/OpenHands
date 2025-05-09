# ToolArena Evaluation with OpenHands

This folder contains the evaluation harness for [ToolArena](https://github.com/KatherLab/ToolArena).

## Setup Environment and LLM Configuration

Please follow instruction [here](../../README.md#setup) to setup your local development environment and LLM.

## Set up ToolArena
1. In a different folder, clone ToolArena, and follow the [ToolArena installation instructions](https://github.com/georg-wolflein/ToolArena?tab=readme-ov-file#installation).
2. Download the ToolArena data files using the following command:
   ```bash
   toolarena download
   ```
   This will populate files at `ToolArena/tasks/*/data`.
3. Copy the `ToolArena/tasks` folder to the OpenHands repository at `evaluation/benchmark/toolarena/tasks` (this folder is ignored by git).

The script below should perform all of these steps for you:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Clone the repository
git clone https://github.com/KatherLab/ToolArena ../ToolArena
# Temporarily change to ../ToolArena directory
pushd ../ToolArena
# Install dependencies
uv sync --all-groups
# Download data files
uv run toolarena download
# Change back to OpenHands repository
popd
# Copy data files
cp -r ../ToolArena/tasks evaluation/benchmarks/toolarena/tasks
```


## Run Inference on ToolArena

```bash
./evaluation/benchmarks/toolarena/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [dataset_split]

# Example
./evaluation/benchmarks/toolarena/scripts/run_infer.sh llm.eval_gpt4o HEAD
```

where `model_config` is mandatory, and the rest are optional.

- `model_config`, e.g. `eval_gpt4_1106_preview`, is the config group name for your
LLM settings, as defined in your `config.toml`.
- `git-version`, e.g. `HEAD`, is the git commit hash of the OpenHands version you would
like to evaluate. It could also be a release tag like `0.6.2`.
- `agent`, e.g. `CodeActAgent`, is the name of the agent for benchmarks, defaulting
to `CodeActAgent`.
- `eval_limit`, e.g. `10`, limits the evaluation to the first `eval_limit` instances. By
default, the script evaluates the entire ToolArena benchmark (30 tasks). Note:
in order to use `eval_limit`, you must also set `agent`.
- `max_iter`, e.g. `20`, is the maximum number of iterations for the agent to run. By
default, it is set to 30.
- `num_workers`, e.g. `3`, is the number of parallel workers to run the evaluation. By
default, it is set to 1.

