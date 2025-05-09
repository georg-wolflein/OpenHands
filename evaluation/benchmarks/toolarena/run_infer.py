import asyncio
from collections.abc import Container, Mapping
import os
import re
from typing import Any

import pandas as pd
from datasets import load_dataset

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    "CodeActAgent": codeact_user_response,
}

ENV_VAR_SUBSTITUTION_PATTERN = re.compile(
    r"(?P<full>\$\{env:(?P<var>[A-Za-z_][A-Za-z0-9_]*)\})"
)
LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "tasks")


def substitute_env_vars(s: str) -> str:
    # See https://github.com/KatherLab/ToolArena/blob/main/toolarena/utils.py.

    def substitute(match: re.Match[str]) -> str:
        var = match.group("var")
        if var not in os.environ:
            logger.warning(f"Environment variable {var} not found in environment")
            return match.group("full")
        logger.debug(f"Substituting environment variable {var}")
        return os.environ[var]

    return ENV_VAR_SUBSTITUTION_PATTERN.sub(substitute, s)


def environment_variables_prompt(env: dict) -> str:
    return (
        f"""IMPORTANT: the following environment variables are set in your system environment: {", ".join(f"`{v['name']}`" for v in env)}.
These environment variables are automatically available in your system and will be available within the Python function you implement.
However, if you decide to run the python function yourself, you will need to pass them explicitly to the subprocess.
"""
        if env
        else ""
    )


def make_prompt(instance: pd.Series) -> str:
    return f"""
Your task is to create a tool from the repository {instance["repo"]["name"]} which implements the function `{instance["name"]}` to perform the following task: `{instance["description"]}`.
While you may perform any necessary installations, configurations, downloads or setups, your deliverables are the following two files:
1. A bash script, named `/workspace/install.sh` that will install all necessary dependencies for the tool to run.
2. A Python file, named `/workspace/implementation.py` that will contain the code for the tool.

# Part 1: Install the repository
Clone and locally set up the {instance["repo"]["name"]} repository from GitHub.
Follow these steps:
1. Git clone the repository {instance["repo"]["info"]}.
2. Check the README (find it if it is not in the root directory) and closely follow the recommended instructions to set up the entire repository correctly for the user.
3. Follow the instructions in the README to correctly set up the repository for the user. Perform any necessary installations, configurations, downloads or setups as described. If the repository is in Python, prefer using `pip` as opposed to conda, virtualenv, or similar. Install the repository and its dependencies globally.
4. Make sure that you complete every step, so that a user could directly use this repository without the need to do further setups, installations or downloads. This includes downloading any necessary models. However, do NOT download any datasets.
If you encounter any issues, try to solve them.

{environment_variables_prompt(instance["repo"]["env"])}

# Part 2: Implement the tool function
You need to implement a standalone python function, that can be called independently. 
This function will be called `{instance["name"]}`, and it is described as follows: `{instance["description"]}`.
The function will have the following arguments:
{"\n".join((f"- {arg['name']} ({arg['type']}): {arg['description']}") for arg in instance["arguments"])}

As such, the signature of the function will be:
```python
{instance["python_signature"]}
```
You **must** output a valid, standalone python function that is callable without any modification by a user.
The requirements for the code are:
1. Import the required modules/libraries.
2. You are only allowed to write a single python function. It must start with 'def ...' and end with 'return ...'.
3. You are not allowed to output free texts, test code for the function or anything outside of the function definition.
4. The function needs to be a standalone function that can be called independently.
5. Make sure all required imports are included in the function.
6. The function must perform the task you are given. As a reminder, the task is: `{instance["description"]}`.
7. Make sure the function accepts all required parameters as inputs.
8. The function must have type hints and a docstring.
9. The function must be named exactly `{instance["name"]}`.
10. The function must be a valid python function, that can be executed by a python interpreter.

{environment_variables_prompt(instance["repo"]["env"])}

Remember, you should use the repository `{instance["repo"]["name"]}` to complete the task.
Finally, ensure your function is ready-to-use without any modifications by a user. In many cases, wrapping an existing function, script or module in a subprocess is enough.
Note: It may be useful to run the function with the following example invocation to test it:
```python3
from implementation import {instance["name"]}
{instance["name"]}({", ".join(f"{arg['name']}={next(iter(example_arg['value'] for example_arg in instance['example']['arguments'] if example_arg['name'] == arg['name']))!r}" for arg in instance["arguments"])})
```

# IMPORTANT:
- The only two files that you need to produce are `/workspace/install.sh` and `/workspace/implementation.py` (though you may create other files as well, or install additional dependencies in the process).
- You may use any tools at your disposal to complete the task.
- From within a fresh environment that contains the `/workspace` directory which is empty except for your `install.sh` and `implementation.py` files, it should be possible to run the `install.sh` script, and then run the `implementation.py` file, without any additional prior installations or dependencies.
- The `implementation.py` file should NOT contain any imports at the top of the file. The first line of the file should be the function signature (of the `{instance["name"]}` function). In the body of the function, you may import any necessary modules.
"""


def get_config(
    metadata: EvalMetadata,
    instance: pd.Series,
) -> AppConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = (
        f"ghcr.io/katherlab/toolarena:{instance['requires']}"
    )
    sandbox_config.runtime_startup_env_vars.update(
        {
            arg["name"]: substitute_env_vars(arg["value"])
            for arg in instance["repo"]["env"]
        }
    )
    # # Mount data as read-only
    # sandbox_config.volumes = ", ".join(
    #     f"{LOCAL_DATASET_PATH}/tasks/data/{mount['source']}:/mount/input/{mount['target']}:ro"
    #     for mount in task["example"]["mount"]
    # )
    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=os.environ.get("RUNTIME", "docker"),
        max_budget_per_task=4,
        max_iterations=metadata.max_iterations,
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config,
            metadata.eval_output_dir,
            instance["name"],
        )
    )
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}")
    obs: CmdOutputObservation

    # Set up workspace directories
    action = CmdRunAction(command="mkdir -p /workspace")
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    # Copy data mounts to the workspace
    local_data_dir = os.path.join(LOCAL_DATASET_PATH, instance["name"], "data")
    for mount in instance["example"]["mount"]:
        src = os.path.join(local_data_dir, mount["source"])
        dst = f"/mount/input/{mount['target']}"
        logger.info(f"Copying local data from {src} to {dst} in the runtime.")
        runtime.copy_to(src, dst, recursive=True)

        # Check the mount point exists
        action = CmdRunAction(command=f"ls {dst}")
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.exit_code == 0

    logger.info(f"{'-' * 50} END Runtime Initialization Fn {'-' * 50}")


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called after the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Completion Fn {'-' * 50}")
    obs: CmdOutputObservation

    test_result = {}

    for file in ("install.sh", "implementation.py"):
        action = CmdRunAction(command=f"cat /workspace/{file}")
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)

        test_result[file] = obs.content if obs.exit_code == 0 else None

    logger.info(f"{'-' * 50} END Runtime Completion Fn {'-' * 50}")
    return test_result


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(metadata, instance)

    # Set up the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, "infer_logs")
        reset_logger_for_multiprocessing(logger, instance["name"], log_dir)
    else:
        logger.info(f"Starting evaluation for instance {instance['name']}.")

    instruction = make_prompt(instance)

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)
    initialize_runtime(runtime, instance)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                metadata.agent_class
            ),
        )
    )

    # ======= Attempt to evaluate the agent's edits =======
    test_result = complete_runtime(runtime, instance)

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError("State should not be None.")
    metrics = state.metrics.get() if state.metrics else None

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = compatibility_for_eval_history_pairs(state.history)

    # Save the output
    output = EvalOutput(
        instance_id=instance["instance_id"],
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result=test_result,
    )
    return output


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()

    toolarena_dataset = load_dataset("KatherLab/ToolArena", split="evaluation")
    dataset = pd.DataFrame(toolarena_dataset)

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False
    if llm_config is None:
        raise ValueError(f"Could not find LLM config: --llm_config {args.llm_config}")

    metadata = make_metadata(
        llm_config,
        "ToolArena",
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
    dataset["instance_id"] = dataset["name"].apply(str)
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    run_evaluation(
        instances, metadata, output_file, args.eval_num_workers, process_instance
    )
