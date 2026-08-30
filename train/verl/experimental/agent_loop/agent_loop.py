# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the ContextPilot project in 2026.

import asyncio
import copy
import logging
import os
import queue
import random
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, List, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_handler)
    logger.propagate = False


def _seed_component(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value)
        return sum((index + 1) * ord(char) for index, char in enumerate(text))


def contextpilot_trajectory_seed(trajectory: dict[str, Any]) -> int:
    base_seed = int(os.getenv("CONTEXTPILOT_ROLLOUT_SEED", "0"))
    step = _seed_component(trajectory.get("step", 0))
    sample_index = _seed_component(trajectory.get("sample_index", 0))
    rollout_n = _seed_component(trajectory.get("rollout_n", 0))
    return (
        base_seed
        + 1_000_003 * (step + 1)
        + 10_007 * sample_index
        + 97_409 * rollout_n
    ) & 0x7FFFFFFF


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing
    - Sticky session: keep multi-turn requests on one server for prefix caching while
      allowing overloaded sessions to migrate and avoid replica tail stragglers
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)
        if not self.server_handles:
            raise ValueError("AsyncLLMServerManager requires at least one server handle")

        self._server_order = {server: index for index, server in enumerate(self.server_handles)}
        self._inflight_requests = {server: 0 for server in self.server_handles}
        self._total_assignments = {server: 0 for server in self.server_handles}
        self._sticky_max_load_skew = max(
            0,
            int(os.getenv("VERL_AGENT_STICKY_MAX_LOAD_SKEW", "2")),
        )
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _least_loaded_server(self) -> ray.actor.ActorHandle:
        return min(
            self.server_handles,
            key=lambda server: (
                self._inflight_requests[server],
                self._total_assignments[server],
                self._server_order[server],
            ),
        )

    def _acquire_server(self, request_id: str) -> ray.actor.ActorHandle:
        least_loaded = self._least_loaded_server()
        server = self.request_id_to_server.get(request_id)
        if server is None or (
            self._inflight_requests[server]
            > self._inflight_requests[least_loaded] + self._sticky_max_load_skew
        ):
            server = least_loaded
            self.request_id_to_server[request_id] = server

        self._inflight_requests[server] += 1
        self._total_assignments[server] += 1
        return server

    def _release_server(self, server: ray.actor.ActorHandle) -> None:
        inflight = self._inflight_requests[server]
        if inflight <= 0:
            raise RuntimeError("AsyncLLMServerManager request accounting underflow")
        self._inflight_requests[server] = inflight - 1

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        server = self._acquire_server(request_id)
        self._release_server(server)
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._acquire_server(request_id)
        try:
            return await server.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )
        finally:
            self._release_server(server)


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class MultiTrajectoryAgentLoopOutput(BaseModel):
    """Agent loop output containing ContextPilot snapshot trajectories."""

    trajectories: list[AgentLoopOutput]
    """List of trajectories, including snapshots and final trajectory."""
    contextpilot_branch_points: list[dict[str, Any]] = Field(default_factory=list)
    contextpilot_state: Any = None


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote(num_cpus=0)
class BatchExecutor:
    """Batch executor is used to collect requests into a batch execution"""

    def __init__(self, batch_func, micro_batch_size=1, max_batch_size=None):
        """

        Args:
            batch_func: batch processing function.
            micro_batch_size (int, optional): micro batch size. Defaults to 1.
            max_batch_size: batch size for batching.
        """
        self._q = queue.Queue()
        self._batch_func = batch_func
        self._max_batch = max_batch_size
        self._micro_batch_size = micro_batch_size

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    async def submit_task(self, item):
        """
        Blocking submission, returning Future
        Args:
            item: function input

        Returns:
            fut: function output
        """
        fut = Future()
        self._q.put((item, fut))
        async_fut = asyncio.wrap_future(fut)
        res = await async_fut
        return res

    def _worker_loop(self):
        while True:
            first, first_fut = self._q.get()
            items = [first]
            futs = [first_fut]

            while True:
                try:
                    next_item, next_fut = self._q.get_nowait()
                    items.append(next_item)
                    futs.append(next_fut)
                    if self._max_batch and len(items) >= self._max_batch:
                        break
                except queue.Empty:
                    while len(items) % self._micro_batch_size != 0:
                        next_item, next_fut = self._q.get()
                        items.append(next_item)
                        futs.append(next_fut)
                        if self._max_batch and len(items) >= self._max_batch:
                            break
                    break

            try:
                results = self._batch_func(items)
            except Exception as e:
                for f in futs:
                    f.set_exception(e)
            else:
                for f, r in zip(futs, results, strict=False):
                    f.set_result(r)


@ray.remote(num_cpus=0)
class RewardManagerWorker:
    """Reward manager worker to compute reward score asynchronously to overlap with agent loop."""

    def __init__(self, config: DictConfig, local_path: str, rm_executor: BatchExecutor = None) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reward_manager = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.rm_executor = rm_executor
        self.loop = asyncio.get_event_loop()

    async def compute_score(
        self,
        data: DataProto,
    ) -> dict:
        """Compute reward score for agent loop output.

        NOTE: Since `reward_manager.__call__` is blocking function, we run it in thread pool to
        compute multiple samples in parallel.

        Args:
            data: reward function input

        Returns:
            dict: Reward score and reward extra info.
        """
        result = await self.loop.run_in_executor(
            None,
            self.reward_wrapper,
            data,
            True,  # return_dict
        )

        reward_score = result["reward_tensor"].sum(dim=-1).item()
        reward_extra_info = {k: v[0] for k, v in result.get("reward_extra_info", {}).items()}
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}

    def reward_wrapper(self, data: DataProto, return_dict=False) -> torch.Tensor:
        """Assemble reward functions and reward model into one function and expose it to the event loop
        Args:
            return_dict: whether return as dict
            data: DataProto from compute reward score
        Returns:
            torch.Tensor: Reward score tensor.
        """
        if self.rm_executor is not None:
            res = ray.get(self.rm_executor.submit_task.remote(data))
            data = data.union(res)

        return self.reward_manager(data, return_dict)


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], rm_executor: BatchExecutor = None
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        self.rm_executor = rm_executor

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path, self.rm_executor)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

        self.downsample_mode = config.actor_rollout_ref.rollout.multi_turn.downsample_mode
        self.max_samples_per_traj = config.actor_rollout_ref.rollout.multi_turn.max_samples_per_trajectory

        try:
            _cp_cfg = config.actor_rollout_ref.rollout.multi_turn.get("contextpilot", {})
            self._cp_enable_rpen_flag = bool(_cp_cfg.get("enable", False))
            _partial_cfg = _cp_cfg.get("partial_rollout", {}) if hasattr(_cp_cfg, "get") else {}
            self._cp_partial_rollout_enable = bool(_partial_cfg.get("enable", False))
            self._cp_snapshot_budget = int(_partial_cfg.get("snapshot_budget", 128) or 128)
            self._cp_entropy_top_k = int(_partial_cfg.get("entropy_top_k", 10) or 10)
            self._cp_max_concurrent_branches = int(
                _partial_cfg.get("max_concurrent_branches", 64) or 64
            )
        except Exception:
            self._cp_enable_rpen_flag = False
            self._cp_partial_rollout_enable = False
            self._cp_snapshot_budget = 128
            self._cp_entropy_top_k = 10
            self._cp_max_concurrent_branches = 64
        if self._cp_enable_rpen_flag:
            logger.info(
                "[AgentLoopWorker] ContextPilot R_pen ENABLED "
                "(-0.5 on tool failure or context/turn-limit violation)."
            )
        if self._cp_partial_rollout_enable:
            logger.info(
                "[AgentLoopWorker] ContextPilot query-global partial rollout enabled: "
                f"snapshot_budget={self._cp_snapshot_budget}, entropy_top_k={self._cp_entropy_top_k}, "
                f"max_concurrent_branches={self._cp_max_concurrent_branches}."
            )

    def _contextpilot_enable_rpen(self) -> bool:
        return getattr(self, "_cp_enable_rpen_flag", False)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _is_contextpilot_terminal(self, output: AgentLoopOutput) -> bool:
        if output.extra_fields.get("contextpilot_terminal", False):
            return True
        return not self._as_bool(output.extra_fields.get("is_snapshot", False))

    def _is_contextpilot_partial_branch(self, output: AgentLoopOutput) -> bool:
        return self._as_bool(output.extra_fields.get("contextpilot_is_partial_branch", False))

    async def _compute_agent_output_reward(
        self,
        output: AgentLoopOutput,
        kwargs: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """Compute terminal reward for one agent-loop output.

        This mirrors the reward path used later for padded training outputs,
        but is executed before snapshot downsampling so ContextPilot can assign
        subtree-mean rewards to intermediate snapshots using all terminal
        continuations, including partial-rollout branches.
        """
        if output.reward_score is not None:
            return {
                "reward_score": float(output.reward_score),
                "reward_extra_info": output.extra_fields.get("reward_extra_info", {}),
            }

        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if not torch.is_tensor(response_output["input_ids"]):
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            response_output["input_ids"] = torch.full(
                (1, self.config.actor_rollout_ref.rollout.response_length),
                int(pad_token_id),
                dtype=torch.long,
            )
            response_output["attention_mask"] = torch.zeros_like(response_output["input_ids"])
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch = TensorDict(
            {
                "prompts": prompt_output["input_ids"],
                "responses": response_output["input_ids"],
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
            },
            batch_size=1,
        )
        non_tensor_batch = {
            **{k: np.array([v]) for k, v in kwargs.items()},
            "__num_turns__": np.array([output.num_turns]),
        }
        for key, val in output.extra_fields.items():
            non_tensor_batch[key] = np.array([val], dtype=object)

        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is not None:
            existing_extra_info = non_tensor_batch.get("extra_info", np.array([{}], dtype=object))[0]
            if isinstance(existing_extra_info, dict):
                extra_info_copy = copy.deepcopy(existing_extra_info)
            else:
                extra_info_copy = {}
            msgs = list(raw_prompt) if isinstance(raw_prompt, list) else raw_prompt
            extra_info_copy["raw_prompt"] = msgs
            for flag_key in ("had_format_violation", "context_len_or_turn_exceeded"):
                if flag_key in output.extra_fields:
                    extra_info_copy[flag_key] = self._as_bool(output.extra_fields.get(flag_key))
            non_tensor_batch["extra_info"] = np.array([extra_info_copy], dtype=object)

        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        logger.info(f"[AgentLoopWorker][{request_id}] Computing terminal reward for ContextPilot subtree credit...")
        result = await self.reward_manager_worker.compute_score.remote(data)

        if self._contextpilot_enable_rpen():
            had_failure = self._as_bool(output.extra_fields.get("had_tool_failure", False))
            exceeded_budget = self._as_bool(output.extra_fields.get("context_len_or_turn_exceeded", False))
            if had_failure or exceeded_budget:
                r_pen = -0.5
                result["reward_score"] = float(result["reward_score"]) + r_pen
                rei = result.setdefault("reward_extra_info", {})
                if isinstance(rei, dict):
                    rei["contextpilot_r_pen"] = r_pen
                    rei["contextpilot_had_tool_failure"] = had_failure
                    rei["contextpilot_exceeded_budget"] = exceeded_budget
                logger.info(
                    f"[AgentLoopWorker][{request_id}] ContextPilot R_pen applied to terminal: -0.5 "
                    f"-> reward_score={result['reward_score']}"
                )
        return result

    async def _assign_contextpilot_subtree_rewards(
        self,
        trajectories: list[AgentLoopOutput],
        kwargs: dict[str, Any],
        request_id: str,
    ) -> None:
        enable_async_reward = (
            self.rm_executor is not None and self.config.reward_model.enable_resource_pool
        ) or not self.config.reward_model.enable
        if not enable_async_reward:
            return

        terminals = [output for output in trajectories if self._is_contextpilot_terminal(output)]
        if not terminals:
            for output in trajectories:
                if self._as_bool(output.extra_fields.get("is_snapshot", False)):
                    output.extra_fields["contextpilot_drop_from_training"] = True
                    rei = dict(output.extra_fields.get("reward_extra_info", {}) or {})
                    rei["contextpilot_subtree_terminal_count"] = 0
                    output.extra_fields["reward_extra_info"] = rei
            logger.warning(
                "[ContextPilot] Dropping snapshots with no terminal continuation instead of assigning a fallback reward."
            )
            return

        node_to_rewards: dict[str, list[float]] = {}
        terminal_results = await asyncio.gather(
            *[self._compute_agent_output_reward(output, kwargs, request_id) for output in terminals]
        )
        for output, result in zip(terminals, terminal_results, strict=True):
            reward = float(result["reward_score"])
            output.reward_score = reward
            output.extra_fields["reward_extra_info"] = result.get("reward_extra_info", {})
            for node_id in output.extra_fields.get("contextpilot_prefix_node_ids", []) or []:
                if node_id is None:
                    continue
                node_to_rewards.setdefault(str(node_id), []).append(reward)

        for output in trajectories:
            if not self._as_bool(output.extra_fields.get("is_snapshot", False)):
                continue
            node_id = output.extra_fields.get("contextpilot_node_id")
            rewards = node_to_rewards.get(str(node_id), []) if node_id is not None else []
            if rewards:
                assigned_reward = float(np.mean(rewards))
                terminal_count = len(rewards)
                output.reward_score = assigned_reward
            else:
                assigned_reward = None
                terminal_count = 0
                output.reward_score = None
                output.extra_fields["contextpilot_drop_from_training"] = True
            rei = dict(output.extra_fields.get("reward_extra_info", {}) or {})
            if assigned_reward is not None:
                rei["contextpilot_subtree_mean_reward"] = assigned_reward
            rei["contextpilot_subtree_terminal_count"] = terminal_count
            output.extra_fields["reward_extra_info"] = rei

    def _select_contextpilot_validation_trajectory(
        self, trajectories: list[AgentLoopOutput]
    ) -> list[AgentLoopOutput]:
        original_terminals = [
            output
            for output in trajectories
            if self._is_contextpilot_terminal(output) and not self._is_contextpilot_partial_branch(output)
        ]
        if original_terminals:
            return [original_terminals[-1]]
        terminals = [output for output in trajectories if self._is_contextpilot_terminal(output)]
        if terminals:
            return [terminals[-1]]
        return [trajectories[-1]]

    def _downsample_contextpilot_trajectories(
        self, trajectories: list[AgentLoopOutput]
    ) -> list[AgentLoopOutput]:
        if len(trajectories) <= self.max_samples_per_traj:
            return trajectories
        original_terminals = [
            output
            for output in trajectories
            if self._is_contextpilot_terminal(output) and not self._is_contextpilot_partial_branch(output)
        ]
        protected = [original_terminals[-1]] if original_terminals else [trajectories[-1]]
        protected_ids = {id(output) for output in protected}
        remaining_budget = max(0, self.max_samples_per_traj - len(protected))
        candidates = [output for output in trajectories if id(output) not in protected_ids]
        if remaining_budget <= 0:
            return protected[-self.max_samples_per_traj:]
        sample_count = min(remaining_budget, len(candidates))
        if sample_count == len(candidates):
            sampled = candidates
        else:
            positions = np.linspace(0, len(candidates) - 1, num=sample_count, dtype=int)
            sampled = [candidates[int(position)] for position in positions]
        return sampled + protected

    @staticmethod
    def _contextpilot_group_key(raw_run: dict[str, Any]) -> str:
        kwargs = raw_run["kwargs"]
        uid = kwargs.get("uid")
        if uid is not None:
            return str(uid)
        return str(raw_run["trajectory"].get("sample_index"))

    @staticmethod
    def _contextpilot_partial_training_samples(outputs: list[AgentLoopOutput]) -> list[AgentLoopOutput]:
        return [
            output
            for output in outputs
            if not bool(output.extra_fields.get("contextpilot_drop_from_training", False))
            and (
                bool(output.extra_fields.get("is_snapshot", False))
                or bool(output.extra_fields.get("contextpilot_terminal", False))
            )
        ]

    async def _cleanup_contextpilot_raw_runs(self, raw_runs: list[dict[str, Any]]) -> None:
        cleanup_tasks = []
        for raw_run in raw_runs:
            agent_loop_output = raw_run["agent_loop_output"]
            state = getattr(agent_loop_output, "contextpilot_state", None)
            cleanup = getattr(raw_run["agent_loop"], "_cleanup_contextpilot_state", None)
            if state is not None and callable(cleanup):
                cleanup_tasks.append(cleanup(state))
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)

    async def _run_contextpilot_query_group(
        self,
        raw_runs: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> list[_InternalAgentLoopOutput]:
        reward_pool: list[AgentLoopOutput] = []
        selected_by_run: dict[int, list[AgentLoopOutput]] = {}
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for raw_run in raw_runs:
            trajectories = list(raw_run["agent_loop_output"].trajectories)
            reward_pool.extend(trajectories)
            training_trajectories = [
                output
                for output in trajectories
                if not self._as_bool(output.extra_fields.get("contextpilot_drop_from_training", False))
            ]
            selected = self._downsample_contextpilot_trajectories(training_trajectories)
            selected_by_run[id(raw_run)] = selected
            for branch_point in raw_run["agent_loop_output"].contextpilot_branch_points:
                candidates.append((branch_point, raw_run))

        initial_snapshot_count = sum(len(outputs) for outputs in selected_by_run.values())
        remaining_budget = max(0, int(self._cp_snapshot_budget) - initial_snapshot_count)
        ranked = sorted(
            candidates,
            key=lambda item: float(item[0].get("sensitivity", float("-inf"))),
            reverse=True,
        )
        partial_branch_budget = remaining_budget // 5
        selected_candidates = ranked[:partial_branch_budget]

        candidate_entropy_deltas = [float(item[0].get("entropy_delta", 0.0)) for item in ranked]
        selected_entropy_deltas = [float(item[0].get("entropy_delta", 0.0)) for item in selected_candidates]
        candidate_context_deltas = [float(item[0].get("context_delta", 0.0)) for item in ranked]
        candidate_sensitivities = [float(item[0].get("sensitivity", 0.0)) for item in ranked]
        nonzero_candidate_entropy = sum(abs(value) > 1e-12 for value in candidate_entropy_deltas)
        nonzero_selected_entropy = sum(abs(value) > 1e-12 for value in selected_entropy_deltas)
        distinct_candidate_entropy = len({round(value, 9) for value in candidate_entropy_deltas})

        logger.info(
            "[ContextPilot] Query-global allocation: "
            f"initial_snapshots={initial_snapshot_count}, target={self._cp_snapshot_budget}, "
            f"remaining_budget={remaining_budget}, partial_branch_budget={partial_branch_budget}, "
            f"candidate_actions={len(ranked)}, selected_branches={len(selected_candidates)}, "
            f"nonzero_candidate_entropy={nonzero_candidate_entropy}, "
            f"nonzero_selected_entropy={nonzero_selected_entropy}, "
            f"distinct_candidate_entropy_1e-9={distinct_candidate_entropy}, "
            f"candidate_context_delta_range="
            f"[{min(candidate_context_deltas, default=0.0):.9f}, "
            f"{max(candidate_context_deltas, default=0.0):.9f}], "
            f"candidate_sensitivity_range="
            f"[{min(candidate_sensitivities, default=0.0):.9f}, "
            f"{max(candidate_sensitivities, default=0.0):.9f}], "
            f"selected_entropy_delta_range="
            f"[{min(selected_entropy_deltas, default=0.0):.9f}, "
            f"{max(selected_entropy_deltas, default=0.0):.9f}]."
        )

        semaphore = asyncio.Semaphore(max(1, int(self._cp_max_concurrent_branches)))

        async def run_branch(
            rank: int,
            branch_point: dict[str, Any],
            raw_run: dict[str, Any],
        ) -> tuple[int, dict[str, Any], dict[str, Any], list[AgentLoopOutput]]:
            try:
                async with semaphore:
                    outputs = await raw_run["agent_loop"]._run_contextpilot_partial_branch(
                        branch_point,
                        sampling_params,
                    )
            except Exception:
                logger.exception(
                    "[ContextPilot] Partial branch failed and will be excluded without aborting "
                    f"the query: rank={rank}, tools={branch_point.get('tool_names', [])}."
                )
                outputs = []
            return rank, branch_point, raw_run, outputs

        branch_results = await asyncio.gather(
            *[
                run_branch(rank, branch_point, raw_run)
                for rank, (branch_point, raw_run) in enumerate(selected_candidates)
            ]
        )

        partial_training_candidates: list[tuple[dict[str, Any], AgentLoopOutput]] = []
        per_branch_training_candidates: list[
            tuple[int, dict[str, Any], list[tuple[dict[str, Any], AgentLoopOutput]]]
        ] = []
        for rank, branch_point, raw_run, branch_outputs in branch_results:
            reward_pool.extend(branch_outputs)
            training_samples = self._contextpilot_partial_training_samples(branch_outputs)
            if not training_samples:
                continue
            tagged_samples: list[tuple[dict[str, Any], AgentLoopOutput]] = []
            for sample_index, budget_sample in enumerate(training_samples):
                budget_sample.extra_fields.update(
                    {
                        "contextpilot_budget_role": "partial",
                        "contextpilot_partial_sample_role": (
                            "snapshot"
                            if bool(budget_sample.extra_fields.get("is_snapshot", False))
                            else "terminal"
                        ),
                        "contextpilot_partial_sample_index": sample_index,
                        "contextpilot_branch_rank": rank,
                        "contextpilot_branch_sensitivity": float(branch_point.get("sensitivity", 0.0)),
                        "contextpilot_branch_context_delta": float(branch_point.get("context_delta", 0.0)),
                        "contextpilot_branch_entropy_delta": float(branch_point.get("entropy_delta", 0.0)),
                        "contextpilot_branch_tool_names": list(branch_point.get("tool_names", [])),
                    }
                )
                tagged_samples.append((raw_run, budget_sample))
                partial_training_candidates.append((raw_run, budget_sample))
            per_branch_training_candidates.append((rank, branch_point, tagged_samples))

        kept_partial_candidates: list[tuple[dict[str, Any], AgentLoopOutput]] = []
        for _, _, tagged_samples in per_branch_training_candidates:
            if len(kept_partial_candidates) >= remaining_budget:
                break
            kept_partial_candidates.append(tagged_samples[0])
        if len(kept_partial_candidates) < remaining_budget:
            for _, _, tagged_samples in per_branch_training_candidates:
                for tagged_sample in tagged_samples[1:]:
                    if len(kept_partial_candidates) >= remaining_budget:
                        break
                    kept_partial_candidates.append(tagged_sample)
                if len(kept_partial_candidates) >= remaining_budget:
                    break

        for raw_run, budget_sample in kept_partial_candidates:
            selected_by_run[id(raw_run)].append(budget_sample)
        if len(partial_training_candidates) > len(kept_partial_candidates):
            logger.info(
                "[ContextPilot] Capped merged partial training pool: "
                f"generated={len(partial_training_candidates)}, "
                f"kept={len(kept_partial_candidates)}, "
                f"query_target={self._cp_snapshot_budget}."
            )

        if reward_pool:
            first = raw_runs[0]
            await self._assign_contextpilot_subtree_rewards(
                reward_pool,
                first["kwargs"],
                first["request_id"],
            )

        finalized: list[_InternalAgentLoopOutput] = []
        for raw_run in raw_runs:
            selected_outputs = selected_by_run[id(raw_run)]
            kept_outputs = [
                output
                for output in selected_outputs
                if not self._as_bool(output.extra_fields.get("contextpilot_drop_from_training", False))
            ]
            dropped_count = len(selected_outputs) - len(kept_outputs)
            if dropped_count:
                logger.warning(
                    f"[ContextPilot] Dropped {dropped_count} selected snapshots without terminal descendants."
                )
            selected_by_run[id(raw_run)] = kept_outputs
            for output in selected_by_run[id(raw_run)]:
                output.extra_fields.setdefault("contextpilot_budget_role", "initial")
                output.extra_fields["contextpilot_query_initial_snapshot_count"] = initial_snapshot_count
                output.extra_fields["contextpilot_query_target_snapshot_count"] = int(self._cp_snapshot_budget)
                output.extra_fields["contextpilot_query_selected_branch_count"] = len(selected_candidates)
                output.extra_fields["contextpilot_query_partial_sample_count_before_cap"] = len(
                    partial_training_candidates
                )
                output.extra_fields["contextpilot_query_partial_sample_count_after_cap"] = len(
                    kept_partial_candidates
                )
            finalized.extend(
                await self._finalize_agent_loop_trajectories(
                    selected_by_run[id(raw_run)],
                    raw_run["trajectory"],
                    raw_run["kwargs"],
                    raw_run["request_id"],
                    rewards_assigned=True,
                    apply_trajectory_limit=False,
                )
            )
        return finalized

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        is_validation = bool(batch.meta_info.get("validate", False))
        requested_logprobs: bool | int = config.calculate_log_probs
        if self._cp_partial_rollout_enable and not is_validation:
            requested_logprobs = max(1, int(self._cp_entropy_top_k))
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=requested_logprobs,
        )

        if is_validation:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        batch_size = len(batch)

        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[AgentLoopWorker][Worker ID={worker_id}] Creating {batch_size} agent_loop tasks...")

        
        for i in range(batch_size):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop_initial(sampling_params, trajectory_info[i], **kwargs)
                )
            )
        initial_results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_runs = [result for result in initial_results if not isinstance(result, BaseException)]
        initial_errors = [result for result in initial_results if isinstance(result, BaseException)]
        flattened_outputs: list[_InternalAgentLoopOutput] = []
        try:
            if initial_errors:
                raise RuntimeError(
                    f"{len(initial_errors)} initial agent-loop task(s) failed; "
                    "successful task state will be cleaned before propagating the failure."
                ) from initial_errors[0]
            if self._cp_partial_rollout_enable and not is_validation:
                grouped: dict[str, list[dict[str, Any]]] = {}
                for raw_run in raw_runs:
                    grouped.setdefault(self._contextpilot_group_key(raw_run), []).append(raw_run)
                query_outputs = await asyncio.gather(
                    *[
                        self._run_contextpilot_query_group(query_runs, sampling_params)
                        for query_runs in grouped.values()
                    ]
                )
                flattened_outputs.extend(
                    output for outputs_for_query in query_outputs for output in outputs_for_query
                )
            else:
                for raw_run in raw_runs:
                    flattened_outputs.extend(
                        await self._finalize_agent_loop_trajectories(
                            list(raw_run["agent_loop_output"].trajectories),
                            raw_run["trajectory"],
                            raw_run["kwargs"],
                            raw_run["request_id"],
                        )
                    )
        finally:
            await self._cleanup_contextpilot_raw_runs(raw_runs)

        if not flattened_outputs:
            raise RuntimeError("Agent loop produced no valid trajectories for this worker batch.")
        processed_outputs = self._postprocess(flattened_outputs)

        logger.info(f"[AgentLoopWorker][Worker ID={worker_id}] _postprocess COMPLETED, batch has {len(processed_outputs)} samples")
        return processed_outputs

    async def _run_agent_loop_initial(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> dict[str, Any]:
        request_id = ray.get_runtime_context().get_worker_id()[:8]
        
        logger.info(f"[AgentLoopWorker][{request_id}] Entering _run_agent_loop, "
                    f"agent_name={agent_name}, step={trajectory['step']}, "
                    f"sample_index={trajectory['sample_index']}, rollout_n={trajectory['rollout_n']}")
        
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            loop_kwargs = dict(kwargs)
            loop_kwargs["__validate__"] = bool(trajectory.get("validate", False))
            trajectory_sampling_params = dict(sampling_params)
            trajectory_sampling_params["seed"] = contextpilot_trajectory_seed(trajectory)
            logger.info(
                f"[AgentLoopWorker][{request_id}] rollout seed={trajectory_sampling_params['seed']}"
            )
            agent_loop_output = await agent_loop.run(trajectory_sampling_params, **loop_kwargs)
            

        return {
            "agent_loop": agent_loop,
            "agent_loop_output": agent_loop_output,
            "trajectory": trajectory,
            "kwargs": kwargs,
            "request_id": request_id,
        }

    async def _finalize_agent_loop_trajectories(
        self,
        trajectories: list[AgentLoopOutput],
        trajectory: dict[str, Any],
        kwargs: dict[str, Any],
        request_id: str,
        *,
        rewards_assigned: bool = False,
        apply_trajectory_limit: bool = True,
    ) -> list[_InternalAgentLoopOutput]:
        if len(trajectories) == 0:
            return []

        if trajectory.get("validate", False):
            trajectories = self._select_contextpilot_validation_trajectory(trajectories)
        else:
            if self._contextpilot_enable_rpen() and not rewards_assigned:
                await self._assign_contextpilot_subtree_rewards(trajectories, kwargs, request_id)
            trajectories = [
                output
                for output in trajectories
                if not self._as_bool(output.extra_fields.get("contextpilot_drop_from_training", False))
            ]
            if not trajectories:
                return []
            if apply_trajectory_limit and len(trajectories) > self.max_samples_per_traj:
                if self._contextpilot_enable_rpen():
                    trajectories = self._downsample_contextpilot_trajectories(trajectories)
                elif self.downsample_mode == "last":
                    trajectories = trajectories[-self.max_samples_per_traj:]
                else:
                    prev_trajectories = random.sample(trajectories[:-1], self.max_samples_per_traj - 1)
                    trajectories = prev_trajectories + [trajectories[-1]]

        final_trajectory_reward = None
        padded_outputs: List[_InternalAgentLoopOutput] = []
        for output in trajectories[::-1]:

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = output.multi_modal_data.get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            enable_async_reward = (
                self.rm_executor is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable

            if output.reward_score is None and enable_async_reward:
                if final_trajectory_reward is None:
                    batch = TensorDict(
                        {
                            "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                            "responses": response_output["input_ids"],  # [1, response_length]
                            "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                            "input_ids": input_ids,  # [1, prompt_length + response_length]
                            "position_ids": position_ids,
                        },
                        batch_size=1,
                    )
                    non_tensor_batch = {
                        **{k: np.array([v]) for k, v in kwargs.items()},
                        "__num_turns__": np.array([output.num_turns]),
                    }
                    extra_fields = {}
                    for key, val in output.extra_fields.items():
                        extra_fields[key] = np.array([val], dtype=object)

                    non_tensor_batch.update(extra_fields)

                    raw_prompt = kwargs.get("raw_prompt")
                    if raw_prompt is not None:
                        if "extra_info" not in non_tensor_batch:
                            extra_info_copy = {}
                        else:
                            extra_info_copy = copy.deepcopy(non_tensor_batch["extra_info"][0])
                        msgs = list(raw_prompt) if isinstance(raw_prompt, list) else raw_prompt
                        extra_info_copy["raw_prompt"] = msgs
                        for flag_key in ("had_format_violation", "context_len_or_turn_exceeded"):
                            if flag_key in output.extra_fields:
                                extra_info_copy[flag_key] = self._as_bool(output.extra_fields.get(flag_key))
                        non_tensor_batch["extra_info"] = np.array([extra_info_copy], dtype=object)

                    data = DataProto(
                        batch=batch,
                        non_tensor_batch=non_tensor_batch,
                    )
                    logger.info(f"[AgentLoopWorker][{request_id}] Computing reward score asynchronously...")
                    result = await self.reward_manager_worker.compute_score.remote(data)
                    
                    
                    logger.info(f"[AgentLoopWorker][{request_id}] Reward score computed: {result}")
                    final_trajectory_reward = result

                    if self._contextpilot_enable_rpen():
                        had_failure = self._as_bool(output.extra_fields.get("had_tool_failure", False))
                        exceeded_budget = self._as_bool(output.extra_fields.get("context_len_or_turn_exceeded", False))
                        if had_failure or exceeded_budget:
                            r_pen = -0.5
                            final_trajectory_reward["reward_score"] = (
                                float(final_trajectory_reward["reward_score"]) + r_pen
                            )
                            rei = final_trajectory_reward.setdefault("reward_extra_info", {})
                            if isinstance(rei, dict):
                                rei["contextpilot_r_pen"] = r_pen
                                rei["contextpilot_had_tool_failure"] = had_failure
                                rei["contextpilot_exceeded_budget"] = exceeded_budget
                            logger.info(
                                f"[AgentLoopWorker][{request_id}] ContextPilot R_pen applied: -0.5 "
                                f"-> reward_score={final_trajectory_reward['reward_score']}"
                            )

                    output.reward_score = final_trajectory_reward["reward_score"]
                    output.extra_fields["reward_extra_info"] = final_trajectory_reward["reward_extra_info"]

                else:
                    output.reward_score = final_trajectory_reward["reward_score"]
                    output.extra_fields["reward_extra_info"] = final_trajectory_reward["reward_extra_info"]
            
            elif output.reward_score is not None:
                output.extra_fields["reward_extra_info"] = output.extra_fields.get("reward_extra_info", {})

            padded_outputs.append(
                _InternalAgentLoopOutput(
                    prompt_ids=prompt_output["input_ids"],
                    response_ids=response_output["input_ids"],
                    input_ids=input_ids,
                    position_ids=position_ids,
                    response_mask=response_mask,
                    attention_mask=attention_mask,
                    response_logprobs=response_logprobs,
                    multi_modal_inputs=multi_modal_inputs,
                    multi_modal_data=output.multi_modal_data,
                    reward_score=output.reward_score,
                    num_turns=output.num_turns,
                    metrics=output.metrics,
                    extra_fields={**kwargs, **output.extra_fields},
                )
            )
        return padded_outputs

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        contextpilot_reward_extra_keys = {
            "contextpilot_r_pen",
            "contextpilot_had_tool_failure",
            "contextpilot_exceeded_budget",
            "contextpilot_subtree_mean_reward",
            "contextpilot_subtree_terminal_count",
        }
        reward_extra_keys = sorted({key for info in reward_extra_infos for key in info.keys()} | contextpilot_reward_extra_keys)
        for key in reward_extra_keys:
            present_values = [info[key] for info in reward_extra_infos if key in info]
            if not present_values or all(isinstance(value, (int, float, bool, np.number)) for value in present_values):
                default_value = 0.0
            elif all(isinstance(value, str) for value in present_values):
                default_value = ""
            else:
                default_value = None
            non_tensor_batch[key] = np.array([info.get(key, default_value) for info in reward_extra_infos])

        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        
        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[AgentLoopWorker][{worker_id}] Final non_tensor_batch keys: {list(non_tensor_batch.keys())}")
        
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
        """
        self.config = config
        self.worker_group = worker_group
        self.rm_executor = None
        self.rm_micro_batch_size = None
        if rm_wg:

            def batch_fn(data_list: list[DataProto]) -> list[torch.Tensor]:
                n = len(data_list)
                if n == 0:
                    return []

                mb = rm_wg.world_size
                pad = (-n) % mb
                if pad > 0:
                    data_list = data_list + [data_list[-1]] * pad

                new_data_list = []
                for data in data_list:
                    temp_non_tensor_batch = {"__num_turns__": data.non_tensor_batch["__num_turns__"]}
                    temp_data = DataProto(batch=data.batch, non_tensor_batch=temp_non_tensor_batch)
                    new_data_list.append(temp_data)

                new_batch = DataProto.concat(new_data_list)
                out_data = rm_wg.compute_rm_score(new_batch)
                res = out_data.split(1)
                return res[:n]

            self.rm_executor = BatchExecutor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(batch_fn, rm_wg.world_size)

            self.rm_micro_batch_size = rm_wg.world_size

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size
        self.world_size = world_size

        rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.rm_executor)
            )

    @staticmethod
    def _align_non_tensor_batch_keys(outputs: list[DataProto]) -> None:
        """Make worker outputs concat-safe by padding missing non-tensor keys.

        DataProto.concat expects every shard to expose exactly the same
        non_tensor_batch keys. Agent-loop metadata can differ across samples
        because optional ContextPilot fields are only present on selected
        trajectories, so we normalize shards here before concatenation.
        """
        if not outputs:
            return
        all_keys = set()
        for output in outputs:
            all_keys.update(output.non_tensor_batch.keys())
        for output in outputs:
            batch_len = len(output)
            for key in all_keys:
                if key not in output.non_tensor_batch:
                    output.non_tensor_batch[key] = np.array([None] * batch_len, dtype=object)

    def _chunk_prompts_preserving_queries(self, prompts: DataProto) -> list[DataProto]:
        uids = prompts.non_tensor_batch.get("uid")
        if uids is None:
            return prompts.chunk(len(self.agent_loop_workers))

        groups: dict[str, list[int]] = {}
        for index, uid in enumerate(uids):
            groups.setdefault(str(uid), []).append(index)
        worker_count = min(len(self.agent_loop_workers), len(groups))
        if worker_count <= 0:
            return []

        buckets: list[list[int]] = [[] for _ in range(worker_count)]
        bucket_sizes = [0] * worker_count
        for indices in sorted(groups.values(), key=len, reverse=True):
            bucket_index = min(range(worker_count), key=lambda idx: bucket_sizes[idx])
            buckets[bucket_index].extend(indices)
            bucket_sizes[bucket_index] += len(indices)
        return [prompts[np.asarray(indices, dtype=np.int64)] for indices in buckets if indices]

    @staticmethod
    def _neutralize_world_size_padding(output: DataProto, original_size: int) -> None:
        if original_size >= len(output):
            return
        padding_slice = slice(original_size, len(output))
        output.batch["response_mask"][padding_slice] = 0
        if "rm_scores" in output.batch:
            output.batch["rm_scores"][padding_slice] = 0

        is_padding = np.zeros(len(output), dtype=bool)
        is_padding[padding_slice] = True
        output.non_tensor_batch["contextpilot_is_padding"] = is_padding

        if "uid" in output.non_tensor_batch:
            uids = np.asarray(output.non_tensor_batch["uid"], dtype=object).copy()
            for offset, row_index in enumerate(range(original_size, len(output))):
                uids[row_index] = f"__contextpilot_padding__{original_size}_{offset}"
            output.non_tensor_batch["uid"] = uids

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if self.rm_micro_batch_size and len(prompts) % self.rm_micro_batch_size != 0:
            raise ValueError(
                f"The length of prompts {len(prompts)} cannot divide the world size of rm_wg {self.rm_micro_batch_size}"
            )
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = self._chunk_prompts_preserving_queries(prompts)
        print(f"[AgentLoopManager] Starting to generate sequences, num_chunks: {len(chunkes)}...")
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers[: len(chunkes)], chunkes, strict=True)
            ]
        )
        print(f"[AgentLoopManager] Generated a batch of {len(outputs)} samples, ray.get COMPLETED...")
        self._align_non_tensor_batch_keys(outputs)

        output = DataProto.concat(outputs)
        is_validation = prompts.meta_info.get("validate", False)
        if not is_validation:
            original_size = len(output)
            output, pad_size = pad_dataproto_to_divisor(output, self.world_size)
            if pad_size > 0:
                self._neutralize_world_size_padding(output, original_size)
                logger.info(f"[AgentLoopManager] Padded {pad_size} samples to reach world_size {self.world_size}")

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
