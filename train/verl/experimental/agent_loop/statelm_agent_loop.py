import asyncio
import copy
import json
import logging
import os
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    MultiTrajectoryAgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@lru_cache(maxsize=4)
def _load_contextpilot_system_prompt(prompt_path: str) -> str:
    path = Path(prompt_path)
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Unable to load ContextPilot system prompt from {path}: {exc}") from exc
    if not prompt:
        raise RuntimeError(f"ContextPilot system prompt is empty: {path}")
    return prompt


def contextpilot_system_prompt_path() -> Path:
    configured = os.getenv("CONTEXTPILOT_SYSTEM_PROMPT_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "configs" / "contextpilot_system_prompt.txt"


def with_contextpilot_system_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(list(messages))
    system_prompt = _load_contextpilot_system_prompt(str(contextpilot_system_prompt_path()))
    for message in normalized:
        if message.get("role") == "system":
            message["content"] = system_prompt
            break
    else:
        normalized.insert(0, {"role": "system", "content": system_prompt})
    return normalized


def _tool_content_with_message_ids(content: Any, msg_id: int, assistant_msg_id: int) -> str:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = {"message": content}
    else:
        payload = copy.deepcopy(content)
    if not isinstance(payload, dict):
        payload = {"message": payload}
    payload["msg_id"] = msg_id
    payload["msg_id(invoking_assistant)"] = assistant_msg_id
    return json.dumps(payload, ensure_ascii=False)


def _tool_content_with_repetition_warning(content: Any, tool_name: str, consecutive_count: int) -> str:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = {"message": content}
    else:
        payload = copy.deepcopy(content)
    if not isinstance(payload, dict):
        payload = {"message": payload}
    payload["repetition_warning"] = (
        f"You have called the `{tool_name}` tool {consecutive_count} consecutive times. "
        "Do not call this tool again on the next turn. Stop repeating the same tool; "
        "choose a different tool and make concrete progress toward completing the task."
    )
    return json.dumps(payload, ensure_ascii=False)


def _tool_response_has_top_level_error(content: Any) -> bool:
    if not isinstance(content, str):
        return isinstance(content, dict) and "error" in content
    stripped = content.lstrip()
    if not stripped:
        return False
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return stripped.lower().startswith("error")
    return isinstance(payload, dict) and "error" in payload


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"

def strip_the_last_thinking_tag(response_ids: list[int]) -> list[int]:
    """
    Strip the last <think> ... </think> tag from the response_ids.
    
    ['<|im_start|>', 'assistant', 'Ċ', '<think>', 'ĊĊ', '</think>', 'ĊĊ']
    --> [151644, 77091, 198, 151667, 271, 151668, 271]
    
    So each time we find the last pattern appeared in the prompt_ids
    and remove the thinking tags: [151667, 271, 151668, 271]
    """
    


def render_context(
    conv_history: list[dict[str, Any]],
    memory_notes: dict[str, dict],
    deleted_msg_ids: set[int],
    simple_notes: Optional[dict[str, dict]] = None,
    summarized_msg_ids: Optional[dict[int, str]] = None,
    truncated_msg_ids: Optional[dict[int, str]] = None,
) -> list[dict[str, Any]]:
    """
    Build a rendered message-history view after applying context-editing operations.
    We also need to set the response_mask and response_logprobs to empty lists for the current turn.
    """
    conv_history_cp = copy.deepcopy(conv_history)
    memory_notes_cp = copy.deepcopy(memory_notes)
    simple_notes_cp = copy.deepcopy(simple_notes or {})
    deleted_msg_ids_cp = copy.deepcopy(deleted_msg_ids)
    summarized_msg_ids_cp = copy.deepcopy(summarized_msg_ids or {})
    truncated_msg_ids_cp = copy.deepcopy(truncated_msg_ids or {})
    
    stub_message = "Content has been deleted to save space."
    rendered_context: list[dict[str, Any]] = []

    if memory_notes_cp:
        memory_summary = (
            "\n\n<external_memory>\n## Available Memories\n"
            + "\n".join([f"- **{key}**: {data.get('summary', '')}" for key, data in memory_notes_cp.items()])
            + "\n</external_memory>"
        )
    else:
        memory_summary = (
            "\n\n<external_memory>\n## Available Memories\n"
            "No memories recorded.\n</external_memory>"
        )

    if simple_notes_cp:
        note_summary = (
            "\n\n<external_note>\n## Available Notes\n"
            + "\n".join([f"- **{key}**: {data.get('summary', '')}" for key, data in simple_notes_cp.items()])
            + "\n</external_note>"
        )
    else:
        note_summary = (
            "\n\n<external_note>\n## Available Notes\n"
            "No notes recorded.\n</external_note>"
        )
    external_context_summary = memory_summary + note_summary

    first_user_msg_seen = False
    for idx, msg in enumerate(conv_history_cp):
        role = msg.get("role")

        if role == "system":
            rendered_context.append({"role": "system", "content": msg["content"]})

        elif role == "user":
            if not first_user_msg_seen:
                text = msg["content"] + external_context_summary
                first_user_msg_seen = True
            else:
                text = msg["content"]
            rendered_context.append({"role": "user", "content": text})

        elif role == "assistant":
            msg_id = msg["msg_id"]
            tool_calls = msg.get("tool_calls", [])

            if msg_id in deleted_msg_ids_cp:
                replacement_text = stub_message
            elif msg_id in summarized_msg_ids_cp:
                replacement_text = summarized_msg_ids_cp[msg_id]
            elif msg_id in truncated_msg_ids_cp:
                replacement_text = truncated_msg_ids_cp[msg_id]
            else:
                replacement_text = None

            if replacement_text is not None:
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    stub_tool_calls = []
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        name = fn.get("name") or ""

                        stub_tool_calls.append({
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    {"message": replacement_text},
                                ),
                            },
                        })
                    rendered_context.append({
                        "role": "assistant",
                        "content": replacement_text,
                        "tool_calls": stub_tool_calls,
                    })
                else:
                    rendered_context.append({
                        "role": "assistant",
                        "content": replacement_text,
                    })
            else:
                assistant_content = msg["content"]
                assert len(assistant_content) == 1, "Expected single content block in assistant message."
                raw_text = assistant_content[0]["text"]
                cleaned_text = raw_text.strip()
                assistant_msg = {
                    "role": "assistant",
                    "content": (cleaned_text if cleaned_text else ""),
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = msg["tool_calls"]
                rendered_context.append(assistant_msg)

        elif role == "tool":
            msg_id = msg["msg_id"]
            msg_id_ia = msg["msg_id(invoking_assistant)"]
            
            original_content = msg["content"]
            if isinstance(original_content, str):
                try:
                    tool_result_content_cp = json.loads(original_content)
                except (json.JSONDecodeError, TypeError):
                    tool_result_content_cp = {"message": original_content}
            else:
                tool_result_content_cp = copy.deepcopy(original_content)
            if not isinstance(tool_result_content_cp, dict):
                tool_result_content_cp = {"message": tool_result_content_cp}
            
            tool_result_content_cp["msg_id"] = msg_id
            tool_result_content_cp["msg_id(invoking_assistant)"] = msg_id_ia
            
            if msg_id in deleted_msg_ids_cp:
                tool_name = msg.get("tool_name", "unknown")
                tool_result_content_cp = {
                    "msg_id": msg_id,
                    "msg_id(invoking_assistant)": msg_id_ia,
                    "status": "success",
                    "message": stub_message,
                    "original_tool": tool_name
                }
            elif msg_id in summarized_msg_ids_cp:
                tool_result_content_cp = {
                    "msg_id": msg_id,
                    "msg_id(invoking_assistant)": msg_id_ia,
                    "status": "success",
                    "message": summarized_msg_ids_cp[msg_id],
                    "original_tool": msg.get("tool_name", "unknown")
                }
            elif msg_id in truncated_msg_ids_cp:
                tool_result_content_cp = {
                    "msg_id": msg_id,
                    "msg_id(invoking_assistant)": msg_id_ia,
                    "status": "success",
                    "message": truncated_msg_ids_cp[msg_id],
                    "original_tool": msg.get("tool_name", "unknown")
                }
            
            rendered_context.append(
                {
                    "role": "tool",
                    "content": json.dumps(tool_result_content_cp, ensure_ascii=False),
                }
            )
    return rendered_context

def strip_the_last_think_tags(response_ids: list[int]) -> list[int]:
    """
    Strip the last <think> ... </think> tags from the response_ids.
    
    Target Logic:
    1. Find the last occurrence of the header: [151644, 77091, 198, 151667, 271]
       (<|im_start|>assistant\n<think>\n\n)
    2. Find the first occurrence of the footer AFTER that header: [151668, 271]
       (</think>\n\n)
    3. Remove everything from <think> (inclusive) to the footer (inclusive).
    """
    
    HEADER_IDS = [151644, 77091, 198, 151667, 271]
    FOOTER_IDS = [151668, 271]
    
    n = len(response_ids)
    h_len = len(HEADER_IDS)
    f_len = len(FOOTER_IDS)
    
    if n < h_len + f_len:
        return response_ids

    start_idx = -1
    for i in range(n - h_len, -1, -1):
        if (response_ids[i] == HEADER_IDS[0] and 
            response_ids[i+1] == HEADER_IDS[1] and 
            response_ids[i+2] == HEADER_IDS[2] and 
            response_ids[i+3] == HEADER_IDS[3] and 
            response_ids[i+4] == HEADER_IDS[4]):
            start_idx = i
            break
            
    if start_idx == -1:
        return response_ids

    search_pos = start_idx + h_len
    end_idx = -1
    
    for i in range(search_pos, n - f_len + 1):
        if (response_ids[i] == FOOTER_IDS[0] and 
            response_ids[i+1] == FOOTER_IDS[1]):
            end_idx = i
            break

    if end_idx != -1:
        keep_until = start_idx + 3
        
        resume_at = end_idx + f_len
        
        return response_ids[:keep_until] + response_ids[resume_at:]

    return response_ids


def strip_the_last_think_tags_with_span(response_ids: list[int]) -> tuple[list[int], Optional[tuple[int, int]]]:
    header_ids = [151644, 77091, 198, 151667, 271]
    footer_ids = [151668, 271]
    for start_idx in range(len(response_ids) - len(header_ids), -1, -1):
        if response_ids[start_idx : start_idx + len(header_ids)] != header_ids:
            continue
        search_pos = start_idx + len(header_ids)
        for footer_idx in range(search_pos, len(response_ids) - len(footer_ids) + 1):
            if response_ids[footer_idx : footer_idx + len(footer_ids)] == footer_ids:
                remove_start = start_idx + 3
                remove_end = footer_idx + len(footer_ids)
                return response_ids[:remove_start] + response_ids[remove_end:], (remove_start, remove_end)
        break
    return response_ids, None

class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        document_content: Optional[str] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.document_content = document_content

        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        self.tool_calls: list[FunctionCall] = []

        
        self.full_history: list[dict] = []
        self.deleted_msg_ids: set[int] = set()  # Track deleted message IDs
        self.msg_id_counter: int = 0  # Assign unique IDs to each message
        
        self.emission_views: list[list[int]] = []  # prompt_ids when each assistant turn started
        self.assistant_turn_boundaries: list[tuple[int, int]] = []  # (start_idx, end_idx) in response_ids
        
        self.trajectory_snapshots: list[dict[str, Any]] = []  # Store trajectory state before each delete
        
        self.had_delete_operation: bool = False

        self.had_tool_failure: bool = False

        self.had_format_violation: bool = False

        self.contextpilot_prefix_node_ids: list[str] = []
        self.contextpilot_branch_points: list[dict[str, Any]] = []
        self.contextpilot_pending_branch_points: list[dict[str, Any]] = []
        self.contextpilot_active_action: Optional[dict[str, Any]] = None
        self.contextpilot_branch_root_id: Optional[str] = None
        self.contextpilot_is_partial_branch: bool = False
        self.contextpilot_initial_uncertainty: Optional[float] = None
        self.context_len_or_turn_exceeded: bool = False
        self.contextpilot_length_boundary_count: int = 0
        self.contextpilot_auto_delete_count: int = 0
        self.last_model_tool_name: Optional[str] = None
        self.consecutive_model_tool_calls: int = 0

        self.sampling_seed: Optional[int] = None
        self.api_call_counter: int = 0

        self.notes: dict[str, dict] = {}
        self.memories: dict[str, dict] = self.notes
        self.simple_notes: dict[str, dict] = {}
        self.summarized_msg_ids: dict[int, str] = {}
        self.truncated_msg_ids: dict[int, str] = {}
        self.compressed_msg_ids: set[int] = set()
        
        self.doc_state_manager = None


try:
    from verl.trainer.contextpilot.constants import (
        OFFLOADING_TOOL_NAMES as _CP_OFFLOADING_TOOLS,
        MEMORY_WRITING_TOOL_NAMES as _CP_MEMORY_WRITING_TOOLS,
    )
except Exception:  # pragma: no cover
    _CP_OFFLOADING_TOOLS = ("deleteContext", "summarizeContext", "compressContext", "truncateContext")
    _CP_MEMORY_WRITING_TOOLS = ("memorize", "updateMemory", "note", "updateNote")


@register("statelm_tool_agent")
class StatelmToolAgentLoop(AgentLoopBase):
    _cp_retrieval_snapshot_tools = {"readChunk", "readMultiChunks"}

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Initializing StatelmToolAgentLoop")

        _mt_cfg = config.actor_rollout_ref.rollout.multi_turn
        _cp_cfg = _mt_cfg.get("contextpilot", {}) if hasattr(_mt_cfg, "get") else {}
        try:
            cls.contextpilot_enable = bool(_cp_cfg.get("enable", False))
        except Exception:
            cls.contextpilot_enable = False
        cls._cp_offloading_tools = set(_CP_OFFLOADING_TOOLS)
        cls._cp_memory_writing_tools = set(_CP_MEMORY_WRITING_TOOLS)
        cls._cp_ce_tools = cls._cp_offloading_tools | cls._cp_memory_writing_tools
        cls._cp_retrieval_snapshot_tools = {"readChunk", "readMultiChunks"}
        try:
            cls.contextpilot_budget_token_limit = int(_cp_cfg.get("budget_token_limit", 26000) or 26000)
        except Exception:
            cls.contextpilot_budget_token_limit = 26000
        try:
            cls.contextpilot_auto_delete_token_limit = int(
                _cp_cfg.get("auto_delete_token_limit", 24000) or 0
            )
        except Exception:
            cls.contextpilot_auto_delete_token_limit = 24000
        if cls.contextpilot_enable:
            logger.info(
                "[ContextPilot] enabled. Snapshot boundaries: "
                f"offloading={sorted(cls._cp_offloading_tools)} | "
                f"memory_writing={sorted(cls._cp_memory_writing_tools)} | "
                f"retrieval={sorted(cls._cp_retrieval_snapshot_tools)} | "
                f"auto_delete_token_limit={cls.contextpilot_auto_delete_token_limit} | "
                f"budget_token_limit={cls.contextpilot_budget_token_limit} | R_pen=-0.5"
            )
        _partial_cfg = _cp_cfg.get("partial_rollout", {}) if hasattr(_cp_cfg, "get") else {}
        try:
            cls.contextpilot_partial_rollout_enable = bool(_partial_cfg.get("enable", False)) and cls.contextpilot_enable
        except Exception:
            cls.contextpilot_partial_rollout_enable = False
        try:
            cls.contextpilot_context_weight = float(_partial_cfg.get("context_weight", 1.0))
        except Exception:
            cls.contextpilot_context_weight = 1.0
        try:
            cls.contextpilot_entropy_weight = float(_partial_cfg.get("entropy_weight", 1.0))
        except Exception:
            cls.contextpilot_entropy_weight = 1.0
        try:
            cls.contextpilot_entropy_token_window = int(_partial_cfg.get("entropy_token_window", 20) or 20)
        except Exception:
            cls.contextpilot_entropy_token_window = 20
        if cls.contextpilot_partial_rollout_enable:
            logger.info(
                "[ContextPilot] partial rollout enabled: "
                f"context_weight={cls.contextpilot_context_weight}, "
                f"entropy_weight={cls.contextpilot_entropy_weight}, "
                f"entropy_token_window={cls.contextpilot_entropy_token_window}, "
                "entropy_aggregation=topk_sum"
            )

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.min_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.get("min_assistant_turns", None)
        cls.below_min_turns_reward = config.actor_rollout_ref.rollout.multi_turn.get("below_min_turns_reward", -1.0)
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        logger.info("Initialized %d tools: %s", len(cls.tools), sorted(cls.tools))

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)
        
        cls.max_model_length = config.actor_rollout_ref.rollout.get("max_model_len", 8192)
        cls.max_response_length = config.actor_rollout_ref.rollout.multi_turn.get("single_turn_max_tokens", 2048)
        cls.exceed_length_penalty = config.actor_rollout_ref.rollout.multi_turn.get("exceed_length_penalty", -1.0)
        cls.model_type = config.actor_rollout_ref.rollout.multi_turn.get("model_type")

        cls.dump_trajectories_enabled = config.actor_rollout_ref.rollout.multi_turn.get("dump_trajectories_enabled", False)
        cls.dump_trajectories_dir = config.actor_rollout_ref.rollout.multi_turn.get("dump_trajectories_dir", "trajectories")
        cls.dump_trajectories_freq = config.actor_rollout_ref.rollout.multi_turn.get("dump_trajectories_freq", 1)
        cls._trajectory_dump_counter = 0
        if cls.dump_trajectories_enabled:
            os.makedirs(cls.dump_trajectories_dir, exist_ok=True)
            logger.info("Trajectory dumping enabled: %s", cls.dump_trajectories_dir)
        
        logger.info("[ContextPilot] Agent loop initialized")

    def _dump_trajectory(
        self,
        agent_data: AgentData,
        trajectories: list[AgentLoopOutput],
        request_id: str,
        **kwargs,
    ) -> None:
        """Dump trajectory data to a JSON file for debugging and analysis.
        
        Args:
            agent_data: The agent data containing full conversation history
            trajectories: List of trajectory outputs
            request_id: Unique request identifier
            **kwargs: Additional context from the rollout
        """
        if not self.dump_trajectories_enabled:
            return
        
        self.__class__._trajectory_dump_counter += 1
        if self.__class__._trajectory_dump_counter % self.dump_trajectories_freq != 0:
            return
        
        try:
            trajectory_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "user_turns": agent_data.user_turns,
                "assistant_turns": agent_data.assistant_turns,
                "had_delete_operation": agent_data.had_delete_operation,
                "contextpilot_auto_delete_count": agent_data.contextpilot_auto_delete_count,
                "deleted_msg_ids": list(agent_data.deleted_msg_ids),
            }
            
            messages_for_dump = []
            for msg in agent_data.full_history:
                msg_copy = dict(msg)
                if isinstance(msg_copy.get("content"), list):
                    text_parts = []
                    for block in msg_copy["content"]:
                        if isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    msg_copy["content"] = "\n".join(text_parts)
                messages_for_dump.append(msg_copy)
            trajectory_data["messages"] = messages_for_dump
            
            trajectory_outputs = []
            for i, traj in enumerate(trajectories):
                traj_info = {
                    "index": i,
                    "is_snapshot": traj.extra_fields.get("is_snapshot", False) if traj.extra_fields else False,
                    "num_turns": traj.num_turns,
                    "prompt_length": len(traj.prompt_ids) if traj.prompt_ids else 0,
                    "response_length": len(traj.response_ids) if traj.response_ids else 0,
                    "response_mask_sum": sum(traj.response_mask) if traj.response_mask else 0,
                }
                if traj.extra_fields:
                    for key in (
                        "contextpilot_terminal",
                        "contextpilot_reward_only_terminal",
                        "contextpilot_length_boundary",
                        "contextpilot_length_boundary_reason",
                        "contextpilot_branchable",
                        "contextpilot_auto_delete_context",
                        "contextpilot_forced_action",
                        "contextpilot_budget_role",
                        "contextpilot_drop_from_training",
                        "context_len_or_turn_exceeded",
                        "had_tool_failure",
                        "had_format_violation",
                    ):
                        if key in traj.extra_fields:
                            traj_info[key] = traj.extra_fields[key]
                if self.tokenizer:
                    try:
                        traj_info["prompt_text"] = self.tokenizer.decode(traj.prompt_ids, skip_special_tokens=False)
                        traj_info["response_text"] = self.tokenizer.decode(traj.response_ids, skip_special_tokens=False)
                    except Exception as e:
                        logger.warning(f"Failed to decode trajectory tokens: {e}")
                trajectory_outputs.append(traj_info)
            trajectory_data["trajectory_outputs"] = trajectory_outputs
            
            if "extra_info" in kwargs:
                extra_info = kwargs["extra_info"]
                if isinstance(extra_info, dict):
                    safe_extra = {}
                    for k, v in extra_info.items():
                        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                            safe_extra[k] = v
                    trajectory_data["extra_info"] = safe_extra
            
            filename = f"trajectory_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.dump_trajectories_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("[ContextPilot] Trajectory written to %s", filepath)
        except Exception as e:
            logger.warning("[ContextPilot] Failed to write trajectory: %s", e)

    def _contextpilot_sequence_uncertainty(
        self,
        token_entropies: Optional[list[float]],
    ) -> float:
        """Return ARPO-style summed top-k entropy over the initial token window."""
        window = max(1, int(getattr(self, "contextpilot_entropy_token_window", 20)))
        if token_entropies:
            vals = []
            for val in token_entropies[:window]:
                try:
                    vals.append(float(val))
                except Exception:
                    continue
            if vals:
                return float(sum(vals))
        return 0.0

    @staticmethod
    def _contextpilot_parent_checkpoint(agent_data: AgentData) -> dict[str, Any]:
        return {
            "prompt_ids": copy.deepcopy(agent_data.prompt_ids),
            "response_ids": copy.deepcopy(agent_data.response_ids),
            "response_mask": copy.deepcopy(agent_data.response_mask),
            "response_logprobs": copy.deepcopy(agent_data.response_logprobs),
            "turn_scores": copy.deepcopy(agent_data.turn_scores),
            "tool_rewards": copy.deepcopy(agent_data.tool_rewards),
            "assistant_turns": agent_data.assistant_turns,
            "full_history_len": len(agent_data.full_history),
            "msg_id_counter": agent_data.msg_id_counter,
            "had_format_violation": agent_data.had_format_violation,
            "contextpilot_prefix_node_ids": copy.deepcopy(agent_data.contextpilot_prefix_node_ids),
            "last_model_tool_name": agent_data.last_model_tool_name,
            "consecutive_model_tool_calls": agent_data.consecutive_model_tool_calls,
        }

    def _clone_doc_state_manager_for_branch(self, doc_state_manager):
        if doc_state_manager is None:
            return None
        from verl.tools.statelm_tools import DocStateManager

        cloned = DocStateManager.__new__(DocStateManager)
        cloned.tokenizer = getattr(doc_state_manager, "tokenizer", self.tokenizer)
        cloned.document_content = getattr(doc_state_manager, "document_content", "")
        cloned.encoded_doc = getattr(
            doc_state_manager,
            "encoded_doc",
            {"input_ids": [], "offset_mapping": []},
        )
        cloned.index = list(getattr(doc_state_manager, "index", []))
        cloned.keywords_searched = copy.deepcopy(getattr(doc_state_manager, "keywords_searched", set()))
        cloned.chunk_pointer = copy.deepcopy(getattr(doc_state_manager, "chunk_pointer", [-1, 0]))
        cloned.scan_mode = bool(getattr(doc_state_manager, "scan_mode", False))
        cloned.last_scanned_chunk_id = int(getattr(doc_state_manager, "last_scanned_chunk_id", -1))
        cloned._es = None
        cloned._es_index_name = getattr(doc_state_manager, "_es_index_name", "lc_agent_document")
        cloned._es_host = getattr(doc_state_manager, "_es_host", "http://localhost:9200")
        cloned._es_user = getattr(doc_state_manager, "_es_user", None)
        cloned._es_pass = getattr(doc_state_manager, "_es_pass", None)
        cloned._es_api_key = getattr(doc_state_manager, "_es_api_key", None)
        cloned._es_ca_cert = getattr(doc_state_manager, "_es_ca_cert", None)
        cloned._doc_id = getattr(doc_state_manager, "_doc_id", None)
        cloned._owns_doc_id = False
        cloned._orphan_doc_ids = set()
        return cloned

    def _clone_agent_data_for_branch(self, agent_data: AgentData, parent_node_id: str) -> AgentData:
        branch_data = AgentData(
            messages=copy.deepcopy(agent_data.messages),
            image_data=copy.deepcopy(agent_data.image_data),
            metrics={},
            request_id=f"{agent_data.request_id}_br_{uuid4().hex[:8]}",
            tools_kwargs=copy.deepcopy(agent_data.tools_kwargs),
            document_content=agent_data.document_content,
        )
        branch_data.prompt_ids = copy.deepcopy(agent_data.prompt_ids)
        branch_data.response_ids = copy.deepcopy(agent_data.response_ids)
        branch_data.response_mask = copy.deepcopy(agent_data.response_mask)
        branch_data.response_logprobs = copy.deepcopy(agent_data.response_logprobs)
        branch_data.turn_scores = copy.deepcopy(agent_data.turn_scores)
        branch_data.tool_rewards = copy.deepcopy(agent_data.tool_rewards)
        branch_data.user_turns = agent_data.user_turns
        branch_data.assistant_turns = agent_data.assistant_turns
        branch_data.full_history = copy.deepcopy(agent_data.full_history)
        branch_data.deleted_msg_ids = copy.deepcopy(agent_data.deleted_msg_ids)
        branch_data.msg_id_counter = agent_data.msg_id_counter
        branch_data.had_delete_operation = agent_data.had_delete_operation
        branch_data.had_tool_failure = agent_data.had_tool_failure
        branch_data.had_format_violation = agent_data.had_format_violation
        branch_data.memories = copy.deepcopy(agent_data.memories)
        branch_data.notes = branch_data.memories
        branch_data.simple_notes = copy.deepcopy(agent_data.simple_notes)
        branch_data.summarized_msg_ids = copy.deepcopy(agent_data.summarized_msg_ids)
        branch_data.truncated_msg_ids = copy.deepcopy(agent_data.truncated_msg_ids)
        branch_data.compressed_msg_ids = copy.deepcopy(agent_data.compressed_msg_ids)
        branch_data.doc_state_manager = self._clone_doc_state_manager_for_branch(agent_data.doc_state_manager)
        branch_data.contextpilot_prefix_node_ids = copy.deepcopy(agent_data.contextpilot_prefix_node_ids)
        branch_data.contextpilot_branch_points = []
        branch_data.contextpilot_pending_branch_points = []
        branch_data.contextpilot_active_action = None
        branch_data.contextpilot_branch_root_id = parent_node_id
        branch_data.contextpilot_is_partial_branch = True
        branch_data.contextpilot_initial_uncertainty = agent_data.contextpilot_initial_uncertainty
        branch_data.context_len_or_turn_exceeded = agent_data.context_len_or_turn_exceeded
        branch_data.contextpilot_length_boundary_count = agent_data.contextpilot_length_boundary_count
        branch_data.contextpilot_auto_delete_count = agent_data.contextpilot_auto_delete_count
        branch_data.last_model_tool_name = agent_data.last_model_tool_name
        branch_data.consecutive_model_tool_calls = agent_data.consecutive_model_tool_calls
        branch_data.sampling_seed = agent_data.sampling_seed
        branch_data.api_call_counter = agent_data.api_call_counter
        return branch_data

    def _clone_parent_agent_data_for_branch(
        self,
        agent_data: AgentData,
        checkpoint: dict[str, Any],
        branch_point_id: str,
    ) -> AgentData:
        parent = self._clone_agent_data_for_branch(agent_data, branch_point_id)
        parent.prompt_ids = copy.deepcopy(checkpoint["prompt_ids"])
        parent.response_ids = []
        self._start_training_segment(parent)
        parent.turn_scores = copy.deepcopy(checkpoint["turn_scores"])
        parent.tool_rewards = copy.deepcopy(checkpoint["tool_rewards"])
        parent.assistant_turns = int(checkpoint["assistant_turns"])
        parent.full_history = parent.full_history[: int(checkpoint["full_history_len"])]
        parent.msg_id_counter = int(checkpoint["msg_id_counter"])
        parent.had_format_violation = bool(checkpoint["had_format_violation"])
        parent.contextpilot_prefix_node_ids = copy.deepcopy(checkpoint["contextpilot_prefix_node_ids"])
        parent.last_model_tool_name = checkpoint.get("last_model_tool_name")
        parent.consecutive_model_tool_calls = int(checkpoint.get("consecutive_model_tool_calls", 0))
        parent.tool_calls = []
        return parent

    @staticmethod
    def _strip_qwen_thinking_from_state(agent_data: AgentData) -> None:
        old_prompt = agent_data.prompt_ids
        new_prompt, removed_span = strip_the_last_think_tags_with_span(old_prompt)
        if removed_span is None:
            return
        remove_start, remove_end = removed_span
        old_prompt_len = len(old_prompt)

        response_region_start = old_prompt_len - len(agent_data.response_mask)
        overlap_start = max(remove_start, response_region_start)
        overlap_end = min(remove_end, old_prompt_len)
        if overlap_start < overlap_end:
            rel_start = overlap_start - response_region_start
            rel_end = overlap_end - response_region_start
            del agent_data.response_mask[rel_start:rel_end]
            if agent_data.response_logprobs:
                del agent_data.response_logprobs[rel_start:rel_end]

        current_response_start = old_prompt_len - len(agent_data.response_ids)
        overlap_start = max(remove_start, current_response_start)
        overlap_end = min(remove_end, old_prompt_len)
        if overlap_start < overlap_end:
            rel_start = overlap_start - current_response_start
            rel_end = overlap_end - current_response_start
            del agent_data.response_ids[rel_start:rel_end]
        agent_data.prompt_ids = new_prompt

    def _start_training_segment(self, agent_data: AgentData) -> bool:
        overflow = max(0, len(agent_data.prompt_ids) - self.prompt_length)
        if overflow > self.response_length or len(agent_data.prompt_ids) > self.max_model_length:
            logger.error(
                "[ContextPilot] Full rollout history cannot fit the fixed training tensor: "
                f"history={len(agent_data.prompt_ids)}, prompt_length={self.prompt_length}, "
                f"response_length={self.response_length}, max_model_length={self.max_model_length}."
            )
            agent_data.context_len_or_turn_exceeded = True
            return False

        agent_data.response_ids = []
        agent_data.response_mask = [0] * overflow
        agent_data.response_logprobs = [0.0] * overflow
        if overflow:
            logger.info(
                "[ContextPilot] Started dynamic training segment with full-history "
                f"conditioning overflow={overflow}, trainable_capacity={self.response_length - overflow}."
            )
        return True

    async def _render_current_prompt(
        self,
        agent_data: AgentData,
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        messages = render_context(
            agent_data.full_history,
            agent_data.notes,
            agent_data.deleted_msg_ids,
            simple_notes=agent_data.simple_notes,
            summarized_msg_ids=agent_data.summarized_msg_ids,
            truncated_msg_ids=agent_data.truncated_msg_ids,
        )
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                tools=self.tool_schemas,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

    async def _contextpilot_auto_delete_target(self, agent_data: AgentData) -> Optional[int]:
        read_candidates: list[tuple[int, str]] = []
        search_candidates: list[tuple[int, str]] = []
        for message in agent_data.full_history:
            if message.get("role") != "tool":
                continue
            raw_msg_id = message.get("msg_id")
            try:
                msg_id = int(raw_msg_id)
            except (TypeError, ValueError):
                continue
            if msg_id in agent_data.deleted_msg_ids:
                continue
            tool_name = message.get("tool_name")
            content = message.get("content", "")
            content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if tool_name in {"readChunk", "readMultiChunks", "nextChunk"}:
                read_candidates.append((msg_id, content_text))
            elif tool_name == "searchEngine":
                search_candidates.append((msg_id, content_text))

        protected_read_id: Optional[int] = None
        if read_candidates and (len(read_candidates) > 1 or search_candidates):
            protected_read_id = max(msg_id for msg_id, _ in read_candidates)

        candidates = [
            candidate
            for candidate in read_candidates + search_candidates
            if candidate[0] != protected_read_id
        ]
        protected_candidate = next(
            (candidate for candidate in read_candidates if candidate[0] == protected_read_id),
            None,
        )
        if not candidates and protected_candidate is None:
            return None

        ranked = sorted(
            candidates,
            key=lambda item: (len(item[1]), -item[0]),
            reverse=True,
        )
        ranked = ranked[:8]
        if protected_candidate is not None:
            ranked.append(protected_candidate)
        assistant_msg_id = agent_data.msg_id_counter
        tool_msg_id = assistant_msg_id + 1
        tool_call_id = f"tool_call_{assistant_msg_id}"
        current_length = len(agent_data.prompt_ids)

        def _simulated_lengths() -> list[tuple[int, int]]:
            results: list[tuple[int, int]] = []
            for target_msg_id, _ in ranked:
                simulated_history = list(agent_data.full_history)
                simulated_history.extend(
                    [
                        {
                            "role": "assistant",
                            "content": [{"text": ""}],
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "deleteContext",
                                        "arguments": json.dumps({"msg_id": target_msg_id}),
                                    },
                                }
                            ],
                            "msg_id": assistant_msg_id,
                            "contextpilot_synthetic": True,
                        },
                        {
                            "role": "tool",
                            "content": json.dumps(
                                {
                                    "status": "success",
                                    "deleted_msg_id": target_msg_id,
                                    "deleted_role": "tool",
                                },
                                ensure_ascii=False,
                            ),
                            "tool_name": "deleteContext",
                            "msg_id": tool_msg_id,
                            "msg_id(invoking_assistant)": assistant_msg_id,
                            "tool_use_id": tool_call_id,
                            "contextpilot_synthetic": True,
                        },
                    ]
                )
                messages = render_context(
                    simulated_history,
                    agent_data.notes,
                    set(agent_data.deleted_msg_ids) | {target_msg_id},
                    simple_notes=agent_data.simple_notes,
                    summarized_msg_ids=agent_data.summarized_msg_ids,
                    truncated_msg_ids=agent_data.truncated_msg_ids,
                )
                rendered_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                )
                results.append((target_msg_id, len(rendered_ids)))
            return results

        simulated = await self.loop.run_in_executor(None, _simulated_lengths)
        reducing = [
            item
            for item in simulated
            if item[1] < current_length and item[0] != protected_read_id
        ]
        if not reducing and protected_read_id is not None:
            reducing = [
                item
                for item in simulated
                if item[1] < current_length and item[0] == protected_read_id
            ]
        if not reducing:
            logger.warning(
                "[ContextPilot] No candidate deleteContext action would reduce the rendered prompt; "
                f"current_length={current_length}, simulated={simulated}."
            )
            return None
        target_msg_id, estimated_length = min(reducing, key=lambda item: (item[1], item[0]))
        logger.info(
            "[ContextPilot] Selected productive auto-delete target: "
            f"msg_id={target_msg_id}, estimated_prompt_length={current_length}->{estimated_length}."
        )
        return target_msg_id

    async def _inject_auto_delete_context(
        self,
        agent_data: AgentData,
        *,
        reason: str,
        close_training_segment: bool,
    ) -> bool:
        target_msg_id = await self._contextpilot_auto_delete_target(agent_data)
        if target_msg_id is None:
            return False

        if close_training_segment and any(agent_data.response_mask):
            await self._emit_length_boundary(
                agent_data,
                reason="before_auto_deleteContext",
            )

        pre_delete_prompt_ids = await self._render_current_prompt(agent_data)
        agent_data.prompt_ids = pre_delete_prompt_ids
        assistant_msg_id = agent_data.msg_id_counter
        tool_msg_id = assistant_msg_id + 1
        tool_call_id = f"tool_call_{assistant_msg_id}"
        agent_data.msg_id_counter += 2
        agent_data.had_delete_operation = True
        agent_data.contextpilot_auto_delete_count += 1

        agent_data.full_history.append(
            {
                "role": "assistant",
                "content": [{"text": ""}],
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "deleteContext",
                            "arguments": json.dumps({"msg_id": target_msg_id}),
                        },
                    }
                ],
                "msg_id": assistant_msg_id,
                "contextpilot_synthetic": True,
            }
        )

        trainable_action = False
        if len(pre_delete_prompt_ids) <= self.max_model_length and self._start_training_segment(agent_data):
            assistant_sequence_ids = await self._render_current_prompt(
                agent_data,
                add_generation_prompt=False,
            )
            if assistant_sequence_ids[: len(pre_delete_prompt_ids)] == pre_delete_prompt_ids:
                action_token_ids = assistant_sequence_ids[len(pre_delete_prompt_ids) :]
                if action_token_ids and len(assistant_sequence_ids) <= self.max_model_length:
                    agent_data.prompt_ids = assistant_sequence_ids
                    agent_data.response_ids = list(action_token_ids)
                    agent_data.response_mask += [1] * len(action_token_ids)
                    agent_data.response_logprobs += [0.0] * len(action_token_ids)
                    await self._emit_length_boundary(
                        agent_data,
                        reason="auto_deleteContext",
                        tool_names=["deleteContext"],
                        auto_delete_context=True,
                        forced_action=True,
                    )
                    trainable_action = True
            if not trainable_action:
                logger.warning(
                    "[ContextPilot] Could not isolate a trainable token suffix for the synthetic "
                    "deleteContext action; applying it as emergency conditioning only."
                )
        else:
            logger.warning(
                "[ContextPilot] Pre-delete prompt is too long to train the synthetic action: "
                f"{len(pre_delete_prompt_ids)} > {self.max_model_length}. Applying emergency cleanup only."
            )

        agent_data.deleted_msg_ids.add(target_msg_id)
        result = {
            "status": "success",
            "deleted_msg_id": target_msg_id,
            "deleted_role": "tool",
        }
        agent_data.full_history.append(
            {
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
                "tool_name": "deleteContext",
                "msg_id": tool_msg_id,
                "msg_id(invoking_assistant)": assistant_msg_id,
                "tool_use_id": tool_call_id,
                "contextpilot_synthetic": True,
            }
        )

        agent_data.prompt_ids = await self._render_current_prompt(agent_data)
        logger.info(
            "[ContextPilot] Auto context recovery injected deleteContext: "
            f"target_msg_id={target_msg_id}, new_prompt_length={len(agent_data.prompt_ids)}, "
            f"count={agent_data.contextpilot_auto_delete_count}, reason={reason}."
        )
        return True

    async def _proactively_reclaim_context(
        self,
        agent_data: AgentData,
        *,
        token_limit: int,
        reason: str,
        close_training_segment: bool = True,
    ) -> bool:
        changed = False
        close_segment = close_training_segment
        while token_limit > 0 and len(agent_data.prompt_ids) > token_limit:
            previous_length = len(agent_data.prompt_ids)
            if not await self._inject_auto_delete_context(
                agent_data,
                reason=reason,
                close_training_segment=close_segment,
            ):
                logger.warning(
                    "[ContextPilot] Prompt exceeds the proactive auto-delete limit "
                    f"({previous_length} > {token_limit}), but no read/search result remains to delete."
                )
                break
            changed = True
            close_segment = False
            if len(agent_data.prompt_ids) >= previous_length:
                logger.warning(
                    "[ContextPilot] Auto delete did not reduce the rendered prompt "
                    f"({previous_length} -> {len(agent_data.prompt_ids)}); trying another target."
                )

        if changed:
            if len(agent_data.prompt_ids) > self.max_model_length:
                logger.error(
                    "[ContextPilot] Auto cleanup could not fit the rendered prompt in the model window: "
                    f"{len(agent_data.prompt_ids)} > {self.max_model_length}."
                )
                agent_data.context_len_or_turn_exceeded = True
                return False
            if not self._start_training_segment(agent_data):
                return False
        return changed

    async def _emit_length_boundary(
        self,
        agent_data: AgentData,
        *,
        reason: str,
        tool_names: Optional[list[str]] = None,
        auto_delete_context: bool = False,
        forced_action: bool = False,
    ) -> None:
        if not any(agent_data.response_mask):
            return

        node_id = uuid4().hex
        agent_data.contextpilot_prefix_node_ids.append(node_id)
        current_response_ids = copy.deepcopy(agent_data.response_ids)
        if current_response_ids and agent_data.prompt_ids[-len(current_response_ids) :] != current_response_ids:
            current_response_ids = []
        agent_data.trajectory_snapshots.append(
            {
                "prompt_ids": copy.deepcopy(agent_data.prompt_ids),
                "response_ids": current_response_ids,
                "response_mask": copy.deepcopy(agent_data.response_mask),
                "response_logprobs": copy.deepcopy(agent_data.response_logprobs),
                "num_turns": agent_data.assistant_turns,
                "extra_fields": {
                    "contextpilot_node_id": node_id,
                    "contextpilot_prefix_node_ids": copy.deepcopy(agent_data.contextpilot_prefix_node_ids),
                    "contextpilot_tool_names": list(tool_names or []),
                    "contextpilot_is_partial_branch": agent_data.contextpilot_is_partial_branch,
                    "contextpilot_branch_root_id": agent_data.contextpilot_branch_root_id,
                    "contextpilot_length_boundary": True,
                    "contextpilot_branchable": False,
                    "contextpilot_length_boundary_reason": reason,
                    "contextpilot_auto_delete_context": auto_delete_context,
                    "contextpilot_forced_action": forced_action,
                },
            }
        )
        self._start_training_segment(agent_data)
        agent_data.contextpilot_length_boundary_count += 1
        logger.info(
            "[ContextPilot] Emitted non-branchable length boundary "
            f"#{agent_data.contextpilot_length_boundary_count}: reason={reason}."
        )

    def _create_reward_only_terminal(
        self,
        agent_data: AgentData,
        *,
        terminal_is_partial: bool,
    ) -> AgentLoopOutput:
        return AgentLoopOutput(
            prompt_ids=copy.deepcopy(agent_data.prompt_ids[-self.prompt_length :]),
            response_ids=[],
            response_mask=[],
            response_logprobs=[],
            num_turns=agent_data.assistant_turns,
            metrics=agent_data.metrics,
            extra_fields={
                "is_snapshot": False,
                "contextpilot_terminal": True,
                "contextpilot_terminal_id": uuid4().hex,
                "contextpilot_prefix_node_ids": copy.deepcopy(agent_data.contextpilot_prefix_node_ids),
                "contextpilot_is_partial_branch": terminal_is_partial,
                "contextpilot_branch_root_id": agent_data.contextpilot_branch_root_id,
                "contextpilot_reward_only_terminal": True,
                "contextpilot_drop_from_training": True,
                "had_format_violation": agent_data.had_format_violation,
                "had_tool_failure": agent_data.had_tool_failure,
                "context_len_or_turn_exceeded": agent_data.context_len_or_turn_exceeded,
            },
        )

    def _collect_trajectory_outputs(self, agent_data: AgentData, terminal_is_partial: bool = False) -> list[AgentLoopOutput]:
        trajectories: list[AgentLoopOutput] = []

        for snapshot in agent_data.trajectory_snapshots:
            snapshot_output = self._create_trajectory_output(
                prompt_ids=snapshot["prompt_ids"],
                response_ids=snapshot["response_ids"],
                response_mask=snapshot["response_mask"],
                response_logprobs=snapshot["response_logprobs"],
                num_turns=snapshot["num_turns"],
                is_snapshot=True,
                metrics=agent_data.metrics,
                had_tool_failure=agent_data.had_tool_failure,
                extra_fields={
                    "had_format_violation": agent_data.had_format_violation,
                    "context_len_or_turn_exceeded": agent_data.context_len_or_turn_exceeded,
                    **snapshot.get("extra_fields", {}),
                },
            )
            if snapshot_output is not None:
                trajectories.append(snapshot_output)

        if not any(agent_data.response_mask):
            if len(trajectories) > 0:
                logger.info(
                    "[ContextPilot] Final response segment is empty; "
                    "adding a reward-only terminal for snapshot credit propagation."
                )
                trajectories.append(
                    self._create_reward_only_terminal(
                        agent_data,
                        terminal_is_partial=terminal_is_partial,
                    )
                )
                return trajectories
            logger.warning(
                "[ContextPilot] No valid trajectories were produced; returning an empty output."
            )
            return []

        if (self.min_assistant_turns is not None) and (agent_data.assistant_turns < self.min_assistant_turns):
            reward_score = self.below_min_turns_reward
        else:
            reward_score = None

        terminal_extra_fields = {
            "contextpilot_terminal": True,
            "contextpilot_terminal_id": uuid4().hex,
            "contextpilot_prefix_node_ids": copy.deepcopy(agent_data.contextpilot_prefix_node_ids),
            "contextpilot_is_partial_branch": terminal_is_partial,
            "contextpilot_branch_root_id": agent_data.contextpilot_branch_root_id,
            "had_format_violation": agent_data.had_format_violation,
            "context_len_or_turn_exceeded": agent_data.context_len_or_turn_exceeded,
        }
        last_traj_output = self._create_trajectory_output(
            prompt_ids=copy.deepcopy(agent_data.prompt_ids),
            response_ids=copy.deepcopy(agent_data.response_ids),
            response_mask=copy.deepcopy(agent_data.response_mask),
            response_logprobs=copy.deepcopy(agent_data.response_logprobs),
            num_turns=agent_data.assistant_turns,
            is_snapshot=False,
            reward_score=reward_score,
            metrics=agent_data.metrics,
            enforce_output=True,
            had_tool_failure=agent_data.had_tool_failure,
            extra_fields=terminal_extra_fields,
        )
        trajectories.append(last_traj_output)
        return trajectories

    async def _run_contextpilot_partial_branch(self, branch_point: dict[str, Any], sampling_params: dict[str, Any]) -> list[AgentLoopOutput]:
        branch_point_id = str(branch_point.get("branch_point_id") or branch_point.get("node_id") or uuid4().hex)
        branch_data = self._clone_agent_data_for_branch(branch_point["agent_data"], branch_point_id)
        original_doc_id = getattr(branch_data.doc_state_manager, "_doc_id", None) if branch_data.doc_state_manager else None
        state = AgentState.GENERATING
        try:
            while state != AgentState.TERMINATED:
                if state == AgentState.GENERATING:
                    state = await self._handle_generating_state(branch_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(branch_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(branch_data)
                elif state == AgentState.PENDING:
                    state = await self._handle_pending_state(branch_data, sampling_params)
                else:
                    state = AgentState.TERMINATED
            return self._collect_trajectory_outputs(branch_data, terminal_is_partial=True)
        finally:
            current_doc_id = getattr(branch_data.doc_state_manager, "_doc_id", None) if branch_data.doc_state_manager else None
            if branch_data.doc_state_manager:
                def _cleanup_branch_manager():
                    try:
                        if current_doc_id and current_doc_id != original_doc_id:
                            branch_data.doc_state_manager.clear_current_document()
                    finally:
                        branch_data.doc_state_manager.close()

                try:
                    await asyncio.wait_for(
                        self.loop.run_in_executor(
                            None,
                            _cleanup_branch_manager,
                        ),
                        timeout=30.0,
                    )
                except Exception as e:
                    logger.warning(f"Error clearing partial-branch state manager: {e}")

    async def _cleanup_contextpilot_state(self, agent_data: AgentData) -> None:
        if not agent_data.doc_state_manager:
            return

        def _cleanup_manager():
            try:
                agent_data.doc_state_manager.clear_current_document()
            finally:
                agent_data.doc_state_manager.close()

        try:
            await asyncio.wait_for(
                self.loop.run_in_executor(None, _cleanup_manager),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Elasticsearch cleanup timed out after 30 seconds")
        except Exception as e:
            logger.warning(f"Error clearing state manager: {e}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> MultiTrajectoryAgentLoopOutput:
        logger.debug("[ContextPilot] Starting agent loop")
        messages = with_contextpilot_system_prompt(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        document_content = kwargs.get("document_content", "")
        logger.debug("Document content length: %d", len(document_content) if document_content else 0)
        
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            document_content=document_content,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        agent_data.msg_id_counter = 0
        if sampling_params.get("seed") is not None:
            agent_data.sampling_seed = int(sampling_params["seed"]) & 0x7FFFFFFF
        for msg in messages:
            agent_data.full_history.append({
                "role": msg.get("role"),
                "content": msg.get("content"),
                "msg_id": agent_data.msg_id_counter
            })
            agent_data.msg_id_counter += 1
        
        if document_content:
            from verl.tools.statelm_tools import DocStateManager
            agent_data.doc_state_manager = DocStateManager(self.tokenizer, document_content)
        else:
            logger.warning("[ContextPilot] No document content was provided; using an empty document")
            agent_data.doc_state_manager = DocStateManager(self.tokenizer, " ")

        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    state = AgentState.TERMINATED

            trajectories = self._collect_trajectory_outputs(agent_data, terminal_is_partial=False)
            if not trajectories:
                await self._cleanup_contextpilot_state(agent_data)
                return MultiTrajectoryAgentLoopOutput(trajectories=[])

            if self.dump_trajectories_enabled:
                self._dump_trajectory(agent_data, trajectories, request_id, **kwargs)

            return MultiTrajectoryAgentLoopOutput(
                trajectories=trajectories,
                contextpilot_branch_points=list(agent_data.contextpilot_branch_points),
                contextpilot_state=agent_data,
            )
        except BaseException:
            await self._cleanup_contextpilot_state(agent_data)
            raise

    def _create_trajectory_output(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        num_turns: int,
        reward_score: float = None,
        is_snapshot: bool = False,
        metrics: Optional[dict[str, Any]] = None,
        enforce_output: bool = False,
        had_tool_failure: bool = False,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> Optional[AgentLoopOutput]:
        """
        Create an AgentLoopOutput from trajectory data.
        
        Args:
            prompt_ids: Full prompt token ids (includes response tokens)
            response_ids: Convenience copy of the latest assistant turn. The
                authoritative segment layout is ``response_mask`` because a
                tool observation may follow the latest assistant turn.
            response_mask: Mask for the logical response suffix of prompt_ids.
                Leading zeros may hold conditioning overflow from a long real
                prompt; ones mark generated tokens optimized by PPO.
            response_logprobs: Log probabilities aligned with response_mask.
            num_turns: Number of assistant turns in this trajectory
            reward_score: Reward score for this trajectory
            is_snapshot: Whether this is a snapshot trajectory
            metrics: Metrics for this trajectory
            enforce_output: Whether to enforce the output to be non-empty
        Returns:
            AgentLoopOutput for this trajectory, None if the trajectory is not valid
        """
        assert len(prompt_ids) <= self.max_model_length, (
            f"prompt_ids length {len(prompt_ids)} exceeds max_model_length {self.max_model_length}."
        )
        assert len(response_mask) <= len(prompt_ids), (
            f"response_mask length {len(response_mask)} exceeds full sequence length {len(prompt_ids)}."
        )
        if response_logprobs:
            assert len(response_logprobs) == len(response_mask), (
                f"response_logprobs length {len(response_logprobs)} must match "
                f"response_mask length {len(response_mask)}."
            )

        logical_prefix_len = len(prompt_ids) - len(response_mask)
        absolute_mask = [0] * logical_prefix_len + list(response_mask)
        if response_logprobs:
            absolute_logprobs = [0.0] * logical_prefix_len + list(response_logprobs)
        else:
            absolute_logprobs = []

        try:
            first_trainable = absolute_mask.index(1)
        except ValueError:
            first_trainable = len(prompt_ids)
        split_index = min(self.prompt_length, first_trainable)
        minimum_split = max(0, len(prompt_ids) - self.response_length)
        assert split_index >= minimum_split, (
            "Trainable response cannot fit without truncation: "
            f"sequence={len(prompt_ids)}, first_trainable={first_trainable}, "
            f"prompt_length={self.prompt_length}, response_length={self.response_length}."
        )

        trajectory_prompt_ids = prompt_ids[:split_index]
        trajectory_response_ids = prompt_ids[split_index:]
        response_mask = absolute_mask[split_index:]
        if absolute_logprobs:
            response_logprobs = absolute_logprobs[split_index:]
        else:
            response_logprobs = []

        assert len(trajectory_prompt_ids) <= self.prompt_length
        assert len(trajectory_response_ids) <= self.response_length
        assert trajectory_prompt_ids + trajectory_response_ids == prompt_ids
            
        return AgentLoopOutput(
            prompt_ids=trajectory_prompt_ids,
            response_ids=trajectory_response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else [],
            num_turns=num_turns,
            reward_score=reward_score,
            metrics=metrics if metrics is not None else {},
            extra_fields={
                "is_snapshot": is_snapshot,
                "had_tool_failure": had_tool_failure,
                **(extra_fields or {}),
            },
        )

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""

        messages = render_context(
            agent_data.full_history,
            agent_data.notes,
            agent_data.deleted_msg_ids,
            simple_notes=agent_data.simple_notes,
            summarized_msg_ids=agent_data.summarized_msg_ids,
            truncated_msg_ids=agent_data.truncated_msg_ids,
        )

        agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                tools=self.tool_schemas,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        if not self._start_training_segment(agent_data):
            return AgentState.TERMINATED
        
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""

        auto_delete_limit = int(getattr(self, "contextpilot_auto_delete_token_limit", 0) or 0)
        if (
            getattr(self, "contextpilot_enable", False)
            and auto_delete_limit > 0
            and len(agent_data.prompt_ids) > auto_delete_limit
        ):
            await self._proactively_reclaim_context(
                agent_data,
                token_limit=auto_delete_limit,
                reason="next rollout prompt exceeded the proactive token limit",
            )
            if agent_data.context_len_or_turn_exceeded:
                return AgentState.TERMINATED

        budget_limit = int(getattr(self, "contextpilot_budget_token_limit", 26000))
        if getattr(self, "contextpilot_enable", False) and len(agent_data.prompt_ids) > budget_limit:
            logger.warning(
                f"[ContextPilot] input length {len(agent_data.prompt_ids)} exceeds budget_token_limit "
                f"{budget_limit}; applying R_pen and terminating the trajectory."
            )
            agent_data.context_len_or_turn_exceeded = True
            if (
                agent_data.response_ids
                and agent_data.prompt_ids[-len(agent_data.response_ids) :] != agent_data.response_ids
            ):
                agent_data.response_ids = []
            return AgentState.TERMINATED

        response_tokens_left = self.response_length - len(agent_data.response_mask)
        model_tokens_left = self.max_model_length - len(agent_data.prompt_ids)
        if (
            getattr(self, "contextpilot_enable", False)
            and agent_data.response_mask
            and (
                response_tokens_left < self.max_response_length
                or model_tokens_left < self.max_response_length
            )
        ):
            await self._emit_length_boundary(agent_data, reason="next_full_assistant_turn")

        parent_checkpoint = None
        if (
            getattr(self, "contextpilot_partial_rollout_enable", False)
            and not agent_data.contextpilot_is_partial_branch
        ):
            parent_checkpoint = self._contextpilot_parent_checkpoint(agent_data)

        response_tokens_left = self.response_length - len(agent_data.response_mask)
        model_tokens_left = self.max_model_length - len(agent_data.prompt_ids)
        generation_token_budget = min(
            int(self.max_response_length),
            int(response_tokens_left),
            int(model_tokens_left),
        )
        if generation_token_budget <= 0:
            logger.warning(
                "[ContextPilot] No token budget remains for the next generation: "
                f"response_tokens_left={response_tokens_left}, "
                f"model_tokens_left={model_tokens_left}. Terminating the current segment."
            )
            return AgentState.TERMINATED

        with simple_timer("generate_sequences", agent_data.metrics):
            request_sampling_params = dict(sampling_params)
            request_sampling_params["max_tokens"] = generation_token_budget
            if agent_data.sampling_seed is not None:
                request_sampling_params["seed"] = (
                    agent_data.sampling_seed + agent_data.api_call_counter
                ) & 0x7FFFFFFF
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=request_sampling_params,
                image_data=agent_data.image_data,
            )
            agent_data.api_call_counter += 1

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        elif agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(agent_data.response_ids)

        current_uncertainty = self._contextpilot_sequence_uncertainty(output.token_entropies)
        if getattr(self, "contextpilot_partial_rollout_enable", False):
            if agent_data.contextpilot_initial_uncertainty is None:
                agent_data.contextpilot_initial_uncertainty = current_uncertainty
            if agent_data.contextpilot_pending_branch_points:
                initial_uncertainty = float(agent_data.contextpilot_initial_uncertainty or 0.0)
                for branch_point in agent_data.contextpilot_pending_branch_points:
                    entropy_delta = current_uncertainty - initial_uncertainty
                    branch_point["post_observation_uncertainty"] = current_uncertainty
                    branch_point["initial_uncertainty"] = initial_uncertainty
                    branch_point["entropy_delta"] = entropy_delta
                    branch_point["sensitivity"] = (
                        float(getattr(self, "contextpilot_context_weight", 1.0))
                        * float(branch_point.get("context_delta", 0.0))
                        + float(getattr(self, "contextpilot_entropy_weight", 1.0)) * entropy_delta
                    )
                    agent_data.contextpilot_branch_points.append(branch_point)
                agent_data.contextpilot_pending_branch_points = []

        text_response, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        management_tool_names = [tool_call.name for tool_call in agent_data.tool_calls if tool_call.name != "finish"]
        agent_data.contextpilot_active_action = None
        if parent_checkpoint is not None and management_tool_names:
            branch_point_id = uuid4().hex
            agent_data.contextpilot_active_action = {
                "branch_point_id": branch_point_id,
                "tool_names": copy.deepcopy(management_tool_names),
                "context_len_before": len(parent_checkpoint["prompt_ids"]),
                "agent_data": self._clone_parent_agent_data_for_branch(
                    agent_data,
                    parent_checkpoint,
                    branch_point_id,
                ),
            }

        tool_calls_formatted = []
        current_msg_id = agent_data.msg_id_counter
        for i, tc in enumerate(agent_data.tool_calls):
            tool_calls_formatted.append({
                "id": f"tool_call_{current_msg_id}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
            })
        agent_data.full_history.append({
            "role": "assistant",
            "content": [{"text": text_response}],
            "tool_calls": tool_calls_formatted,
            "msg_id": current_msg_id,
        })
        agent_data.msg_id_counter += 1

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            agent_data.had_format_violation = True
            logger.info("[ContextPilot] No tool call was emitted; terminating the trajectory")
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []

        tool_calls_to_process = agent_data.tool_calls[: self.max_parallel_calls]
        with simple_timer("tool_calls", agent_data.metrics):
            tool_responses = await self.loop.run_in_executor(
                None,
                lambda: [
                    self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
                    for tool_call in tool_calls_to_process
                ],
            )


        finish_tool_call = False
        editor_tool_call = False
        snapshot_only_tool_call = False
        retrieval_snapshot_tool_call = False
        delete_msg_ids = []
        tool_result_list = []
        assistant_msg_id = agent_data.msg_id_counter - 1
        boundary_tool_names: list[str] = []
        snapshot_node_id: Optional[str] = None
        for tool_response, tool_reward, tool_result_dict, tool_name in tool_responses:
            message = {"role": "tool", "content": tool_response.text or ""}

            if tool_name == agent_data.last_model_tool_name:
                agent_data.consecutive_model_tool_calls += 1
            else:
                agent_data.last_model_tool_name = tool_name
                agent_data.consecutive_model_tool_calls = 1
            if agent_data.consecutive_model_tool_calls >= 3 and tool_name != "finish":
                message["content"] = _tool_content_with_repetition_warning(
                    message["content"],
                    tool_name,
                    agent_data.consecutive_model_tool_calls,
                )
                logger.warning(
                    "[ContextPilot] Added repeated-tool warning: "
                    f"tool={tool_name}, consecutive_count={agent_data.consecutive_model_tool_calls}."
                )

            if getattr(self, "contextpilot_enable", False) and _tool_response_has_top_level_error(
                tool_response.text or ""
            ):
                agent_data.had_tool_failure = True

            if tool_name == 'finish':
                finish_tool_call = True
            elif tool_name == 'deleteContext':
                editor_tool_call = True
                agent_data.had_delete_operation = True
                if tool_result_dict and "deleted_msg_ids" in tool_result_dict:
                    delete_msg_ids.extend(tool_result_dict["deleted_msg_ids"])
            elif tool_name in ('truncateContext', 'summarizeContext', 'compressContext'):
                editor_tool_call = True
                agent_data.had_delete_operation = True
            elif getattr(self, "contextpilot_enable", False) and tool_name in getattr(self, "_cp_memory_writing_tools", set()):
                snapshot_only_tool_call = True
            elif getattr(self, "contextpilot_enable", False) and tool_name in getattr(
                self, "_cp_retrieval_snapshot_tools", set()
            ):
                retrieval_snapshot_tool_call = True
            if (
                tool_name in getattr(self, "_cp_ce_tools", set())
                or tool_name in getattr(self, "_cp_retrieval_snapshot_tools", set())
            ):
                boundary_tool_names.append(tool_name)

            current_msg_id = agent_data.msg_id_counter
            tool_result_list.append({
                "role": "tool",
                "content": message["content"],
                "tool_name": tool_name,
                "msg_id": current_msg_id,
                "msg_id(invoking_assistant)": assistant_msg_id
            })
            add_messages.append(
                {
                    "role": "tool",
                    "content": _tool_content_with_message_ids(
                        message["content"], current_msg_id, assistant_msg_id
                    ),
                }
            )
            agent_data.msg_id_counter += 1

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if finish_tool_call:
            return AgentState.TERMINATED

        is_contextpilot_boundary = bool(
            editor_tool_call or snapshot_only_tool_call or retrieval_snapshot_tool_call
        )
        should_emit_snapshot = is_contextpilot_boundary or (
            self.model_type == "qwen3" and not getattr(self, "contextpilot_enable", False)
        )
        if should_emit_snapshot:
            if getattr(self, "contextpilot_enable", False) and is_contextpilot_boundary:
                snapshot_node_id = uuid4().hex
                agent_data.contextpilot_prefix_node_ids.append(snapshot_node_id)

            def _create_snapshot():
                return {
                    "prompt_ids": copy.deepcopy(agent_data.prompt_ids),
                    "response_ids": copy.deepcopy(agent_data.response_ids),
                    "response_mask": copy.deepcopy(agent_data.response_mask),
                    "response_logprobs": copy.deepcopy(agent_data.response_logprobs),
                    "num_turns": agent_data.assistant_turns,
                    "extra_fields": {
                        "contextpilot_node_id": snapshot_node_id,
                        "contextpilot_prefix_node_ids": copy.deepcopy(agent_data.contextpilot_prefix_node_ids),
                        "contextpilot_tool_names": copy.deepcopy(boundary_tool_names),
                        "contextpilot_is_partial_branch": agent_data.contextpilot_is_partial_branch,
                        "contextpilot_branch_root_id": agent_data.contextpilot_branch_root_id,
                    } if snapshot_node_id is not None else {},
                }
            snapshot = await self.loop.run_in_executor(None, _create_snapshot)
            agent_data.trajectory_snapshots.append(snapshot)

            agent_data.response_mask = []
            agent_data.response_ids = []
            agent_data.response_logprobs = []

        if self.model_type == "qwen3" and not getattr(self, "contextpilot_enable", False):
            self._strip_qwen_thinking_from_state(agent_data)

        for msg_id in delete_msg_ids:
            agent_data.deleted_msg_ids.add(msg_id)

        tool_result_token_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        length_boundary_before_tool = False
        if not should_emit_snapshot and (
            len(agent_data.response_mask) + len(tool_result_token_ids) > self.response_length
            or len(agent_data.prompt_ids) + len(tool_result_token_ids) > self.max_model_length
        ):
            await self._emit_length_boundary(agent_data, reason="tool_observation")
            length_boundary_before_tool = True
        
        if agent_data.assistant_turns >= self.max_assistant_turns:
            logger.warning(
                f"[PROCESSING_TOOLS] assistant_turns (={agent_data.assistant_turns}) exceeds(>=) max_assistant_turns (={self.max_assistant_turns}), terminating without appending the tool result."
            )
            agent_data.context_len_or_turn_exceeded = True
            return AgentState.TERMINATED

        for tool_result in tool_result_list:
            agent_data.full_history.append(tool_result)

        starts_fresh_segment = should_emit_snapshot or length_boundary_before_tool
        rebuild_after_tool = is_contextpilot_boundary
        if rebuild_after_tool:
            candidate_prompt_ids = await self._render_current_prompt(agent_data)
        else:
            candidate_prompt_ids = agent_data.prompt_ids + tool_result_token_ids
        action_context_len_after = len(candidate_prompt_ids)
        auto_delete_limit = int(getattr(self, "contextpilot_auto_delete_token_limit", 0) or 0)
        needs_auto_delete = bool(
            getattr(self, "contextpilot_enable", False)
            and auto_delete_limit > 0
            and len(candidate_prompt_ids) > auto_delete_limit
        )
        if needs_auto_delete and not starts_fresh_segment:
            await self._emit_length_boundary(
                agent_data,
                reason="before_auto_deleteContext",
            )
            starts_fresh_segment = True

        agent_data.prompt_ids = candidate_prompt_ids
        if (
            needs_auto_delete
        ):
            await self._proactively_reclaim_context(
                agent_data,
                token_limit=auto_delete_limit,
                reason="tool observation pushed the next prompt over the proactive token limit",
                close_training_segment=False,
            )
            starts_fresh_segment = True
            if agent_data.context_len_or_turn_exceeded:
                return AgentState.TERMINATED
        if len(agent_data.prompt_ids) > self.max_model_length:
            logger.error(
                "[PROCESSING_TOOLS] Tool observation would exceed max_model_length: "
                f"{len(agent_data.prompt_ids)} > {self.max_model_length}. "
                "Terminating without exposing a truncated observation to the next rollout call."
            )
            agent_data.context_len_or_turn_exceeded = True
            return AgentState.TERMINATED

        if starts_fresh_segment:
            if not self._start_training_segment(agent_data):
                return AgentState.TERMINATED
        else:
            agent_data.response_mask += [0] * len(tool_result_token_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(tool_result_token_ids)

        active_action = agent_data.contextpilot_active_action
        if active_action is not None:
            context_len_before = max(1, int(active_action["context_len_before"]))
            active_action["context_delta"] = abs(
                (action_context_len_after - context_len_before) / context_len_before
            )
            active_action["context_len_after"] = action_context_len_after
            active_action["contextpilot_snapshot_node_id"] = snapshot_node_id
            agent_data.contextpilot_pending_branch_points.append(active_action)
            agent_data.contextpilot_active_action = None

        return AgentState.GENERATING


    def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        tool_name = ""
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            exec_kwargs = {}
            statelm_tool_names = {
                'plan', 'analyzeText', 'loadDocument', 'buildIndex', 'readChunk', 'readMultiChunks',
                'searchEngine', 'memorize', 'loadMemory', 'updateMemory', 'note', 'readNote', 'updateNote',
                'mergeNotes', 'checkBudget', 'getContextStats', 'deleteContext', 'truncateContext',
                'summarizeContext', 'compressContext', 'finish'
            }
            if tool_name in statelm_tool_names:
                exec_kwargs['agent_data'] = agent_data
                exec_kwargs['doc_state_manager'] = agent_data.doc_state_manager
                exec_kwargs['tokenizer'] = self.tokenizer
                exec_kwargs['tool_schemas'] = self.tool_schemas
                exec_kwargs['max_context_exp'] = 32000
                exec_kwargs['max_output_tokens'] = self.max_response_length
                exec_kwargs['max_turns'] = self.max_assistant_turns
            tool_execution_response, tool_reward, res = tool.execute(instance_id, tool_args, **exec_kwargs)
        
        except json.JSONDecodeError as e:
            return (
                ToolResponse(
                    text=f"Error: Invalid JSON in tool arguments: {e}",
                ),
                0.0,
                {},
                tool_name,
            )
        except Exception as e:
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
                tool_name,
            )
        finally:
            if tool and instance_id:
                tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        return ToolResponse(text=tool_response_text), tool_reward, res if res else {}, tool_name
