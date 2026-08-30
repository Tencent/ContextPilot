import asyncio
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    AsyncLLMServerManager,
    MultiTrajectoryAgentLoopOutput,
)
from verl.experimental.agent_loop.statelm_agent_loop import (
    AgentData,
    AgentState,
    StatelmToolAgentLoop,
    _tool_response_has_top_level_error,
    render_context,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, HermesToolParser
from verl.protocol import DataProto
from verl.tools.schemas import ToolResponse
from verl.tools.statelm_tools import BuildIndexTool, DocStateManager


def _output(*, snapshot: bool, terminal: bool = False, node_id: str | None = None) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        response_logprobs=[-0.5],
        num_turns=1,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "is_snapshot": snapshot,
            "contextpilot_terminal": terminal,
            "contextpilot_node_id": node_id,
            "contextpilot_prefix_node_ids": [node_id] if node_id else [],
        },
    )


def test_uncertainty_matches_arpo_topk_entropy_sum_without_logprob_fallback():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.contextpilot_entropy_token_window = 2

    entropy_value = loop._contextpilot_sequence_uncertainty([0.1, 0.3, 0.9])
    missing_entropy_value = loop._contextpilot_sequence_uncertainty(None)

    assert entropy_value == pytest.approx(0.4)
    assert missing_entropy_value == 0.0


def test_parent_checkpoint_resamples_selected_action_without_training_prefix_twice():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.prompt_length = 16
    loop.response_length = 8
    loop.max_model_length = 24
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [10, 11]
    data.response_ids = [11]
    data.response_mask = [1]
    data.response_logprobs = [-0.1]
    data.assistant_turns = 1
    data.full_history = [{"role": "user", "content": "q", "msg_id": 0}]
    data.msg_id_counter = 1
    checkpoint = loop._contextpilot_parent_checkpoint(data)

    data.prompt_ids += [20, 21]
    data.response_ids = [20, 21]
    data.response_mask += [1, 1]
    data.response_logprobs += [-0.2, -0.3]
    data.assistant_turns += 1
    data.full_history.append({"role": "assistant", "content": "action", "msg_id": 1})
    data.msg_id_counter += 1

    parent = loop._clone_parent_agent_data_for_branch(data, checkpoint, "bp")

    assert parent.prompt_ids == [10, 11]
    assert parent.response_ids == []
    assert parent.response_mask == []
    assert parent.response_logprobs == []
    assert parent.assistant_turns == 1
    assert len(parent.full_history) == 1
    assert parent.contextpilot_branch_root_id == "bp"


def test_long_partial_branch_prefix_is_conditioning_only():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.prompt_length = 8
    loop.response_length = 8
    loop.max_model_length = 16
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = list(range(12))
    data.response_ids = [10, 11]
    data.response_mask = [1, 1]
    data.response_logprobs = [-0.1, -0.2]
    data.full_history = [{"role": "user", "content": "q", "msg_id": 0}]
    data.msg_id_counter = 1
    checkpoint = loop._contextpilot_parent_checkpoint(data)

    parent = loop._clone_parent_agent_data_for_branch(data, checkpoint, "bp-long")

    assert parent.prompt_ids == list(range(12))
    assert parent.response_ids == []
    assert parent.response_mask == [0, 0, 0, 0]
    assert parent.response_logprobs == [0.0, 0.0, 0.0, 0.0]
    assert not any(parent.response_mask)


def test_retrieval_snapshot_boundaries_are_read_chunk_tools():
    assert StatelmToolAgentLoop._cp_retrieval_snapshot_tools == {"readChunk", "readMultiChunks"}


@pytest.mark.asyncio
async def test_hermes_parser_does_not_persist_terminal_eos_as_assistant_content():
    class FakeTokenizer:
        eos_token = "<|im_end|>"
        pad_token = None

        def decode(self, response_ids):
            return '<tool_call>{"name":"plan","arguments":{"strategy":"x"}}</tool_call><|im_end|>'

    content, calls = await HermesToolParser(FakeTokenizer()).extract_tool_calls([1, 2, 3])

    assert content == ""
    assert [call.name for call in calls] == ["plan"]


def test_tool_failure_detection_only_uses_top_level_error():
    assert _tool_response_has_top_level_error('{"error":"Index not built"}') is True
    assert _tool_response_has_top_level_error("Error when executing tool: boom") is True
    assert _tool_response_has_top_level_error(
        '{"retrieved_chunk":[{"content":"if (error) { handle(); }"}],"status":"success"}'
    ) is False


def test_branch_doc_clone_shares_immutable_document_encoding_without_retokenizing():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.tokenizer = object()
    encoded = {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (1, 2), (2, 3)]}
    chunk = {"chunk_id": 0, "content": "abc", "start_pos": 0, "end_pos": 3}
    manager = SimpleNamespace(
        tokenizer=loop.tokenizer,
        document_content="abc",
        encoded_doc=encoded,
        index=[chunk],
        keywords_searched={"a"},
        chunk_pointer=[0, 1],
        scan_mode=True,
        last_scanned_chunk_id=0,
        _es_index_name="idx",
        _es_host="http://es",
        _es_user=None,
        _es_pass=None,
        _es_api_key=None,
        _es_ca_cert=None,
        _doc_id="parent-doc",
    )

    cloned = loop._clone_doc_state_manager_for_branch(manager)

    assert cloned.encoded_doc is encoded
    assert cloned.index is not manager.index
    assert cloned.index[0] is chunk
    assert cloned._doc_id == "parent-doc"
    assert cloned._owns_doc_id is False


def test_build_index_swaps_atomically_and_cleans_previous_document():
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 4], "offset_mapping": [(0, 1), (1, 2), (2, 3), (3, 4)]}

    manager = DocStateManager(FakeTokenizer(), "abcd")
    manager.index = [{"chunk_id": 99, "content": "old", "start_pos": 0, "end_pos": 1}]
    manager._doc_id = "old-doc"
    deleted = []
    indexed = []
    manager._ensure_es_index = lambda: None
    manager._bulk_index_chunks = lambda *, doc_id, chunks: indexed.append((doc_id, list(chunks)))
    manager._delete_document = deleted.append
    tool = BuildIndexTool.__new__(BuildIndexTool)

    response, _, _ = tool.execute("i", {"chunk_size": 2, "overlap": 0}, doc_state_manager=manager)

    assert "error" not in response.text
    assert indexed and indexed[0][0] == manager._doc_id
    assert deleted == ["old-doc"]
    assert manager._doc_id != "old-doc"
    assert [chunk["content"] for chunk in manager.index] == ["ab", "cd"]


def test_build_index_failure_preserves_previous_in_memory_state():
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (1, 2)]}

    manager = DocStateManager(FakeTokenizer(), "ab")
    old_index = [{"chunk_id": 0, "content": "old", "start_pos": 0, "end_pos": 1}]
    manager.index = old_index
    manager._doc_id = "old-doc"
    cleaned = []
    manager._ensure_es_index = lambda: None

    def fail_bulk(*, doc_id, chunks):
        raise RuntimeError("bulk failed")

    manager._bulk_index_chunks = fail_bulk
    manager._delete_document = cleaned.append
    tool = BuildIndexTool.__new__(BuildIndexTool)

    response, _, _ = tool.execute("i", {"chunk_size": 1, "overlap": 0}, doc_state_manager=manager)

    assert "error" in response.text
    assert manager._doc_id == "old-doc"
    assert manager.index is old_index
    assert len(cleaned) == 1
    assert cleaned[0] != "old-doc"


def test_build_index_tracks_failed_temporary_cleanup_for_final_reaping():
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (1, 2)]}

    manager = DocStateManager(FakeTokenizer(), "ab")
    old_index = [{"chunk_id": 0, "content": "old", "start_pos": 0, "end_pos": 1}]
    manager.index = old_index
    manager._doc_id = "old-doc"
    manager._ensure_es_index = lambda: None

    def fail_bulk(*, doc_id, chunks):
        raise RuntimeError("bulk failed")

    def fail_cleanup(doc_id):
        raise RuntimeError(f"delete failed: {doc_id}")

    manager._bulk_index_chunks = fail_bulk
    manager._delete_document = fail_cleanup
    tool = BuildIndexTool.__new__(BuildIndexTool)

    response, _, _ = tool.execute("i", {"chunk_size": 1, "overlap": 0}, doc_state_manager=manager)

    assert "error" in response.text
    assert manager._doc_id == "old-doc"
    assert manager.index is old_index
    assert len(manager._orphan_doc_ids) == 1
    assert "old-doc" not in manager._orphan_doc_ids


@pytest.mark.asyncio
async def test_auto_delete_target_prefers_large_old_results_and_preserves_latest_read():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.loop = asyncio.get_running_loop()
    loop.tool_schemas = []
    loop.apply_chat_template_kwargs = {}

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))

        def apply_chat_template(self, messages, **kwargs):
            size = 100 + sum(len(str(message.get("content", ""))) for message in messages)
            return [1] * size

    loop.tokenizer = FakeTokenizer()
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.full_history = [
        {
            "role": "tool",
            "tool_name": "readChunk",
            "content": "R" * 1000,
            "msg_id": 2,
            "msg_id(invoking_assistant)": 1,
        },
        {
            "role": "tool",
            "tool_name": "searchEngine",
            "content": "S" * 800,
            "msg_id": 4,
            "msg_id(invoking_assistant)": 3,
        },
        {
            "role": "tool",
            "tool_name": "readMultiChunks",
            "content": "N" * 1200,
            "msg_id": 6,
            "msg_id(invoking_assistant)": 5,
        },
    ]
    data.msg_id_counter = 7

    def refresh_current_length():
        messages = render_context(data.full_history, {}, data.deleted_msg_ids)
        data.prompt_ids = loop.tokenizer.apply_chat_template(messages)

    refresh_current_length()
    assert await loop._contextpilot_auto_delete_target(data) == 2
    data.deleted_msg_ids.add(2)
    refresh_current_length()
    assert await loop._contextpilot_auto_delete_target(data) == 4
    data.deleted_msg_ids.add(4)
    refresh_current_length()
    assert await loop._contextpilot_auto_delete_target(data) == 6


@pytest.mark.asyncio
async def test_proactive_auto_delete_becomes_its_own_trainable_snapshot():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.prompt_length = 100
    loop.response_length = 100
    loop.max_model_length = 200
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [1] * 120
    data.response_ids = [1] * 10
    data.response_mask = [1] * 10
    data.response_logprobs = [-0.1] * 10
    data.full_history = [
        {"role": "user", "content": "q", "msg_id": 0},
        {"role": "assistant", "content": [{"text": "read"}], "tool_calls": [], "msg_id": 1},
        {
            "role": "tool",
            "content": "large read result",
            "tool_name": "readChunk",
            "msg_id": 2,
            "msg_id(invoking_assistant)": 1,
        },
        {"role": "assistant", "content": [{"text": "read again"}], "tool_calls": [], "msg_id": 3},
        {
            "role": "tool",
            "content": "another large read result",
            "tool_name": "readChunk",
            "msg_id": 4,
            "msg_id(invoking_assistant)": 3,
        },
    ]
    data.msg_id_counter = 5

    async def fake_render(self, agent_data, *, add_generation_prompt=True):
        if not add_generation_prompt and agent_data.full_history[-1].get("contextpilot_synthetic"):
            return [9] * 120 + [7, 8, 9]
        return [9] * (120 - 50 * len(agent_data.deleted_msg_ids))

    loop._render_current_prompt = MethodType(fake_render, loop)

    async def fake_target(self, agent_data):
        return 2 if 2 not in agent_data.deleted_msg_ids else None

    loop._contextpilot_auto_delete_target = MethodType(fake_target, loop)

    changed = await loop._proactively_reclaim_context(
        data,
        token_limit=80,
        reason="unit test overflow",
    )

    assert changed is True
    assert data.deleted_msg_ids == {2}
    assert data.contextpilot_auto_delete_count == 1
    assert data.had_delete_operation is True
    assert len(data.trajectory_snapshots) == 2
    preceding_snapshot, delete_snapshot = data.trajectory_snapshots
    assert preceding_snapshot["extra_fields"]["contextpilot_length_boundary_reason"] == "before_auto_deleteContext"
    assert preceding_snapshot["extra_fields"]["contextpilot_forced_action"] is False
    assert delete_snapshot["extra_fields"]["contextpilot_auto_delete_context"] is True
    assert delete_snapshot["extra_fields"]["contextpilot_forced_action"] is True
    assert delete_snapshot["extra_fields"]["contextpilot_branchable"] is False
    assert delete_snapshot["extra_fields"]["contextpilot_tool_names"] == ["deleteContext"]
    assert sum(delete_snapshot["response_mask"]) == 3
    assert delete_snapshot["response_ids"] == [7, 8, 9]
    assert len(data.prompt_ids) == 70
    assert data.response_mask == []
    assert data.response_logprobs == []
    assert data.full_history[-2]["contextpilot_synthetic"] is True
    assert data.full_history[-2]["content"] == [{"text": ""}]
    assert data.full_history[-2]["tool_calls"][0]["id"] == "tool_call_5"
    assert data.full_history[-2]["tool_calls"][0]["function"]["name"] == "deleteContext"
    assert data.full_history[-1]["tool_name"] == "deleteContext"
    assert "reason" not in data.full_history[-1]["content"]


@pytest.mark.asyncio
async def test_qwen_boundary_snapshot_is_saved_before_next_prompt_rebuild():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.loop = asyncio.get_running_loop()
    loop.contextpilot_enable = True
    loop.contextpilot_auto_delete_token_limit = 0
    loop.model_type = "qwen3"
    loop.max_parallel_calls = 1
    loop.max_assistant_turns = 100
    loop.prompt_length = 32
    loop.response_length = 32
    loop.max_model_length = 64
    loop._cp_memory_writing_tools = {"note"}
    loop._cp_retrieval_snapshot_tools = {"readChunk", "readMultiChunks"}
    loop._cp_ce_tools = {"deleteContext", "note"}
    loop.apply_chat_template_kwargs = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [50, 51]

    loop.tokenizer = FakeTokenizer()
    loop.tool_schemas = []

    def fake_call_tool(self, tool_call, tools_kwargs, agent_data):
        return ToolResponse(text='{"status":"success","results":[]}'), 0.0, {}, "readChunk"

    loop._call_tool = MethodType(fake_call_tool, loop)

    def unexpected_strip(agent_data):
        raise AssertionError("ContextPilot segments must remain append-only until the boundary rebuild")

    loop._strip_qwen_thinking_from_state = unexpected_strip

    async def fake_render(self, agent_data, *, add_generation_prompt=True):
        return [200, 300, 50, 51]

    loop._render_current_prompt = MethodType(fake_render, loop)
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [100, 200, 300]
    data.response_ids = [300]
    data.response_mask = [1]
    data.response_logprobs = [-0.1]
    data.assistant_turns = 1
    data.tool_calls = [FunctionCall(name="readChunk", arguments='{"chunk_id":0}')]
    data.full_history = [
        {"role": "user", "content": "q", "msg_id": 0},
        {
            "role": "assistant",
            "content": [{"text": ""}],
            "tool_calls": [],
            "msg_id": 1,
        },
    ]
    data.msg_id_counter = 2

    state = await loop._handle_processing_tools_state(data)

    assert state is AgentState.GENERATING
    assert data.trajectory_snapshots[0]["prompt_ids"] == [100, 200, 300]
    assert data.trajectory_snapshots[0]["response_mask"] == [1]
    assert data.prompt_ids == [200, 300, 50, 51]


@pytest.mark.asyncio
async def test_memory_write_rebuilds_prompt_and_incremental_tool_result_contains_message_ids():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.loop = asyncio.get_running_loop()
    loop.contextpilot_enable = True
    loop.contextpilot_auto_delete_token_limit = 0
    loop.model_type = "other"
    loop.max_parallel_calls = 1
    loop.max_assistant_turns = 100
    loop.prompt_length = 64
    loop.response_length = 64
    loop.max_model_length = 128
    loop._cp_memory_writing_tools = {"note"}
    loop._cp_retrieval_snapshot_tools = {"readChunk", "readMultiChunks"}
    loop._cp_ce_tools = {"deleteContext", "note"}
    loop.apply_chat_template_kwargs = {}
    rendered_note_keys = []

    class FakeTokenizer:
        seen_messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.seen_messages = messages
            return [40, 41, 42]

    loop.tokenizer = FakeTokenizer()
    loop.tool_schemas = []

    async def fake_render(self, agent_data, *, add_generation_prompt=True):
        rendered_note_keys.append(sorted(agent_data.simple_notes))
        return [9] * 12

    loop._render_current_prompt = MethodType(fake_render, loop)

    def fake_call_tool(self, tool_call, tools_kwargs, agent_data):
        agent_data.simple_notes["fact"] = {"summary": "s", "content": "c"}
        return ToolResponse(text='{"status":"success","key":"fact"}'), 0.0, {}, "note"

    loop._call_tool = MethodType(fake_call_tool, loop)
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [1, 2, 3]
    data.response_ids = [3]
    data.response_mask = [1]
    data.response_logprobs = [-0.1]
    data.assistant_turns = 1
    data.tool_calls = [FunctionCall(name="note", arguments='{"key":"fact"}')]
    data.full_history = [
        {"role": "user", "content": "q", "msg_id": 0},
        {
            "role": "assistant",
            "content": [{"text": ""}],
            "tool_calls": [],
            "msg_id": 1,
        },
    ]
    data.msg_id_counter = 2

    state = await loop._handle_processing_tools_state(data)

    assert state is AgentState.GENERATING
    assert rendered_note_keys == [["fact"]]
    assert data.prompt_ids == [9] * 12
    incremental_payload = loop.tokenizer.seen_messages[0]["content"]
    assert '"msg_id": 2' in incremental_payload
    assert '"msg_id(invoking_assistant)": 1' in incremental_payload


@pytest.mark.asyncio
async def test_query_global_budget_selects_top_remaining_candidates_concurrently():
    worker_class = AgentLoopWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.max_samples_per_traj = 8
    worker._cp_snapshot_budget = 12
    worker._cp_max_concurrent_branches = 8

    active = 0
    max_active = 0
    selected_scores = []

    class FakeLoop:
        async def _run_contextpilot_partial_branch(self, branch_point, sampling_params):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            selected_scores.append(branch_point["sensitivity"])
            await asyncio.sleep(0.01)
            active -= 1
            node_id = f"partial-{branch_point['sensitivity']}"
            return [_output(snapshot=True, node_id=node_id), _output(snapshot=False, terminal=True)]

    async def fake_finalize(self, trajectories, *args, **kwargs):
        return list(trajectories)

    async def fake_assign_rewards(self, trajectories, kwargs, request_id):
        for trajectory in trajectories:
            trajectory.reward_score = 1.0

    worker._finalize_agent_loop_trajectories = MethodType(fake_finalize, worker)
    worker._assign_contextpilot_subtree_rewards = MethodType(fake_assign_rewards, worker)

    def raw_run(uid, scores):
        loop = FakeLoop()
        initial = _output(snapshot=False, terminal=True)
        return {
            "agent_loop": loop,
            "agent_loop_output": MultiTrajectoryAgentLoopOutput(
                trajectories=[initial],
                contextpilot_branch_points=[
                    {"sensitivity": score, "context_delta": score / 10, "entropy_delta": score / 20}
                    for score in scores
                ],
            ),
            "trajectory": {"validate": False},
            "kwargs": {"uid": uid},
            "request_id": uid,
        }

    outputs = await worker._run_contextpilot_query_group(
        [raw_run("q", [0.1, 0.9]), raw_run("q", [0.5, 0.7])],
        {},
    )

    assert sorted(selected_scores, reverse=True) == [0.9, 0.7]
    assert len(outputs) == 6
    assert max_active > 1


@pytest.mark.asyncio
async def test_partial_cap_keeps_at_least_one_sample_from_each_completed_branch():
    worker_class = AgentLoopWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.max_samples_per_traj = 8
    worker._cp_snapshot_budget = 11
    worker._cp_max_concurrent_branches = 8

    class FakeLoop:
        async def _run_contextpilot_partial_branch(self, branch_point, sampling_params):
            count = 9 if branch_point["sensitivity"] == 0.9 else 2
            return [
                _output(snapshot=True, node_id=f"{branch_point['sensitivity']}-{index}")
                for index in range(count)
            ]

    async def fake_finalize(self, trajectories, *args, **kwargs):
        return list(trajectories)

    async def fake_assign_rewards(self, trajectories, kwargs, request_id):
        for trajectory in trajectories:
            trajectory.reward_score = 1.0

    worker._finalize_agent_loop_trajectories = MethodType(fake_finalize, worker)
    worker._assign_contextpilot_subtree_rewards = MethodType(fake_assign_rewards, worker)
    raw_run = {
        "agent_loop": FakeLoop(),
        "agent_loop_output": MultiTrajectoryAgentLoopOutput(
            trajectories=[_output(snapshot=False, terminal=True)],
            contextpilot_branch_points=[
                {"sensitivity": 0.9, "context_delta": 0.0, "entropy_delta": 0.0},
                {"sensitivity": 0.8, "context_delta": 0.0, "entropy_delta": 0.0},
            ],
        ),
        "trajectory": {"validate": False},
        "kwargs": {"uid": "q"},
        "request_id": "q",
    }

    outputs = await worker._run_contextpilot_query_group([raw_run], {})
    partial_ranks = [
        output.extra_fields.get("contextpilot_branch_rank")
        for output in outputs
        if output.extra_fields.get("contextpilot_budget_role") == "partial"
    ]

    assert len(outputs) == 11
    assert 0 in partial_ranks
    assert 1 in partial_ranks


@pytest.mark.asyncio
async def test_failed_partial_branch_does_not_abort_other_branches():
    worker_class = AgentLoopWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.max_samples_per_traj = 8
    worker._cp_snapshot_budget = 11
    worker._cp_max_concurrent_branches = 8

    class FakeLoop:
        async def _run_contextpilot_partial_branch(self, branch_point, sampling_params):
            if branch_point["sensitivity"] == 0.9:
                raise RuntimeError("branch failed")
            return [_output(snapshot=True, node_id="survivor")]

    async def fake_finalize(self, trajectories, *args, **kwargs):
        return list(trajectories)

    async def fake_assign_rewards(self, trajectories, kwargs, request_id):
        for trajectory in trajectories:
            trajectory.reward_score = 1.0

    worker._finalize_agent_loop_trajectories = MethodType(fake_finalize, worker)
    worker._assign_contextpilot_subtree_rewards = MethodType(fake_assign_rewards, worker)
    raw_run = {
        "agent_loop": FakeLoop(),
        "agent_loop_output": MultiTrajectoryAgentLoopOutput(
            trajectories=[_output(snapshot=False, terminal=True)],
            contextpilot_branch_points=[
                {"sensitivity": 0.9, "context_delta": 0.0, "entropy_delta": 0.0},
                {"sensitivity": 0.8, "context_delta": 0.0, "entropy_delta": 0.0},
            ],
        ),
        "trajectory": {"validate": False},
        "kwargs": {"uid": "q"},
        "request_id": "q",
    }

    outputs = await worker._run_contextpilot_query_group([raw_run], {})

    assert any(output.extra_fields.get("contextpilot_branch_rank") == 1 for output in outputs)


def test_partial_branch_contributes_all_post_branch_snapshots_and_terminal():
    first = _output(snapshot=True, node_id="first")
    later = _output(snapshot=True, node_id="later")
    terminal = _output(snapshot=False, terminal=True)

    selected = AgentLoopWorker._contextpilot_partial_training_samples([first, later, terminal])

    assert selected == [first, later, terminal]


def test_reward_only_terminal_propagates_credit_but_is_not_a_training_sample():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.max_model_length = 40960
    loop.prompt_length = 28672
    loop.response_length = 12288
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [10, 11, 12]
    data.contextpilot_prefix_node_ids = ["snapshot-node"]
    data.trajectory_snapshots = [
        {
            "prompt_ids": [10, 11, 12],
            "response_ids": [12],
            "response_mask": [1],
            "response_logprobs": [-0.1],
            "num_turns": 1,
            "extra_fields": {
                "contextpilot_node_id": "snapshot-node",
                "contextpilot_prefix_node_ids": ["snapshot-node"],
            },
        }
    ]
    data.response_ids = []
    data.response_mask = []
    data.response_logprobs = []
    data.assistant_turns = 1
    data.context_len_or_turn_exceeded = True

    outputs = loop._collect_trajectory_outputs(data)

    assert len(outputs) == 2
    snapshot, reward_terminal = outputs
    assert snapshot.extra_fields["is_snapshot"] is True
    assert reward_terminal.extra_fields["contextpilot_terminal"] is True
    assert reward_terminal.extra_fields["contextpilot_reward_only_terminal"] is True
    assert reward_terminal.extra_fields["contextpilot_drop_from_training"] is True
    assert reward_terminal.extra_fields["contextpilot_prefix_node_ids"] == ["snapshot-node"]
    assert reward_terminal.response_ids == []
    assert AgentLoopWorker._contextpilot_partial_training_samples(outputs) == [snapshot]


@pytest.mark.asyncio
async def test_turn_limit_terminal_is_materialized_as_padding_and_receives_r_pen():
    worker_class = AgentLoopWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(prompt_length=16, response_length=8),
        ),
    )
    worker._cp_enable_rpen_flag = True

    class FakeTokenizer:
        padding_side = "left"
        pad_token_id = 0
        eos_token_id = 2

        def pad(self, inputs, *, padding, max_length, return_tensors, return_attention_mask):
            ids = list(inputs["input_ids"])
            if not ids:
                return {"input_ids": [], "attention_mask": []}
            pad_count = max_length - len(ids)
            if self.padding_side == "left":
                padded = [self.pad_token_id] * pad_count + ids
                mask = [0] * pad_count + [1] * len(ids)
            else:
                padded = ids + [self.pad_token_id] * pad_count
                mask = [1] * len(ids) + [0] * pad_count
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }

    class FakeRemoteMethod:
        async def remote(self, data):
            assert data.batch["responses"].shape == (1, 8)
            assert int(data.batch["attention_mask"][:, -8:].sum()) == 0
            return {"reward_score": 0.0, "reward_extra_info": {}}

    worker.tokenizer = FakeTokenizer()
    worker.reward_manager_worker = SimpleNamespace(compute_score=FakeRemoteMethod())
    terminal = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[],
        response_mask=[],
        response_logprobs=[],
        num_turns=100,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "contextpilot_terminal": True,
            "contextpilot_reward_only_terminal": True,
            "contextpilot_drop_from_training": True,
            "context_len_or_turn_exceeded": True,
        },
    )

    result = await worker._compute_agent_output_reward(terminal, {}, "request")

    assert result["reward_score"] == -0.5
    assert result["reward_extra_info"]["contextpilot_r_pen"] == -0.5
    assert result["reward_extra_info"]["contextpilot_exceeded_budget"] is True


@pytest.mark.asyncio
async def test_partial_branch_count_is_fifth_remaining_budget_without_refill():
    worker_class = AgentLoopWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.max_samples_per_traj = 8
    worker._cp_snapshot_budget = 128
    worker._cp_max_concurrent_branches = 64
    selected_scores = []

    class FakeLoop:
        async def _run_contextpilot_partial_branch(self, branch_point, sampling_params):
            selected_scores.append(branch_point["sensitivity"])
            node_id = f"partial-{branch_point['sensitivity']}"
            return [
                _output(snapshot=True, node_id=node_id),
                _output(snapshot=True, node_id=f"{node_id}-later"),
                _output(snapshot=False, terminal=True),
            ]

    async def fake_finalize(self, trajectories, *args, **kwargs):
        return list(trajectories)

    async def fake_assign_rewards(self, trajectories, kwargs, request_id):
        for trajectory in trajectories:
            trajectory.reward_score = 1.0

    worker._finalize_agent_loop_trajectories = MethodType(fake_finalize, worker)
    worker._assign_contextpilot_subtree_rewards = MethodType(fake_assign_rewards, worker)

    raw_runs = []
    for run_index in range(31):
        scores = list(range(100)) if run_index == 0 else []
        raw_runs.append(
            {
                "agent_loop": FakeLoop(),
                "agent_loop_output": MultiTrajectoryAgentLoopOutput(
                    trajectories=[_output(snapshot=False, terminal=True)],
                    contextpilot_branch_points=[
                        {"sensitivity": score, "context_delta": 0.0, "entropy_delta": 0.0}
                        for score in scores
                    ],
                ),
                "trajectory": {"validate": False},
                "kwargs": {"uid": "q"},
                "request_id": f"q-{run_index}",
            }
        )

    outputs = await worker._run_contextpilot_query_group(raw_runs, {})

    assert len(selected_scores) == 19
    assert len(outputs) == 88


@pytest.mark.asyncio
async def test_generation_starts_new_12k_segment_before_truncating_a_full_turn():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.contextpilot_partial_rollout_enable = False
    loop.contextpilot_enable = True
    loop.contextpilot_budget_token_limit = 26000
    loop.prompt_length = 28672
    loop.response_length = 12288
    loop.max_response_length = 2048
    loop.max_model_length = 40960

    captured = {}

    class FakeServer:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            captured["prompt_ids"] = list(kwargs["prompt_ids"])
            captured["sampling_params"] = dict(kwargs["sampling_params"])
            return SimpleNamespace(
                token_ids=[7] * 96,
                log_probs=[-0.1] * 96,
                token_entropies=[0.2] * 96,
            )

    class FakeParser:
        async def extract_tool_calls(self, response_ids):
            return "", []

    loop.server_manager = FakeServer()
    loop.tool_parser = FakeParser()
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [1] * 12000
    data.response_mask = [1] * 11800
    data.response_logprobs = [-0.2] * 11800

    state = await loop._handle_generating_state(
        data,
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    )

    assert state is AgentState.TERMINATED
    assert captured["sampling_params"]["max_tokens"] == 2048
    assert captured["sampling_params"]["temperature"] == 0.7
    assert captured["sampling_params"]["top_p"] == 0.8
    assert captured["sampling_params"]["top_k"] == 20
    assert len(data.trajectory_snapshots) == 1
    assert len(data.trajectory_snapshots[0]["response_mask"]) == 11800
    assert data.trajectory_snapshots[0]["extra_fields"]["contextpilot_length_boundary"] is True
    assert data.trajectory_snapshots[0]["extra_fields"]["contextpilot_branchable"] is False
    assert len(data.response_mask) == 96
    assert data.context_len_or_turn_exceeded is False


@pytest.mark.asyncio
async def test_context_budget_is_a_hard_rollout_input_limit():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.contextpilot_enable = True
    loop.contextpilot_budget_token_limit = 8

    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [1] * 9

    state = await loop._handle_generating_state(data, {})

    assert state is AgentState.TERMINATED
    assert data.context_len_or_turn_exceeded is True
    assert data.assistant_turns == 0


@pytest.mark.asyncio
async def test_hard_context_limit_keeps_masked_segment_after_tool_observation():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.contextpilot_enable = True
    loop.contextpilot_budget_token_limit = 8
    loop.max_model_length = 40
    loop.prompt_length = 20
    loop.response_length = 20
    loop.min_assistant_turns = None

    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    data.response_ids = [5, 6]
    data.response_mask = [1, 1, 0, 0, 0]
    data.response_logprobs = [-0.2, -0.3, 0.0, 0.0, 0.0]
    data.assistant_turns = 1

    state = await loop._handle_generating_state(data, {})
    outputs = loop._collect_trajectory_outputs(data)

    assert state is AgentState.TERMINATED
    assert data.context_len_or_turn_exceeded is True
    assert data.response_ids == []
    assert len(outputs) == 1
    terminal = outputs[0]
    assert terminal.extra_fields["contextpilot_terminal"] is True
    assert terminal.extra_fields["context_len_or_turn_exceeded"] is True
    assert terminal.prompt_ids == [1, 2, 3, 4]
    assert terminal.response_ids == [5, 6, 7, 8, 9]
    assert terminal.response_mask == [1, 1, 0, 0, 0]


def test_snapshot_can_train_more_than_4k_without_truncation():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.max_model_length = 40960
    loop.prompt_length = 28672
    loop.response_length = 12288
    prompt_prefix = [1] * 100
    response = list(range(11000))

    output = loop._create_trajectory_output(
        prompt_ids=prompt_prefix + response,
        response_ids=response,
        response_mask=[1] * len(response),
        response_logprobs=[-0.1] * len(response),
        num_turns=1,
        enforce_output=True,
    )

    assert output is not None
    assert output.response_ids == response
    assert output.prompt_ids == prompt_prefix
    assert len(output.response_mask) == 11000


def test_dynamic_segment_preserves_37k_history_and_masks_only_new_tokens():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.prompt_length = 28672
    loop.response_length = 12288
    loop.max_model_length = 40960
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    original_history = list(range(37000))
    new_response = [50000] * 2048
    data.prompt_ids = original_history.copy()

    assert loop._start_training_segment(data) is True
    assert len(data.response_mask) == 37000 - 28672
    assert not any(data.response_mask)

    data.prompt_ids += new_response
    data.response_ids = new_response.copy()
    data.response_mask += [1] * len(new_response)
    data.response_logprobs += [-0.25] * len(new_response)
    output = loop._create_trajectory_output(
        prompt_ids=data.prompt_ids,
        response_ids=data.response_ids,
        response_mask=data.response_mask,
        response_logprobs=data.response_logprobs,
        num_turns=1,
    )

    assert output.prompt_ids == original_history[:28672]
    assert output.prompt_ids + output.response_ids == original_history + new_response
    assert output.response_mask[: 37000 - 28672] == [0] * (37000 - 28672)
    assert output.response_mask[37000 - 28672 :] == [1] * 2048


@pytest.mark.asyncio
async def test_25k_history_still_allows_one_full_2k_generation():
    loop = StatelmToolAgentLoop.__new__(StatelmToolAgentLoop)
    loop.contextpilot_partial_rollout_enable = False
    loop.contextpilot_enable = True
    loop.contextpilot_budget_token_limit = 26000
    loop.prompt_length = 28672
    loop.response_length = 12288
    loop.max_response_length = 2048
    loop.max_model_length = 40960

    captured = {}

    class FakeServer:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            captured["prompt_ids"] = list(kwargs["prompt_ids"])
            captured["sampling_params"] = dict(kwargs["sampling_params"])
            return SimpleNamespace(
                token_ids=[7] * 32,
                log_probs=[-0.1] * 32,
                token_entropies=[0.2] * 32,
            )

    class FakeParser:
        async def extract_tool_calls(self, response_ids):
            return "", []

    loop.server_manager = FakeServer()
    loop.tool_parser = FakeParser()
    data = AgentData(messages=[], image_data=None, metrics={}, request_id="r", tools_kwargs={})
    data.prompt_ids = [1] * 25000
    assert loop._start_training_segment(data) is True

    state = await loop._handle_generating_state(data, {"temperature": 0.7})

    assert state is AgentState.TERMINATED
    assert len(captured["prompt_ids"]) == 25000
    assert captured["sampling_params"]["max_tokens"] == 2048
    assert len(data.response_mask) == 32
    assert sum(data.response_mask) == 32


def test_manager_keeps_all_rollouts_for_a_uid_on_one_worker():
    manager = SimpleNamespace(agent_loop_workers=[object(), object(), object()])
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    manager._chunk_prompts_preserving_queries = MethodType(
        AgentLoopManager._chunk_prompts_preserving_queries,
        manager,
    )
    uids = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    data = DataProto(
        batch=TensorDict({"x": torch.arange(6).unsqueeze(1)}, batch_size=6),
        non_tensor_batch={"uid": uids},
    )

    chunks = manager._chunk_prompts_preserving_queries(data)
    locations = {}
    for chunk_index, chunk in enumerate(chunks):
        for uid in chunk.non_tensor_batch["uid"]:
            locations.setdefault(uid, set()).add(chunk_index)

    assert all(len(chunk_ids) == 1 for chunk_ids in locations.values())


def test_server_manager_migrates_sticky_session_away_from_live_straggler(monkeypatch):
    monkeypatch.setenv("VERL_AGENT_STICKY_MAX_LOAD_SKEW", "2")
    servers = [object(), object(), object(), object()]
    manager = AsyncLLMServerManager(SimpleNamespace(), servers)

    sticky_server = manager.server_handles[0]
    idle_server = manager.server_handles[1]
    manager.request_id_to_server["long-branch"] = sticky_server
    manager._inflight_requests.update({server: 2 for server in manager.server_handles})
    manager._inflight_requests[sticky_server] = 5
    manager._inflight_requests[idle_server] = 0

    selected = manager._acquire_server("long-branch")

    assert selected is idle_server
    assert manager.request_id_to_server["long-branch"] is idle_server
    assert manager._inflight_requests[idle_server] == 1
    manager._release_server(selected)
    assert manager._inflight_requests[idle_server] == 0


def test_server_manager_preserves_stickiness_when_load_is_balanced(monkeypatch):
    monkeypatch.setenv("VERL_AGENT_STICKY_MAX_LOAD_SKEW", "2")
    servers = [object(), object()]
    manager = AsyncLLMServerManager(SimpleNamespace(), servers)

    sticky_server = manager.server_handles[0]
    manager.request_id_to_server["cached-branch"] = sticky_server
    manager._inflight_requests[sticky_server] = 2
    manager._inflight_requests[manager.server_handles[1]] = 0

    selected = manager._acquire_server("cached-branch")

    assert selected is sticky_server
    manager._release_server(selected)
