#!/usr/bin/env python3
"""FSM overlay that exposes only the tools legal at the current agent state."""

from __future__ import annotations

import json
import os
import time
import uuid
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from infer.src.contextpilot import ContextPilot, ExecLogger, _strip_think


class FSMState(Enum):
    S0_INIT = auto()
    S0_5_ANALYZED = auto()
    S1_PLANNED = auto()
    S1_5_INDEX_BUILT = auto()
    S2_PENDING_SEARCH = auto()
    S2_5_PENDING_PLAN = auto()
    S3_SEARCHED = auto()
    S3_5_PENDING_POST_READ_PLAN = auto()
    S4_READ = auto()
    S4_5_PENDING_SEARCH_CLEANUP = auto()
    S4_6_PENDING_PLAN_CLEANUP = auto()
    S5_NOTED = auto()
    S5_5_PENDING_DELETE = auto()
    S6_REQUIRE_READ_NOTE = auto()
    S6_5_REQUIRE_LOAD_MEMORY = auto()
    S7_REVIEWING = auto()
    DONE = auto()


class ContextPilotFSM(ContextPilot):
    """ContextPilot agent with finite-state tool constraints."""

    def __init__(self, **kwargs):
        self._use_more_plan: bool = bool(kwargs.pop("use_more_plan", False))
        self._use_required_tool_choice: bool = bool(
            kwargs.pop("use_required_tool_choice", True)
        )
        super().__init__(**kwargs)
        self._has_plan: bool = any(
            t["function"]["name"] == "plan" for t in self.tools
        )
        self._has_analyze_text: bool = any(
            t["function"]["name"] == "analyzeText" for t in self.tools
        )
        self._fsm_state: FSMState = self._initial_fsm_state()
        self._last_retrieved_chunks: List = []
        self._read_chunk_count_this_cycle: int = 0
        self._read_chunk_ids_this_cycle: Set[int] = set()
        self._max_read_chunks: int = 0
        self._min_read_chunks_before_memory: int = 2
        self._max_read_calls_per_cycle: int = 2
        self._s5_delete_count: int = 0
        self._allowed_msg_ids_for_s5: Set[int] = set()
        self._last_note_assistant_msg_id: Optional[int] = None
        self._allowed_msg_ids_for_s5_5: Set[int] = set()
        self._readchunk_tool_msg_ids: List[int] = []
        self._readchunk_msg_id_to_chunk_ids: Dict[int, Set[int]] = {}
        self._search_tool_msg_ids: Set[int] = set()
        self._search_call_count: int = 0
        self._search_delete_count: int = 0
        self._allowed_msg_ids_for_s4_5: Set[int] = set()
        self._plan_assistant_msg_ids: Set[int] = set()
        self._plan_call_count: int = 0
        self._plan_delete_count: int = 0
        self._allowed_msg_ids_for_s4_6: Set[int] = set()
        self._has_build_embedding: bool = any(
            t["function"]["name"] == "buildEmbedding" for t in self.tools
        )
        self._no_tool_retries: int = 0
        self._msg_id_error_retries: int = 0
        self._max_msg_id_error_retries: int = 1
        self._tool_error_retries: int = 0
        self._max_tool_error_retries: int = 3
        self._pending_search_limit_force_finish: bool = False
        self._consecutive_empty_searches: int = 0
        self._max_consecutive_empty_searches: int = max(
            1, int(os.getenv("CONTEXTPILOT_MAX_CONSECUTIVE_EMPTY_SEARCHES", "6"))
        )
        self._note_write_seen: bool = False
        self._memory_write_seen: bool = False
        self._note_review_satisfied: bool = False
        self._memory_review_satisfied: bool = False

    def _has_external_note_keys(self) -> bool:
        return bool(getattr(self.state_manager, "simple_notes", {}))

    def _has_external_memory_keys(self) -> bool:
        return bool(getattr(self.state_manager, "notes", {}))

    def _needs_note_review(self) -> bool:
        return (self._note_write_seen
                and not self._note_review_satisfied
                and self._has_external_note_keys())

    def _needs_memory_review(self) -> bool:
        return (self._memory_write_seen
                and not self._memory_review_satisfied
                and self._has_external_memory_keys())

    def _next_review_state(self) -> FSMState:
        if self._needs_note_review():
            return FSMState.S6_REQUIRE_READ_NOTE
        if self._needs_memory_review():
            return FSMState.S6_5_REQUIRE_LOAD_MEMORY
        return FSMState.S7_REVIEWING

    def _get_allowed_tools_for_state(self) -> List[Dict]:
        """Return the subset of self.tools allowed in the current FSM state."""
        state = self._fsm_state
        if state == FSMState.S0_INIT:
            names = {"analyzeText"}
        elif state == FSMState.S0_5_ANALYZED:
            names = {"plan"}
        elif state == FSMState.S1_PLANNED:
            names = {"buildIndex"}
        elif state == FSMState.S1_5_INDEX_BUILT:
            names = {"buildEmbedding"}
        elif state == FSMState.S2_PENDING_SEARCH:
            names = {"searchEngine", "semanticSearch", "hybridSearch"}
        elif state == FSMState.S2_5_PENDING_PLAN:
            names = {"plan"}
        elif state == FSMState.S3_SEARCHED:
            names = {"readChunk", "readMultiChunks"}
        elif state == FSMState.S3_5_PENDING_POST_READ_PLAN:
            names = {"plan"}
        elif state == FSMState.S4_READ:
            names = set()
            min_read_chunks = min(
                self._min_read_chunks_before_memory,
                max(1, self._max_read_chunks),
            )
            evidence_ready = len(self._read_chunk_ids_this_cycle) >= min_read_chunks
            can_read_more = self._read_chunk_count_this_cycle < self._max_read_calls_per_cycle
            if evidence_ready:
                names.update({"memorize", "updateMemory", "note", "updateNote"})
            if can_read_more and not evidence_ready:
                names.update({"readChunk", "readMultiChunks"})
            if not names and self._readchunk_tool_msg_ids:
                names.update({"deleteContext", "truncateContext", "summarizeContext", "compressContext"})
        elif state == FSMState.S4_5_PENDING_SEARCH_CLEANUP:
            names = {"deleteContext", "truncateContext", "summarizeContext", "compressContext"}
        elif state == FSMState.S4_6_PENDING_PLAN_CLEANUP:
            names = {"deleteContext", "truncateContext", "summarizeContext", "compressContext"}
        elif state == FSMState.S5_NOTED:
            names = {"deleteContext", "truncateContext", "summarizeContext", "compressContext"}
        elif state == FSMState.S5_5_PENDING_DELETE:
            names = {"deleteContext"}
        elif state == FSMState.S6_REQUIRE_READ_NOTE:
            names = {"readNote", "restoreContext"}
        elif state == FSMState.S6_5_REQUIRE_LOAD_MEMORY:
            names = {"loadMemory", "restoreContext"}
        elif state == FSMState.S7_REVIEWING:
            names = {"searchEngine", "semanticSearch", "hybridSearch", "readChunk", "readMultiChunks", "loadMemory", "readNote", "restoreContext", "finish"}
        else:
            return self.tools

        if "readNote" in names and not self._has_external_note_keys():
            names.discard("readNote")
        if "loadMemory" in names and not self._has_external_memory_keys():
            names.discard("loadMemory")

        filtered = [t for t in self.tools if t["function"]["name"] in names]
        if not filtered:
            print(f"    [FSM-WARN] No tools matched for state {state.name}, "
                  f"falling back to full tool set.")
            return self.tools
        return filtered

    def _validate_or_autocorrect_store_key(self, action: str, params: dict) -> Optional[dict]:
        """Ensure readNote/loadMemory keys come from the advertised store.

        When there is exactly one available key, auto-correcting is safer than
        burning retries on a hallucinated key during a mandatory review gate.
        """
        if action == "readNote":
            store = getattr(self.state_manager, "simple_notes", {})
            label = "note"
        elif action == "loadMemory":
            store = getattr(self.state_manager, "notes", {})
            label = "memory"
        else:
            return None

        insertion_order = [str(k) for k in store.keys()]
        available = sorted(insertion_order)
        requested = str((params or {}).get("key", ""))
        if requested in available:
            return None
        if (available and self._fsm_state in (
                FSMState.S6_REQUIRE_READ_NOTE,
                FSMState.S6_5_REQUIRE_LOAD_MEMORY,
        )):
            corrected = insertion_order[-1]
            print(f"    [FSM] Auto-correcting mandatory {action} key "
                  f"'{requested}' -> most recent key '{corrected}'.")
            params["key"] = corrected
            return None
        if len(available) == 1:
            corrected = available[0]
            print(f"    [FSM] Auto-correcting {action} key "
                  f"'{requested}' -> '{corrected}'.")
            params["key"] = corrected
            return None
        if available:
            return {
                "error": (
                    f"{label.capitalize()} key '{requested}' is not available. "
                    f"Use exactly one of these keys: {available}."
                )
            }
        return {
            "error": (
                f"No {label}s are available. Do not call {action}; "
                f"continue with search/read tools or finish if sufficient."
            )
        }

    def _initial_fsm_state(self) -> "FSMState":
        """Compute the initial FSM state based on which tools are available."""
        if self._has_plan and self._has_analyze_text:
            return FSMState.S0_INIT
        if self._has_plan:
            return FSMState.S0_5_ANALYZED
        return FSMState.S1_PLANNED

    def _validate_s4_5_msg_id(self, params: dict) -> Optional[str]:
        """
        In S4.5, the model may only target msg_ids that are
        the tool-result msg_ids of searchEngine/semanticSearch/hybridSearch calls
        that have not yet been deleted.
        Returns an error string if invalid, None if OK.
        """
        msg_id = params.get("msg_id")
        if msg_id is None:
            return None
        try:
            msg_id = int(msg_id)
        except (ValueError, TypeError):
            return None
        if msg_id not in self._allowed_msg_ids_for_s4_5:
            return (f"msg_id {msg_id} is not allowed in state S4.5. "
                    f"You may only target these search-result msg_ids: "
                    f"{sorted(self._allowed_msg_ids_for_s4_5)}.")
        return None

    def _validate_s4_6_msg_id(self, params: dict) -> Optional[str]:
        """
        In S4.6, the model may only target msg_ids that are
        the assistant msg_ids of plan calls that have not yet been deleted.
        Returns an error string if invalid, None if OK.
        """
        msg_id = params.get("msg_id")
        if msg_id is None:
            return None
        try:
            msg_id = int(msg_id)
        except (ValueError, TypeError):
            return None
        if msg_id not in self._allowed_msg_ids_for_s4_6:
            return (f"msg_id {msg_id} is not allowed in state S4.6. "
                    f"You may only target these plan-result msg_ids: "
                    f"{sorted(self._allowed_msg_ids_for_s4_6)}.")
        return None

    def _validate_s4_read_cleanup_msg_id(self, params: dict) -> Optional[str]:
        msg_id = params.get("msg_id")
        if msg_id is None:
            return None
        try:
            msg_id = int(msg_id)
        except (ValueError, TypeError):
            return None
        allowed = set(self._readchunk_tool_msg_ids)
        if msg_id not in allowed:
            return (f"msg_id {msg_id} is not allowed in state S4. "
                    f"You may only target these read-result msg_ids: "
                    f"{sorted(allowed)}.")
        return None

    def _record_read_chunk_ids(self, tool_name: str, params: Optional[dict]) -> Set[int]:
        """Track distinct document chunks read in the current search cycle."""
        params = params or {}
        read_ids: Set[int] = set()
        if tool_name == "readChunk":
            chunk_id = params.get("chunk_id")
            try:
                read_ids.add(int(chunk_id))
            except (TypeError, ValueError):
                pass
        elif tool_name == "readMultiChunks":
            chunk_ids = params.get("chunk_ids", [])
            if isinstance(chunk_ids, (str, int)):
                chunk_ids = [chunk_ids]
            for chunk_id in chunk_ids or []:
                try:
                    read_ids.add(int(chunk_id))
                except (TypeError, ValueError):
                    pass
        self._read_chunk_ids_this_cycle.update(read_ids)
        return read_ids

    def _autocorrect_duplicate_read(self, tool_name: str, params: dict) -> None:
        """Replace a duplicate follow-up read with an unread search candidate."""
        if self._fsm_state != FSMState.S4_READ:
            return

        requested: Set[int] = set()
        values = (params.get("chunk_ids", []) if tool_name == "readMultiChunks"
                  else [params.get("chunk_id")])
        if isinstance(values, (str, int)):
            values = [values]
        for value in values or []:
            try:
                requested.add(int(value))
            except (TypeError, ValueError):
                pass
        if requested - self._read_chunk_ids_this_cycle:
            return

        candidates = []
        for chunk in self._last_retrieved_chunks or []:
            value = chunk.get("chunk_id") if isinstance(chunk, dict) else chunk
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value not in self._read_chunk_ids_this_cycle:
                candidates.append(value)
        if not candidates:
            return

        replacement = candidates[0]
        if tool_name == "readMultiChunks":
            params["chunk_ids"] = [replacement]
        else:
            params["chunk_id"] = replacement
        print(f"    [FSM] Auto-correcting duplicate follow-up read to unread "
              f"chunk_id={replacement}.")

    def _autocorrect_invalid_read(self, tool_name: str, params: dict) -> None:
        """Normalize malformed or out-of-range chunk ids before execution.

        Prefer a ranked unread candidate from the latest search. Clamping to
        the document boundary is only a fallback because an arbitrary edge
        chunk is less likely to contain the requested evidence.
        """
        max_valid = len(getattr(self.tool_library, "index", []) or []) - 1
        if max_valid < 0:
            return

        candidates = []
        for chunk in self._last_retrieved_chunks or []:
            value = chunk.get("chunk_id") if isinstance(chunk, dict) else chunk
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= value <= max_valid and value not in self._read_chunk_ids_this_cycle:
                candidates.append(value)

        raw_values = (params.get("chunk_ids", []) if tool_name == "readMultiChunks"
                      else [params.get("chunk_id")])
        if isinstance(raw_values, (str, int)):
            raw_values = [raw_values]

        normalized = []
        invalid_values = []
        for raw in raw_values or []:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                invalid_values.append(raw)
                continue
            if 0 <= value <= max_valid:
                normalized.append(value)
            else:
                invalid_values.append(value)

        if not invalid_values:
            return

        replacement = next((c for c in candidates if c not in normalized), None)
        if replacement is None:
            numeric_invalid = [v for v in invalid_values if isinstance(v, int)]
            requested = numeric_invalid[0] if numeric_invalid else 0
            replacement = min(max(requested, 0), max_valid)

        if tool_name == "readMultiChunks":
            corrected = []
            for value in normalized + [replacement]:
                if value not in corrected:
                    corrected.append(value)
            params["chunk_ids"] = corrected
        else:
            params["chunk_id"] = replacement
        print(f"    [FSM] Auto-correcting invalid {tool_name} chunk id(s) "
              f"{invalid_values} -> chunk_id={replacement}.")

    def _drop_read_chunk_msg_id(self, tool_msg_id: int) -> None:
        """Forget read chunks whose tool result has been cleaned from context."""
        self._readchunk_msg_id_to_chunk_ids.pop(tool_msg_id, None)
        merged: Set[int] = set()
        for ids in self._readchunk_msg_id_to_chunk_ids.values():
            merged.update(ids)
        self._read_chunk_ids_this_cycle = merged

    def _auto_delete_context_msg(self, msg_id_to_delete: int, reason: str) -> bool:
        """Execute a synthetic deleteContext transition for FSM recovery."""
        print(f"    [FSM] Auto-cleaning: executing deleteContext(msg_id={msg_id_to_delete}) "
              f"because {reason}.")
        auto_params = {"msg_id": msg_id_to_delete}
        try:
            auto_result = self._execute_tool("deleteContext", auto_params)
        except Exception as exc:
            auto_result = {"error": f"deleteContext raised {type(exc).__name__}: {exc}"}
            print(f"    [FSM] Auto-clean tool error: {auto_result['error']}")

        if "error" in auto_result:
            return False

        self.ctx_counter += 1
        auto_assistant_msg_id = self.ctx_counter
        auto_tool_use_id = f"auto_clean_{auto_assistant_msg_id}"
        auto_assistant_entry = {
            "role": "assistant",
            "content": [],
            "tool_calls": [{
                "id": auto_tool_use_id,
                "type": "function",
                "function": {
                    "name": "deleteContext",
                    "arguments": json.dumps(auto_params, ensure_ascii=False),
                },
            }],
            "msg_id": auto_assistant_msg_id,
        }
        self.ctx_counter += 1
        auto_tool_msg_id = self.ctx_counter
        auto_tool_entry = {
            "role": "tool",
            "content": deepcopy(auto_result),
            "msg_id": auto_tool_msg_id,
            "msg_id(invoking_assistant)": auto_assistant_msg_id,
            "tool_use_id": auto_tool_use_id,
            "tool_name": "deleteContext",
        }
        self.full_history.append(auto_assistant_entry)
        self.full_history.append(auto_tool_entry)
        self._no_tool_retries = 0
        self._msg_id_error_retries = 0
        self._tool_error_retries = 0
        self._pending_error_messages.clear()
        self._fsm_transition(
            "deleteContext", auto_result,
            auto_tool_msg_id, auto_assistant_msg_id,
            auto_params)
        auto_preview = json.dumps(auto_result, ensure_ascii=False)
        if len(auto_preview) > 200:
            auto_preview = auto_preview[:200] + "..."
        print(f"[RUN] Auto-clean result (ID: {auto_tool_msg_id}): {auto_preview}")
        return True

    def _recover_context_overflow(self) -> bool:
        available_reads = []
        available_searches = []
        for message in self.full_history:
            if message.get("role") != "tool":
                continue
            msg_id = message.get("msg_id")
            if msg_id is None or int(msg_id) in self.deleted_msg_ids:
                continue
            tool_name = message.get("tool_name")
            if tool_name in {"readChunk", "readMultiChunks", "nextChunk"}:
                available_reads.append(int(msg_id))
            elif tool_name == "searchEngine":
                available_searches.append(int(msg_id))

        if len(available_reads) > 1:
            target = available_reads[0]
        elif available_searches:
            target = available_searches[0]
        elif available_reads:
            target = available_reads[0]
        else:
            return False

        return self._auto_delete_context_msg(target, "the model context window overflowed")

    def _auto_cleanup_mechanical_context(self) -> bool:
        """Auto-delete deterministic cleanup targets.

        These cleanup states do not require model judgement: the FSM has
        already computed the exact search-result or plan-assistant msg_ids
        that are safe to prune. Running the deletes directly avoids spending
        LLM turns on repetitive context hygiene.
        """
        if self._fsm_state == FSMState.S4_5_PENDING_SEARCH_CLEANUP:
            pending_search = self._search_call_count - self._search_delete_count
            if pending_search >= 2 and self._allowed_msg_ids_for_s4_5:
                return self._auto_delete_context_msg(
                    min(self._allowed_msg_ids_for_s4_5),
                    "S4.5 has excess search-result context")
        elif self._fsm_state == FSMState.S4_6_PENDING_PLAN_CLEANUP:
            pending_plan = self._plan_call_count - self._plan_delete_count
            if pending_plan >= 3 and self._allowed_msg_ids_for_s4_6:
                return self._auto_delete_context_msg(
                    min(self._allowed_msg_ids_for_s4_6),
                    "S4.6 has excess plan context")
        elif self._fsm_state == FSMState.S5_NOTED:
            if self._allowed_msg_ids_for_s5:
                return self._auto_delete_context_msg(
                    min(self._allowed_msg_ids_for_s5),
                    "S5 has a deterministic read-result cleanup target")
        elif self._fsm_state == FSMState.S5_5_PENDING_DELETE:
            if self._allowed_msg_ids_for_s5_5:
                return self._auto_delete_context_msg(
                    min(self._allowed_msg_ids_for_s5_5),
                    "S5.5 has a deterministic note/memory call cleanup target")
        return False

    def _validate_s5_msg_id(self, params: dict) -> Optional[str]:
        """
        In S5, the model may only target msg_ids that are
        the tool-result msg_ids of readChunk/readMultiChunks calls (role: tool).
        Returns an error string if invalid, None if OK.
        """
        msg_id = params.get("msg_id")
        if msg_id is None:
            return None
        try:
            msg_id = int(msg_id)
        except (ValueError, TypeError):
            return None
        if msg_id not in self._allowed_msg_ids_for_s5:
            return (f"msg_id {msg_id} is not allowed in state S5. "
                    f"You may only target these read-result msg_ids: "
                    f"{sorted(self._allowed_msg_ids_for_s5)}.")
        return None

    def _validate_s5_5_msg_id(self, params: dict) -> Optional[str]:
        """
        In S5.5, the model may only target the invoking-assistant msg_id
        of the memorize/updateMemory/note/updateNote call (role: assistant).
        Returns an error string if invalid, None if OK.
        """
        msg_id = params.get("msg_id")
        if msg_id is None:
            return None
        try:
            msg_id = int(msg_id)
        except (ValueError, TypeError):
            return None
        if msg_id not in self._allowed_msg_ids_for_s5_5:
            return (f"msg_id {msg_id} is not allowed in state S5.5. "
                    f"You may only target this note/memory assistant msg_id: "
                    f"{sorted(self._allowed_msg_ids_for_s5_5)}.")
        return None

    def _fsm_transition(self, tool_name: str, result: dict,
                        tool_msg_id: int, assistant_msg_id: int,
                        params: dict = None) -> None:
        """Advance the FSM based on the tool that was just called and its result."""
        old = self._fsm_state

        if tool_name == "plan" and "error" not in result:
            if assistant_msg_id is not None:
                self._plan_assistant_msg_ids.add(assistant_msg_id)
            self._plan_call_count += 1

        if old == FSMState.S0_INIT:
            if tool_name == "analyzeText" and "error" not in result:
                self._fsm_state = FSMState.S0_5_ANALYZED

        elif old == FSMState.S0_5_ANALYZED:
            if tool_name == "plan" and "error" not in result:
                self._fsm_state = FSMState.S1_PLANNED

        elif old == FSMState.S1_PLANNED:
            if tool_name == "buildIndex" and "error" not in result:
                if self._has_build_embedding:
                    self._fsm_state = FSMState.S1_5_INDEX_BUILT
                else:
                    self._fsm_state = FSMState.S2_PENDING_SEARCH

        elif old == FSMState.S1_5_INDEX_BUILT:
            if tool_name == "buildEmbedding" and "error" not in result:
                self._fsm_state = FSMState.S2_PENDING_SEARCH

        elif old == FSMState.S2_PENDING_SEARCH:
            if tool_name in ("searchEngine", "semanticSearch", "hybridSearch") and "error" not in result:
                chunks = result.get("retrieved_chunks", [])
                self._last_retrieved_chunks = chunks
                if chunks:
                    self._consecutive_empty_searches = 0
                    self._search_call_count += 1
                    self._max_read_chunks = len(chunks)
                    self._read_chunk_count_this_cycle = 0
                    self._read_chunk_ids_this_cycle = set()
                    self._readchunk_tool_msg_ids = []
                    self._readchunk_msg_id_to_chunk_ids = {}
                    self._search_tool_msg_ids.add(tool_msg_id)
                    if self._has_plan:
                        self._fsm_state = FSMState.S2_5_PENDING_PLAN
                    else:
                        self._fsm_state = FSMState.S3_SEARCHED
                else:
                    self._consecutive_empty_searches += 1
                    if self._consecutive_empty_searches >= self._max_consecutive_empty_searches:
                        print(f"    [FSM] {self._consecutive_empty_searches} consecutive empty searches; "
                              "switching to best-effort review/finish.")
                        self._fsm_state = FSMState.S7_REVIEWING

        elif old == FSMState.S2_5_PENDING_PLAN:
            if tool_name == "plan" and "error" not in result:
                self._fsm_state = FSMState.S3_SEARCHED

        elif old == FSMState.S3_SEARCHED:
            if tool_name in ("readChunk", "readMultiChunks") and "error" not in result:
                self._read_chunk_count_this_cycle = 1
                read_ids = self._record_read_chunk_ids(tool_name, params)
                self._readchunk_tool_msg_ids = [tool_msg_id]
                self._readchunk_msg_id_to_chunk_ids[tool_msg_id] = read_ids
                if self._has_plan and self._use_more_plan:
                    self._fsm_state = FSMState.S3_5_PENDING_POST_READ_PLAN
                else:
                    self._fsm_state = FSMState.S4_READ

        elif old == FSMState.S3_5_PENDING_POST_READ_PLAN:
            if tool_name == "plan" and "error" not in result:
                self._fsm_state = FSMState.S4_READ

        elif old == FSMState.S4_READ:
            if tool_name in ("readChunk", "readMultiChunks") and "error" not in result:
                self._read_chunk_count_this_cycle += 1
                read_ids = self._record_read_chunk_ids(tool_name, params)
                self._readchunk_tool_msg_ids.append(tool_msg_id)
                self._readchunk_msg_id_to_chunk_ids[tool_msg_id] = read_ids
                self._fsm_state = FSMState.S4_READ
            elif tool_name in ("deleteContext", "truncateContext", "summarizeContext", "compressContext") \
                    and "error" not in result:
                targeted_msg_id = params.get("msg_id") if params else None
                try:
                    targeted_msg_id = int(targeted_msg_id) if targeted_msg_id is not None else None
                except (ValueError, TypeError):
                    targeted_msg_id = None
                if targeted_msg_id is not None:
                    try:
                        self._readchunk_tool_msg_ids.remove(targeted_msg_id)
                    except ValueError:
                        pass
                    self._drop_read_chunk_msg_id(targeted_msg_id)
                if self._readchunk_tool_msg_ids:
                    self._fsm_state = FSMState.S4_READ
                else:
                    self._fsm_state = FSMState.S2_PENDING_SEARCH
            elif tool_name in ("memorize", "updateMemory", "note", "updateNote") and "error" not in result:
                self._last_note_assistant_msg_id = assistant_msg_id
                if tool_name in ("note", "updateNote") and self._has_external_note_keys():
                    self._note_write_seen = True
                    self._note_review_satisfied = False
                if tool_name in ("memorize", "updateMemory") and self._has_external_memory_keys():
                    self._memory_write_seen = True
                    self._memory_review_satisfied = False
                pending_search = self._search_call_count - self._search_delete_count
                pending_plan = self._plan_call_count - self._plan_delete_count
                self._allowed_msg_ids_for_s5 = set(self._readchunk_tool_msg_ids)
                self._allowed_msg_ids_for_s5_5 = set()
                if assistant_msg_id is not None:
                    self._allowed_msg_ids_for_s5_5.add(assistant_msg_id)
                self._s5_delete_count = 0
                if pending_search >= 2:
                    self._allowed_msg_ids_for_s4_5 = set(self._search_tool_msg_ids)
                    self._fsm_state = FSMState.S4_5_PENDING_SEARCH_CLEANUP
                    print(f"    [FSM] pending_search={pending_search} >= 2, "
                          f"entering S4.5 to clean up search results. "
                          f"Allowed search msg_ids: {sorted(self._allowed_msg_ids_for_s4_5)}")
                elif pending_plan >= 3:
                    self._allowed_msg_ids_for_s4_6 = set(self._plan_assistant_msg_ids)
                    self._fsm_state = FSMState.S4_6_PENDING_PLAN_CLEANUP
                    print(f"    [FSM] pending_search={pending_search} < 2, "
                          f"pending_plan={pending_plan} >= 3, entering S4.6 "
                          f"to clean up plan-assistant messages. "
                          f"Allowed plan-assistant msg_ids: {sorted(self._allowed_msg_ids_for_s4_6)}")
                else:
                    self._fsm_state = FSMState.S5_NOTED

        elif old == FSMState.S4_5_PENDING_SEARCH_CLEANUP:
            if tool_name in ("deleteContext", "truncateContext", "summarizeContext", "compressContext") \
                    and "error" not in result:
                targeted_msg_id = params.get("msg_id") if params else None
                try:
                    targeted_msg_id = int(targeted_msg_id) if targeted_msg_id is not None else None
                except (ValueError, TypeError):
                    targeted_msg_id = None
                if targeted_msg_id is not None and targeted_msg_id in self._search_tool_msg_ids:
                    self._search_tool_msg_ids.discard(targeted_msg_id)
                    self._allowed_msg_ids_for_s4_5.discard(targeted_msg_id)
                    self._allowed_msg_ids_for_s5.discard(targeted_msg_id)
                    self._search_delete_count += 1
                    print(f"    [FSM] S4.5: Deleted search result msg_id={targeted_msg_id}, "
                          f"search_delete_count={self._search_delete_count}")
                pending_search = self._search_call_count - self._search_delete_count
                pending_plan = self._plan_call_count - self._plan_delete_count
                if pending_search >= 2:
                    print(f"    [FSM] S4.5: pending_search={pending_search} still >= 2, staying.")
                elif pending_plan >= 3:
                    self._allowed_msg_ids_for_s4_6 = set(self._plan_assistant_msg_ids)
                    self._fsm_state = FSMState.S4_6_PENDING_PLAN_CLEANUP
                    print(f"    [FSM] S4.5: pending_search={pending_search} < 2, "
                          f"pending_plan={pending_plan} >= 3, advancing to S4.6. "
                          f"Allowed plan-assistant msg_ids: {sorted(self._allowed_msg_ids_for_s4_6)}")
                else:
                    self._fsm_state = FSMState.S5_NOTED
                    print(f"    [FSM] S4.5: pending_search={pending_search} < 2, "
                          f"pending_plan={pending_plan} < 3, advancing to S5.")

        elif old == FSMState.S4_6_PENDING_PLAN_CLEANUP:
            if tool_name in ("deleteContext", "truncateContext", "summarizeContext", "compressContext") \
                    and "error" not in result:
                targeted_msg_id = params.get("msg_id") if params else None
                try:
                    targeted_msg_id = int(targeted_msg_id) if targeted_msg_id is not None else None
                except (ValueError, TypeError):
                    targeted_msg_id = None
                if targeted_msg_id is not None and targeted_msg_id in self._plan_assistant_msg_ids:
                    self._plan_assistant_msg_ids.discard(targeted_msg_id)
                    self._allowed_msg_ids_for_s4_6.discard(targeted_msg_id)
                    self._plan_delete_count += 1
                    print(f"    [FSM] S4.6: Deleted plan-assistant msg_id={targeted_msg_id}, "
                          f"plan_delete_count={self._plan_delete_count}")
                pending_plan = self._plan_call_count - self._plan_delete_count
                if pending_plan >= 3:
                    print(f"    [FSM] S4.6: pending_plan={pending_plan} still >= 3, staying.")
                else:
                    self._fsm_state = FSMState.S5_NOTED
                    print(f"    [FSM] S4.6: pending_plan={pending_plan} < 3, advancing to S5.")

        elif old == FSMState.S5_NOTED:
            if tool_name in ("deleteContext", "truncateContext", "summarizeContext", "compressContext") \
                    and "error" not in result:
                targeted_msg_id = params.get("msg_id") if params else None
                try:
                    targeted_msg_id = int(targeted_msg_id) if targeted_msg_id is not None else None
                except (ValueError, TypeError):
                    targeted_msg_id = None
                if targeted_msg_id is not None and targeted_msg_id in self._allowed_msg_ids_for_s5:
                    self._allowed_msg_ids_for_s5.discard(targeted_msg_id)
                    try:
                        self._readchunk_tool_msg_ids.remove(targeted_msg_id)
                    except ValueError:
                        pass
                    self._drop_read_chunk_msg_id(targeted_msg_id)
                    if self._allowed_msg_ids_for_s5:
                        self._fsm_state = FSMState.S5_NOTED
                        print(f"    [FSM] Processed read result msg_id={targeted_msg_id}, "
                              f"remaining read msg_ids: {sorted(self._allowed_msg_ids_for_s5)}")
                    else:
                        self._fsm_state = FSMState.S5_5_PENDING_DELETE
                        print(f"    [FSM] Processed read result msg_id={targeted_msg_id}, advancing to S5.5.")

        elif old == FSMState.S5_5_PENDING_DELETE:
            if tool_name == "deleteContext" and "error" not in result:
                targeted_msg_id = params.get("msg_id") if params else None
                try:
                    targeted_msg_id = int(targeted_msg_id) if targeted_msg_id is not None else None
                except (ValueError, TypeError):
                    targeted_msg_id = None
                if targeted_msg_id is not None:
                    self._allowed_msg_ids_for_s5_5.discard(targeted_msg_id)
                self._fsm_state = self._next_review_state()

        elif old == FSMState.S6_REQUIRE_READ_NOTE:
            if tool_name == "readNote" and "error" not in result:
                self._note_review_satisfied = True
                self._fsm_state = self._next_review_state()
            elif tool_name == "restoreContext" and "error" not in result:
                pass

        elif old == FSMState.S6_5_REQUIRE_LOAD_MEMORY:
            if tool_name == "loadMemory" and "error" not in result:
                self._memory_review_satisfied = True
                self._fsm_state = self._next_review_state()
            elif tool_name == "restoreContext" and "error" not in result:
                pass

        elif old == FSMState.S7_REVIEWING:
            if tool_name in ("searchEngine", "semanticSearch", "hybridSearch") and "error" not in result:
                chunks = result.get("retrieved_chunks", [])
                self._last_retrieved_chunks = chunks
                if chunks:
                    self._search_call_count += 1
                    self._max_read_chunks = len(chunks)
                    self._read_chunk_count_this_cycle = 0
                    self._readchunk_tool_msg_ids = []
                    self._search_tool_msg_ids.add(tool_msg_id)
                    if self._has_plan:
                        self._fsm_state = FSMState.S2_5_PENDING_PLAN
                    else:
                        self._fsm_state = FSMState.S3_SEARCHED
                else:
                    self._fsm_state = FSMState.S7_REVIEWING
            elif tool_name in ("readChunk", "readMultiChunks") and "error" not in result:
                self._read_chunk_count_this_cycle = 1
                self._readchunk_tool_msg_ids = [tool_msg_id]
                if self._has_plan and self._use_more_plan:
                    self._fsm_state = FSMState.S3_5_PENDING_POST_READ_PLAN
                else:
                    self._fsm_state = FSMState.S4_READ
            elif tool_name in ("loadMemory", "readNote", "restoreContext"):
                pass
            elif tool_name == "finish":
                self._fsm_state = FSMState.DONE

        if self._fsm_state != old:
            print(f"    [FSM] {old.name} --({tool_name})--> {self._fsm_state.name}")
        else:
            print(f"    [FSM] {old.name} --({tool_name})--> (same state)")

    def run(self, user_query, max_turns_to_fail=80):
        """
        FSM-aware run loop.  At each turn only the tools allowed by the
        current FSM state are presented to the model.  tool_choice is set
        to "required" in states where the model MUST call a tool.
        """
        self._fsm_state = self._initial_fsm_state()
        self._last_retrieved_chunks = []
        self._read_chunk_count_this_cycle = 0
        self._read_chunk_ids_this_cycle = set()
        self._max_read_chunks = 0
        self._s5_delete_count = 0
        self._allowed_msg_ids_for_s5 = set()
        self._last_note_assistant_msg_id = None
        self._allowed_msg_ids_for_s5_5 = set()
        self._readchunk_tool_msg_ids = []
        self._readchunk_msg_id_to_chunk_ids = {}
        self._search_tool_msg_ids = set()
        self._search_call_count = 0
        self._search_delete_count = 0
        self._allowed_msg_ids_for_s4_5 = set()
        self._plan_assistant_msg_ids = set()
        self._plan_call_count = 0
        self._plan_delete_count = 0
        self._allowed_msg_ids_for_s4_6 = set()
        self._no_tool_retries = 0
        self._msg_id_error_retries = 0
        self._tool_error_retries = 0
        self._pending_search_limit_force_finish = False
        self._consecutive_empty_searches = 0
        self._note_write_seen = False
        self._memory_write_seen = False
        self._note_review_satisfied = False
        self._memory_review_satisfied = False
        self._pending_error_messages: List[Dict] = []

        self.full_history.append({"role": "user", "content": user_query})
        self.ctx_counter = 0
        self.search_call_counter = 0
        turn = 0
        force_finish = False

        try:
            while (
                turn <= max_turns_to_fail
                or (self._fsm_state == FSMState.S7_REVIEWING and not force_finish)
            ):
                if self._fsm_state == FSMState.DONE:
                    print(f"\n--- FSM reached DONE state, stopping. ---")
                    break

                print(f"\n--- Round {turn} | FSM state: {self._fsm_state.name} "
                      f"(Max {max_turns_to_fail} rounds) ---")

                if self._auto_cleanup_mechanical_context():
                    continue

                if (
                    not force_finish
                    and self._fsm_state == FSMState.S7_REVIEWING
                    and (turn >= self.max_turns or self._pending_search_limit_force_finish)
                ):
                    reason = []
                    if turn >= self.max_turns:
                        reason.append(f"turn={turn} >= max_turns={self.max_turns}")
                    if self._pending_search_limit_force_finish:
                        reason.append(f"search_call_counter={self.search_call_counter} "
                                      f">= max_search_calls={self.max_search_calls}")
                    print(f"[INFO] Switching to force-finish mode "
                          f"(FSM state {self._fsm_state.name}; reason: {'; '.join(reason)}).")
                    force_finish = True
                    self._pending_search_limit_force_finish = False
                    self._inject_force_finish_messages()

                api_payload = self._build_api_payload()
                if self._pending_error_messages:
                    for pem in self._pending_error_messages:
                        if pem["role"] == "assistant":
                            raw_text = " ".join(
                                blk.get("text", "")
                                for blk in pem.get("content", [])
                                if blk.get("type") == "text"
                            )
                            cleaned_text = _strip_think(raw_text)
                            entry = {
                                "role": "assistant",
                                "content": cleaned_text if cleaned_text else None,
                            }
                            tc = pem.get("tool_calls")
                            if tc:
                                normalized = []
                                for t in tc:
                                    if hasattr(t, "model_dump"):
                                        normalized.append(t.model_dump())
                                    elif isinstance(t, dict):
                                        normalized.append(t)
                                    else:
                                        normalized.append(json.loads(json.dumps(t, default=str)))
                                entry["tool_calls"] = normalized
                            api_payload.append(entry)
                        elif pem["role"] == "tool":
                            content_cp = deepcopy(pem["content"])
                            content_cp["msg_id"] = pem["msg_id"]
                            content_cp["msg_id(invoking_assistant)"] = pem["msg_id(invoking_assistant)"]
                            api_payload.append({
                                "role": "tool",
                                "content": json.dumps(content_cp, ensure_ascii=False),
                                "tool_call_id": pem["tool_use_id"],
                            })
                        elif pem["role"] == "user":
                            api_payload.append({
                                "role": "user",
                                "content": pem["content"],
                            })
                    self._pending_error_messages.clear()
                if force_finish:
                    allowed_tools = [t for t in self.tools
                                     if t["function"]["name"] == "finish"]
                    if not allowed_tools:
                        allowed_tools = self._get_allowed_tools_for_state()
                else:
                    allowed_tools = self._get_allowed_tools_for_state()

                if self.max_token_window is not None:
                    api_payload = self._enforce_token_window(api_payload, allowed_tools)

                allowed_names = [t["function"]["name"] for t in allowed_tools]
                print(f"    [FSM] Allowed tools: {allowed_names}")

                if self._fsm_state == FSMState.S4_5_PENDING_SEARCH_CLEANUP:
                    print(f"    [FSM] state: {self._fsm_state.name}, "
                          f"allowed search msg_ids: {sorted(self._allowed_msg_ids_for_s4_5)}, "
                          f"pending_search: {self._search_call_count - self._search_delete_count}")

                if self._fsm_state == FSMState.S4_6_PENDING_PLAN_CLEANUP:
                    print(f"    [FSM] state: {self._fsm_state.name}, "
                          f"allowed plan-assistant msg_ids: {sorted(self._allowed_msg_ids_for_s4_6)}, "
                          f"pending_plan: {self._plan_call_count - self._plan_delete_count}")

                if self._fsm_state in (FSMState.S5_NOTED, FSMState.S5_5_PENDING_DELETE):
                    print(f"    [FSM] state: {self._fsm_state.name}, "
                          f"allowed S5 read-result msg_ids: {sorted(self._allowed_msg_ids_for_s5)}, "
                          f"allowed S5.5 note/memory assistant msg_ids: {sorted(self._allowed_msg_ids_for_s5_5)}")

                if len(allowed_tools) == 1:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": allowed_tools[0]["function"]["name"]},
                    }
                elif self._use_required_tool_choice:
                    tool_choice = "required"
                else:
                    tool_choice = None

                try:
                    resp = self._call_llm_api_fsm(api_payload, allowed_tools, tool_choice)
                except Exception as e:
                    err = f"LLM API failed after retries: {type(e).__name__}: {e}"
                    if self.auto_delete_on_context_overflow and (
                        "maximum context length" in str(e)
                        or "context_length_exceeded" in str(e)
                    ):
                        if self._recover_context_overflow():
                            print(f"[RECOVERY] {err}")
                            continue
                    print("[ERROR]", err)
                    mandatory_action = None
                    mandatory_store = None
                    if self._fsm_state == FSMState.S6_REQUIRE_READ_NOTE:
                        mandatory_action = "readNote"
                        mandatory_store = getattr(self.state_manager, "simple_notes", {})
                    elif self._fsm_state == FSMState.S6_5_REQUIRE_LOAD_MEMORY:
                        mandatory_action = "loadMemory"
                        mandatory_store = getattr(self.state_manager, "notes", {})
                    if mandatory_action is not None:
                        if mandatory_store:
                            key = str(list(mandatory_store.keys())[-1])
                            auto_params = {"key": key}
                            auto_result = self._execute_tool(mandatory_action, auto_params)
                            self.ctx_counter += 1
                            aid = self.ctx_counter
                            self.ctx_counter += 1
                            tid = self.ctx_counter
                            call_id = f"auto_mandatory_review_{aid}"
                            self.full_history.append({
                                "role": "assistant", "content": [], "msg_id": aid,
                                "tool_calls": [{
                                    "id": call_id, "type": "function",
                                    "function": {
                                        "name": mandatory_action,
                                        "arguments": json.dumps(auto_params, ensure_ascii=False),
                                    },
                                }],
                            })
                            self.full_history.append({
                                "role": "tool", "content": deepcopy(auto_result),
                                "msg_id": tid, "msg_id(invoking_assistant)": aid,
                                "tool_use_id": call_id, "tool_name": mandatory_action,
                            })
                            self._fsm_transition(
                                mandatory_action, auto_result, tid, aid, auto_params)
                            print(f"[RECOVERY] Completed mandatory {mandatory_action} "
                                  f"deterministically with key='{key}' after API failure.")
                        else:
                            self._fsm_state = FSMState.S7_REVIEWING
                            print("[RECOVERY] Mandatory review store is empty; "
                                  "advancing to S7 after API failure.")
                        turn += 1
                        continue
                    self.full_history.append({
                        "role": "tool",
                        "content": {"status": "error", "message": err},
                        "msg_id": self.ctx_counter + 1,
                        "msg_id(invoking_assistant)": self.ctx_counter,
                        "tool_use_id": "api_failure",
                        "tool_name": "finish"
                    })
                    self.tool_library.clearCurrentDocument()
                    return api_payload

                self.ctx_counter += 1
                thought, action, params, tool_use_id, stop_reason = self._parse_llm_output(resp)
                msg_id = self.ctx_counter

                if action in ("readChunk", "readMultiChunks"):
                    self._autocorrect_invalid_read(action, params)
                    self._autocorrect_duplicate_read(action, params)

                raw_tool_calls = resp.choices[0].message.tool_calls
                if not raw_tool_calls and stop_reason == "tool_calls" and action and tool_use_id:
                    raw_tool_calls = [{
                        "id": tool_use_id,
                        "type": "function",
                        "function": {
                            "name": action,
                            "arguments": json.dumps(params, ensure_ascii=False),
                        },
                    }]
                elif raw_tool_calls and len(raw_tool_calls) > 1:
                    print(f"    [FSM] Model emitted {len(raw_tool_calls)} tool_calls; "
                          f"keeping only the first one.")
                    raw_tool_calls = [raw_tool_calls[0]]
                if raw_tool_calls:
                    tc = raw_tool_calls[0]
                    if hasattr(tc, "model_dump"):
                        tc = tc.model_dump()
                    elif isinstance(tc, dict):
                        tc = deepcopy(tc)
                    else:
                        tc = json.loads(json.dumps(tc, default=str))
                    tc.setdefault("type", "function")
                    tc.setdefault("id", tool_use_id or f"chatcmpl-tool-{uuid.uuid4().hex[:32]}")
                    tc.setdefault("function", {})
                    if action:
                        tc["function"]["name"] = action
                    tc["function"]["arguments"] = json.dumps(params or {}, ensure_ascii=False)
                    raw_tool_calls = [tc]

                assistant_entry = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": thought or ""}],
                    "tool_calls": raw_tool_calls,
                    "msg_id": msg_id
                }
                print("[RUN] Assistant:", thought)

                if stop_reason == 'tool_calls' and action:
                    print(f"[RUN] Assistant action: Call tool `{action}`, parameters: {params}")

                    if action not in allowed_names:
                        result = {
                            "error": f"Tool '{action}' is not allowed in FSM state "
                                     f"'{self._fsm_state.name}'. "
                                     f"Allowed tools: {allowed_names}. "
                                     f"Please call one of the allowed tools."
                        }
                        print(f"    [FSM] REJECTED tool '{action}' -- not in allowed set.")

                    elif action in ("readNote", "loadMemory"):
                        key_error = self._validate_or_autocorrect_store_key(action, params)
                        if key_error:
                            result = key_error
                            print(f"    [FSM] REJECTED {action}: {key_error['error']}")
                        elif action not in self.tool_names:
                            result = {"error": f"Tool '{action}' not found."}
                        else:
                            try:
                                result = self._execute_tool(action, params)
                            except Exception as exc:
                                result = {"error": f"Tool '{action}' raised {type(exc).__name__}: {exc}"}
                                print(f"    [FSM] Tool execution error: {result['error']}")

                    elif self._fsm_state == FSMState.S4_5_PENDING_SEARCH_CLEANUP and \
                            action in ("deleteContext", "truncateContext", "summarizeContext", "compressContext"):
                        err_msg = self._validate_s4_5_msg_id(params)
                        if err_msg:
                            result = {"error": err_msg}
                            print(f"    [FSM] REJECTED {action}: {err_msg}")
                        elif action not in self.tool_names:
                            result = {"error": f"Tool '{action}' not found."}
                        else:
                            try:
                                result = self._execute_tool(action, params)
                            except Exception as exc:
                                result = {"error": f"Tool '{action}' raised {type(exc).__name__}: {exc}"}
                                print(f"    [FSM] Tool execution error: {result['error']}")

                    elif self._fsm_state == FSMState.S4_6_PENDING_PLAN_CLEANUP and \
                            action in ("deleteContext", "truncateContext", "summarizeContext", "compressContext"):
                        err_msg = self._validate_s4_6_msg_id(params)
                        if err_msg:
                            result = {"error": err_msg}
                            print(f"    [FSM] REJECTED {action}: {err_msg}")
                        elif action not in self.tool_names:
                            result = {"error": f"Tool '{action}' not found."}
                        else:
                            try:
                                result = self._execute_tool(action, params)
                            except Exception as exc:
                                result = {"error": f"Tool '{action}' raised {type(exc).__name__}: {exc}"}
                                print(f"    [FSM] Tool execution error: {result['error']}")

                    elif self._fsm_state in (FSMState.S4_READ, FSMState.S5_NOTED, FSMState.S5_5_PENDING_DELETE) and \
                            action in ("deleteContext", "truncateContext", "summarizeContext", "compressContext"):
                        if self._fsm_state == FSMState.S4_READ:
                            err_msg = self._validate_s4_read_cleanup_msg_id(params)
                        elif self._fsm_state == FSMState.S5_NOTED:
                            err_msg = self._validate_s5_msg_id(params)
                        else:
                            err_msg = self._validate_s5_5_msg_id(params)
                        if err_msg:
                            result = {"error": err_msg}
                            print(f"    [FSM] REJECTED {action}: {err_msg}")
                        elif action not in self.tool_names:
                            result = {"error": f"Tool '{action}' not found."}
                        else:
                            try:
                                result = self._execute_tool(action, params)
                            except Exception as exc:
                                result = {"error": f"Tool '{action}' raised {type(exc).__name__}: {exc}"}
                                print(f"    [FSM] Tool execution error: {result['error']}")

                    elif action not in self.tool_names:
                        result = {"error": f"Tool '{action}' not found."}
                    else:
                        try:
                            if action == "plan":
                                result = self._execute_plan_tool(params)
                            else:
                                result = self._execute_tool(action, params)
                        except Exception as exc:
                            result = {"error": f"Tool '{action}' raised {type(exc).__name__}: {exc}"}
                            print(f"    [FSM] Tool execution error: {result['error']}")

                    is_error = "error" in result

                    self.ctx_counter += 1
                    msg_id_tool = self.ctx_counter
                    tool_entry = {
                        "role": "tool",
                        "content": deepcopy(result),
                        "msg_id": msg_id_tool,
                        "msg_id(invoking_assistant)": msg_id,
                        "tool_use_id": tool_use_id,
                        "tool_name": action
                    }

                    result_preview = json.dumps(result, ensure_ascii=False)
                    if len(result_preview) > 200:
                        result_preview = result_preview[:200] + "..."
                    print(f"[RUN] Tool result (ID: {msg_id_tool}): {result_preview}")

                    if is_error:
                        is_msg_id_state = self._fsm_state in (
                            FSMState.S4_READ,
                            FSMState.S4_5_PENDING_SEARCH_CLEANUP,
                            FSMState.S4_6_PENDING_PLAN_CLEANUP,
                            FSMState.S5_NOTED,
                            FSMState.S5_5_PENDING_DELETE,
                        )
                        err_str = result.get("error", "")
                        is_msg_id_err = is_msg_id_state and "msg_id" in err_str and "not allowed" in err_str

                        if is_msg_id_err:
                            self._msg_id_error_retries += 1
                            print(f"    [FSM] msg_id validation error "
                                  f"(attempt {self._msg_id_error_retries}/"
                                  f"{self._max_msg_id_error_retries})")

                        if is_msg_id_err and self._msg_id_error_retries >= self._max_msg_id_error_retries:
                            if self._fsm_state == FSMState.S4_READ:
                                auto_msg_id = min(self._readchunk_tool_msg_ids) \
                                    if self._readchunk_tool_msg_ids else None
                            elif self._fsm_state == FSMState.S4_5_PENDING_SEARCH_CLEANUP:
                                auto_msg_id = min(self._allowed_msg_ids_for_s4_5) \
                                    if self._allowed_msg_ids_for_s4_5 else None
                            elif self._fsm_state == FSMState.S4_6_PENDING_PLAN_CLEANUP:
                                auto_msg_id = min(self._allowed_msg_ids_for_s4_6) \
                                    if self._allowed_msg_ids_for_s4_6 else None
                            elif self._fsm_state == FSMState.S5_NOTED:
                                auto_msg_id = min(self._allowed_msg_ids_for_s5) \
                                    if self._allowed_msg_ids_for_s5 else None
                            else:
                                auto_msg_id = min(self._allowed_msg_ids_for_s5_5) \
                                    if self._allowed_msg_ids_for_s5_5 else None

                            if auto_msg_id is not None:
                                print(f"    [FSM] Auto-correcting: executing "
                                      f"deleteContext(msg_id={auto_msg_id}) "
                                      f"after {self._msg_id_error_retries} failed attempts.")
                                self.ctx_counter -= 2

                                auto_params = {"msg_id": auto_msg_id}
                                try:
                                    auto_result = self._execute_tool("deleteContext", auto_params)
                                except Exception as exc:
                                    auto_result = {"error": f"deleteContext raised {type(exc).__name__}: {exc}"}
                                    print(f"    [FSM] Auto-correct tool error: {auto_result['error']}")

                                if "error" not in auto_result:
                                    self.ctx_counter += 1
                                    auto_assistant_msg_id = self.ctx_counter
                                    auto_tool_use_id = f"auto_correct_{auto_assistant_msg_id}"
                                    auto_assistant_entry = {
                                        "role": "assistant",
                                        "content": [],
                                        "tool_calls": [{
                                            "id": auto_tool_use_id,
                                            "type": "function",
                                            "function": {
                                                "name": "deleteContext",
                                                "arguments": json.dumps(auto_params,
                                                                        ensure_ascii=False),
                                            },
                                        }],
                                        "msg_id": auto_assistant_msg_id,
                                    }
                                    self.ctx_counter += 1
                                    auto_tool_msg_id = self.ctx_counter
                                    auto_tool_entry = {
                                        "role": "tool",
                                        "content": deepcopy(auto_result),
                                        "msg_id": auto_tool_msg_id,
                                        "msg_id(invoking_assistant)": auto_assistant_msg_id,
                                        "tool_use_id": auto_tool_use_id,
                                        "tool_name": "deleteContext",
                                    }
                                    self.full_history.append(auto_assistant_entry)
                                    self.full_history.append(auto_tool_entry)
                                    self._no_tool_retries = 0
                                    self._msg_id_error_retries = 0
                                    self._pending_error_messages.clear()

                                    self._fsm_transition(
                                        "deleteContext", auto_result,
                                        auto_tool_msg_id, auto_assistant_msg_id,
                                        auto_params)

                                    auto_preview = json.dumps(auto_result, ensure_ascii=False)
                                    if len(auto_preview) > 200:
                                        auto_preview = auto_preview[:200] + "..."
                                    print(f"[RUN] Auto-correct result (ID: {auto_tool_msg_id}): "
                                          f"{auto_preview}")
                                    turn += 1
                                    continue
                                else:
                                    print(f"    [FSM] Auto-correct also failed, "
                                          f"falling back to normal error retry.")
                                    self.ctx_counter += 2
                            else:
                                print(f"    [FSM] Auto-correct: no allowed msg_ids available, "
                                      f"cannot auto-correct.")

                        if not is_msg_id_err:
                            self._tool_error_retries += 1
                            print(f"    [FSM] Tool error "
                                  f"(attempt {self._tool_error_retries}/"
                                  f"{self._max_tool_error_retries}): {err_str}")
                            if (self._fsm_state == FSMState.S7_REVIEWING
                                    and action in ("loadMemory", "readNote", "restoreContext")
                                    and not force_finish):
                                self.ctx_counter -= 2
                                self._pending_error_messages.clear()
                                self._tool_error_retries = 0
                                force_finish = True
                                self._inject_force_finish_messages()
                                print("[INFO] Invalid optional review request in S7; "
                                      "switching to finish-only mode.")
                                turn += 1
                                continue
                            if self._tool_error_retries >= self._max_tool_error_retries:
                                self.full_history.append(assistant_entry)
                                self.snapshots.append(self._build_api_payload(
                                    keep_think=True, inject_retry_hint=False))
                                self.full_history.append(tool_entry)
                                print(f"[INFO] Process terminated: tool errors "
                                      f"{self._tool_error_retries} times in a row "
                                      f"(last error: {err_str}).")
                                break
                        self._pending_error_messages.append(assistant_entry)
                        self._pending_error_messages.append(tool_entry)
                        self.ctx_counter -= 2
                        print(f"    [FSM] Error result NOT recorded in full_history/snapshots.")
                    else:
                        self.full_history.append(assistant_entry)
                        self.snapshots.append(self._build_api_payload(keep_think=True, inject_retry_hint=False))
                        self.full_history.append(tool_entry)
                        self._no_tool_retries = 0
                        self._msg_id_error_retries = 0
                        self._tool_error_retries = 0

                        if action in ("searchEngine", "semanticSearch", "hybridSearch"):
                            self.search_call_counter += 1
                            if (self.max_search_calls is not None
                                    and self.search_call_counter >= self.max_search_calls
                                    and not force_finish
                                    and not self._pending_search_limit_force_finish):
                                print(f"[INFO] Search call limit reached "
                                      f"({self.max_search_calls}); "
                                      f"force-finish pending until FSM reaches S7 after mandatory review "
                                      f"(current state: {self._fsm_state.name}).")
                                self._pending_search_limit_force_finish = True

                        self._fsm_transition(action, result, msg_id_tool, msg_id, params)

                        if action == "finish":
                            self._fsm_state = FSMState.DONE
                            print(f"\n--- Final Answer --- \n"
                                  f"{result.get('final_answer', 'No final answer provided.')}")
                            break
                else:
                    self._no_tool_retries += 1
                    if (self._fsm_state == FSMState.S4_READ
                            and self._readchunk_tool_msg_ids
                            and self._no_tool_retries >= max(1, self.max_no_tool_retries - 1)):
                        auto_msg_id = min(self._readchunk_tool_msg_ids)
                        self.ctx_counter -= 1
                        if self._auto_delete_context_msg(
                                auto_msg_id,
                                "the model produced no tool call in S4 after reading chunks"):
                            turn += 1
                            continue
                        self.ctx_counter += 1

                    if (self._fsm_state == FSMState.S3_SEARCHED
                            and self._last_retrieved_chunks
                            and self._no_tool_retries >= max(1, self.max_no_tool_retries - 1)):
                        first_id = self._last_retrieved_chunks[0].get("chunk_id")
                        if first_id is not None:
                            self.ctx_counter -= 1
                            self._pending_error_messages.clear()
                            params = {"chunk_id": int(first_id)}
                            result = self._execute_tool("readChunk", params)
                            if "error" not in result:
                                self.ctx_counter += 1
                                aid = self.ctx_counter
                                self.ctx_counter += 1
                                tid = self.ctx_counter
                                self.full_history.append({
                                    "role": "assistant", "content": [], "msg_id": aid,
                                    "tool_calls": [{
                                        "id": f"auto_read_{aid}", "type": "function",
                                        "function": {"name": "readChunk", "arguments": json.dumps(params)},
                                    }],
                                })
                                self.full_history.append({
                                    "role": "tool", "content": deepcopy(result), "msg_id": tid,
                                    "msg_id(invoking_assistant)": aid,
                                    "tool_use_id": f"auto_read_{aid}", "tool_name": "readChunk",
                                })
                                self._no_tool_retries = 0
                                self._fsm_transition("readChunk", result, tid, aid, params)
                                print(f"[RUN] Auto-read highest-ranked search candidate chunk_id={first_id}.")
                                turn += 1
                                continue

                    if (self._fsm_state == FSMState.S7_REVIEWING
                            and not force_finish
                            and self._no_tool_retries >= max(1, self.max_no_tool_retries - 1)):
                        self.ctx_counter -= 1
                        self._pending_error_messages.clear()
                        force_finish = True
                        self._no_tool_retries = 0
                        self._inject_force_finish_messages()
                        print("[INFO] Switching to force-finish mode after "
                              "repeated no-tool responses in S7_REVIEWING.")
                        turn += 1
                        continue

                    if self._no_tool_retries >= self.max_no_tool_retries:
                        self.full_history.append(assistant_entry)
                        self.snapshots.append(self._build_api_payload(keep_think=True, inject_retry_hint=False))
                        print(f"[INFO] Process terminated: model produced no tool call "
                              f"{self._no_tool_retries} times in a row "
                              f"(stop_reason='{stop_reason}').")
                        break
                    else:
                        nudge_msg = {
                            "role": "user",
                            "content": self.NO_TOOL_NUDGE_USER_PROMPT,
                        }
                        self._pending_error_messages.append(assistant_entry)
                        self._pending_error_messages.append(nudge_msg)
                        self.ctx_counter -= 1
                        print(f"[INFO] No tool call (attempt {self._no_tool_retries}/"
                              f"{self.max_no_tool_retries}), retrying...")

                turn += 1

            if turn > self.max_turns:
                print(f"[INFO] Reached max rounds {self.max_turns}, stopping execution.")

            self.snapshots.append(self._build_api_payload(keep_think=True, inject_retry_hint=False))
            self.tool_library.clearCurrentDocument()
            return self._build_api_payload()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl-C). Exiting gracefully...")
            self.tool_library.clearCurrentDocument()
            return self._build_api_payload()

    def _call_llm_api_fsm(self, messages, tools, tool_choice=None, max_tokens=None):
        """
        Call the LLM API with a specific set of tools (FSM-filtered).
        Call the model with the tools allowed by the current FSM state.

        Args:
            messages: chat-completions messages list.
            tools: list of tool specs (if empty/None, no `tools` field is
                sent at all).
            tool_choice: optional tool_choice override.
            max_tokens: optional override for the generation length cap. If
                None, falls back to ``self.max_output_tokens``. Used by the
                ``plan`` tool to raise its own budget independently of the
                main turn.
        """
        from openai import (
            APIError, RateLimitError, APITimeoutError,
            APIConnectionError, APIStatusError,
        )

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.max_output_tokens
        )

        body_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.topp,
            "max_tokens": effective_max_tokens,
        }
        sampling_seed_base = getattr(self, "_sampling_seed_base", None)
        if sampling_seed_base is not None:
            body_kwargs["seed"] = (
                int(sampling_seed_base) + int(self.api_call_counter)
            ) & 0x7FFFFFFF
        if tools:
            body_kwargs["tools"] = tools
        if tool_choice is not None:
            body_kwargs["tool_choice"] = tool_choice
        extra_body = dict(getattr(self, "openai_extra_body", {}) or {})
        if self.topk:
            extra_body["top_k"] = self.topk
        if extra_body:
            body_kwargs["extra_body"] = extra_body

        tries = 0
        max_tries = max(0, int(os.getenv("CONTEXTPILOT_API_MAX_RETRIES", "3")))
        while True:
            try:
                resp = self.vllm_client.chat.completions.create(**body_kwargs)
                self.api_call_counter += 1
                if getattr(self, "logger", None):
                    self.logger.log_api_call(body_kwargs, resp.model_dump(), self.api_call_counter)
                return resp
            except (APIError, RateLimitError, APITimeoutError,
                    APIConnectionError, APIStatusError) as e:
                status_code = getattr(e, "status_code", None)
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise
                if tries >= max_tries:
                    raise
                wait = 5 * (2 ** tries)
                print(f"[API] {e} - retrying in {wait}s")
                time.sleep(wait)
                tries += 1

    def get_fsm_summary(self):
        """Return a summary of the FSM execution for logging."""
        return {
            "final_state": self._fsm_state.name,
            "last_retrieved_chunks_count": len(self._last_retrieved_chunks),
            "read_chunk_count_this_cycle": self._read_chunk_count_this_cycle,
            "s5_delete_count": self._s5_delete_count,
        }

    def set_use_more_plan(self, flag: bool) -> None:
        """Toggle the extra post-read plan step (S3_5).

        When True AND the `plan` tool is available, every readChunk /
        readMultiChunks call routes the FSM to S3_5, where the model must
        call `plan` (to reflect on the just-read chunk(s)) before it is
        allowed to take notes.
        """
        self._use_more_plan = bool(flag)
        print(f"[INFO] use_more_plan set to {self._use_more_plan}")

    def set_use_required_tool_choice(self, flag: bool) -> None:
        """Toggle whether FSM requests enforce tool_choice=\"required\".

        When False, the FSM still filters the tools list for each state, but
        it stops sending the explicit OpenAI-style `tool_choice="required"`
        override to the backend.
        """
        self._use_required_tool_choice = bool(flag)
        print(f"[INFO] use_required_tool_choice set to {self._use_required_tool_choice}")
