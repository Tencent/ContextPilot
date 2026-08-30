
"""
StateLM Tools - Tools for document manipulation, note-taking, and context management.

These tools provide stateful operations for long-context agent workflows, including:
- Document loading and analysis
- Chunk-based document indexing with Elasticsearch
- Note-taking and knowledge management
- Context budget tracking
"""

import copy
import json
import logging
import os
import random
import re
import threading
import uuid
from typing import Any, Optional

from elasticsearch import Elasticsearch, helpers

from verl.experimental.agent_loop.statelm_agent_loop import render_context
from verl.tools.base_tool import BaseTool
from verl.tools.contextpilot_memory import ContextPilotMemoryState
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DocStateManager:
    """Shared per-trajectory document state for StateLM tools."""

    def __init__(self, tokenizer, document_content: str = ""):
        self.tokenizer = tokenizer
        self.document_content = document_content or ""
        self.index: list[dict[str, Any]] = []
        self.keywords_searched: set[str] = set()
        self.chunk_pointer = [-1, 0]
        self.scan_mode = False
        self.last_scanned_chunk_id = -1

        self._es: Optional[Elasticsearch] = None
        self._es_index_name: str = os.getenv("ES_INDEX_NAME", "lc_agent_document")
        self._es_host: str = os.getenv("ES_HOST", "http://localhost:9200")
        self._es_user: Optional[str] = os.getenv("ES_USER")
        self._es_pass: Optional[str] = os.getenv("ES_PASS")
        self._es_api_key: Optional[str] = os.getenv("ES_API_KEY")
        self._es_ca_cert: Optional[str] = os.getenv("ES_CA_CERT")
        self._doc_id: Optional[str] = None
        self._owns_doc_id: bool = True
        self._orphan_doc_ids: set[str] = set()

        if self.document_content:
            self.encoded_doc = self.tokenizer(
                self.document_content,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
        else:
            self.encoded_doc = {"input_ids": [], "offset_mapping": []}

    def _get_es(self) -> Elasticsearch:
        if self._es is None:
            kwargs = {}
            if self._es_api_key:
                kwargs["api_key"] = self._es_api_key
            elif self._es_user and self._es_pass:
                kwargs["basic_auth"] = (self._es_user, self._es_pass)
            if self._es_ca_cert:
                kwargs["ca_certs"] = self._es_ca_cert
            self._es = Elasticsearch(self._es_host, **kwargs)
        return self._es

    def _ensure_es_index(self):
        es = self._get_es()
        if es.indices.exists(index=self._es_index_name):
            return
        try:
            es.indices.create(
                index=self._es_index_name,
                settings={"index": {"analysis": {"analyzer": {"default": {"type": "standard"}}}}},
                mappings={
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "integer"},
                        "content": {"type": "text"},
                        "start_pos": {"type": "integer"},
                        "end_pos": {"type": "integer"},
                    }
                },
            )
        except Exception as create_error:
            try:
                index_now_exists = es.indices.exists(index=self._es_index_name)
            except Exception:
                raise create_error
            if index_now_exists:
                logger.info("Elasticsearch index %s was created concurrently.", self._es_index_name)
                return
            raise create_error

    def _bulk_index_chunks(
        self,
        *,
        doc_id: Optional[str] = None,
        chunks: Optional[list[dict[str, Any]]] = None,
    ):
        es = self._get_es()
        target_doc_id = doc_id or self._doc_id
        target_chunks = self.index if chunks is None else chunks
        if not target_doc_id:
            raise RuntimeError("Cannot index chunks without a document id.")
        actions = (
            {
                "_op_type": "index",
                "_index": self._es_index_name,
                "_id": f"{target_doc_id}:{chunk['chunk_id']}",
                "doc_id": target_doc_id,
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"],
            }
            for chunk in target_chunks
        )
        helpers.bulk(es, actions)
        es.indices.refresh(index=self._es_index_name)

    def _delete_document(self, doc_id: str) -> None:
        self._get_es().delete_by_query(
            index=self._es_index_name,
            query={"term": {"doc_id": doc_id}},
            refresh=True,
        )

    def close(self) -> None:
        if self._es is not None:
            try:
                self._es.close()
            finally:
                self._es = None

    def clear_current_document(self):
        doc_ids = set(self._orphan_doc_ids)
        if self._doc_id and self._owns_doc_id:
            doc_ids.add(self._doc_id)
        if not doc_ids:
            return {"message": "No active document to clear."}
        for doc_id in doc_ids:
            self._delete_document(doc_id)
        self.index = []
        self.keywords_searched = set()
        self._doc_id = None
        self._owns_doc_id = True
        self._orphan_doc_ids.clear()
        self.chunk_pointer = [-1, 0]
        self.scan_mode = False
        self.last_scanned_chunk_id = -1
        return {"message": "Cleared current document."}


class _StateLMTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()

    def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

    @staticmethod
    def _json(data: Any) -> ToolResponse:
        return ToolResponse(text=json.dumps(data, ensure_ascii=False))


class PlanTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        strategy = parameters.get("strategy") if isinstance(parameters, dict) else None
        if not isinstance(strategy, str) or not strategy.strip():
            return self._json({"error": "plan tool requires a non-empty 'strategy' argument."}), 0.0, {}
        return self._json({"status": "success", "message": "Plan recorded. Please proceed according to your strategy above."}), 0.0, {}


class AnalyzeTextTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        doc_state_manager = kwargs.get("doc_state_manager")
        if not doc_state_manager:
            return self._json({"error": "Document State Manager not available"}), 0.0, {}
        return self._json({"file_name": "attached_document.txt", "total_tokens": len(doc_state_manager.encoded_doc["input_ids"])}), 0.0, {}


class LoadDocumentTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        doc_state_manager = kwargs.get("doc_state_manager")
        if not doc_state_manager:
            return self._json({"error": "Document State Manager not available"}), 0.0, {}
        if not doc_state_manager.document_content:
            return self._json({"error": "Document content is empty."}), 0.0, {}
        return self._json({"document_content": doc_state_manager.document_content}), 0.0, {}


class BuildIndexTool(_StateLMTool):
    _BOUNDARY_REGEXES = [
        re.compile(r"\n\s*\n"),
        re.compile(r"[.!?][\"')\]]?\s"),
        re.compile(r"[\u3002\uff01\uff1f]"),
        re.compile(r"\n"),
    ]
    _SECTION_HEADING_RE = re.compile(
        r"(?im)^\s*((?:chapter|book|part|volume)\s+"
        r"(?:[0-9]+|[ivxlcdm]+|[a-z][a-z '\-]{1,60}))\s*$"
    )

    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        doc_state_manager = kwargs.get("doc_state_manager")
        if not doc_state_manager:
            return self._json({"error": "Document State Manager not available"}), 0.0, {}

        chunk_size = int(parameters.get("chunk_size", 4000) or 4000)
        overlap = int(parameters.get("overlap", 0) or 0)
        if chunk_size <= 0:
            return self._json({"error": "chunk_size must be > 0"}), 0.0, {}
        if overlap < 0:
            return self._json({"error": "overlap must be >= 0"}), 0.0, {}
        if overlap >= chunk_size:
            return self._json({"error": "overlap must be < chunk_size"}), 0.0, {}

        boundary_backtrack = int(parameters.get("boundary_backtrack", 400) or 0)
        boundary_backtrack = max(0, boundary_backtrack)
        max_safe_backtrack = max(0, chunk_size - max(1, overlap + 1))
        boundary_backtrack = min(boundary_backtrack, max_safe_backtrack)

        input_ids = doc_state_manager.encoded_doc["input_ids"]
        offsets = doc_state_manager.encoded_doc["offset_mapping"]
        if not input_ids:
            return self._json({"error": "Document content is empty."}), 0.0, {}

        new_index: list[dict[str, Any]] = []
        new_doc_id = uuid.uuid4().hex
        start_token = 0
        chunk_id = 0
        current_section_hint = None
        while start_token < len(input_ids):
            hard_end = min(start_token + chunk_size, len(input_ids))
            end_token = hard_end
            if boundary_backtrack > 0 and hard_end < len(input_ids):
                lower_bound = max(start_token + 1, hard_end - boundary_backtrack)
                hard_char_end = offsets[hard_end - 1][1]
                for token_index in range(hard_end - 1, lower_bound - 1, -1):
                    token_char_start = offsets[token_index][0]
                    token_char_end = offsets[token_index][1]
                    if token_char_end <= token_char_start:
                        continue
                    segment = doc_state_manager.document_content[token_char_start:hard_char_end]
                    if any(
                        match and match.start() <= token_char_end - token_char_start - 1
                        for match in (pattern.search(segment) for pattern in self._BOUNDARY_REGEXES)
                    ):
                        end_token = token_index + 1
                        break
            chunk_offsets = offsets[start_token:end_token]
            char_start = chunk_offsets[0][0]
            char_end = chunk_offsets[-1][1]
            chunk_content = doc_state_manager.document_content[char_start:char_end]
            heading_window = doc_state_manager.document_content[max(0, char_start - 256):char_end]
            headings = self._SECTION_HEADING_RE.findall(heading_window)
            if headings:
                current_section_hint = " ".join(headings[-1].split())[:120]
            new_index.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "start_pos": start_token,
                    "end_pos": end_token,
                    "section_hint": current_section_hint,
                }
            )
            chunk_id += 1
            next_start = end_token - overlap
            if next_start <= start_token:
                next_start = start_token + 1
            start_token = next_start

        old_doc_id = doc_state_manager._doc_id
        old_doc_owned = bool(getattr(doc_state_manager, "_owns_doc_id", True))
        try:
            doc_state_manager._ensure_es_index()
            doc_state_manager._bulk_index_chunks(doc_id=new_doc_id, chunks=new_index)
        except Exception as exc:
            try:
                doc_state_manager._delete_document(new_doc_id)
            except Exception as cleanup_exc:
                doc_state_manager._orphan_doc_ids.add(new_doc_id)
                logger.warning(
                    "Failed to clean temporary Elasticsearch document %s after build failure: %s",
                    new_doc_id,
                    cleanup_exc,
                )
            return self._json({"error": f"Failed to (re)build Elasticsearch index: {exc}"}), 0.0, {}

        doc_state_manager.index = new_index
        doc_state_manager._doc_id = new_doc_id
        doc_state_manager._owns_doc_id = True
        doc_state_manager.keywords_searched = set()
        doc_state_manager.chunk_pointer = [-1, 0]
        doc_state_manager.scan_mode = False
        doc_state_manager.last_scanned_chunk_id = -1
        if old_doc_id and old_doc_owned:
            try:
                doc_state_manager._delete_document(old_doc_id)
                doc_state_manager._orphan_doc_ids.discard(old_doc_id)
            except Exception as cleanup_exc:
                doc_state_manager._orphan_doc_ids.add(old_doc_id)
                logger.warning(
                    "Deferred cleanup of previous Elasticsearch document %s: %s",
                    old_doc_id,
                    cleanup_exc,
                )

        return self._json(
            {
                "index_id": "document_index",
                "total_chunks": len(new_index),
                "first_chunk_id": 0,
                "last_chunk_id": len(new_index) - 1,
            }
        ), 0.0, {}


class ReadChunkTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        doc_state_manager = kwargs.get("doc_state_manager")
        if not doc_state_manager:
            return self._json({"error": "Document State Manager not available"}), 0.0, {}
        if not doc_state_manager.index:
            return self._json({"error": "Index not built. Please call 'buildIndex' first."}), 0.0, {}

        if parameters.get("enable_scan") is not None:
            doc_state_manager.scan_mode = bool(parameters.get("enable_scan"))

        if doc_state_manager.scan_mode:
            next_chunk_id = doc_state_manager.last_scanned_chunk_id + 1
            if next_chunk_id >= len(doc_state_manager.index):
                return self._json({"error": f"No more chunks available. Last chunk ({len(doc_state_manager.index) - 1}) already retrieved."}), 0.0, {}
            doc_state_manager.last_scanned_chunk_id = next_chunk_id
            return self._json(
                {
                    "retrieved_chunk": [doc_state_manager.index[next_chunk_id]],
                    "chunk_id": next_chunk_id,
                    "reading_progress": f"{next_chunk_id + 1}/{len(doc_state_manager.index)}",
                }
            ), 0.0, {}

        chunk_id = parameters.get("chunk_id")
        if chunk_id is None:
            return self._json({"error": "chunk_id is required when enable_scan is false."}), 0.0, {}
        try:
            chunk_id = int(chunk_id)
        except (ValueError, TypeError):
            return self._json({"error": "chunk_id must be an integer."}), 0.0, {}
        if chunk_id < 0 or chunk_id >= len(doc_state_manager.index):
            return self._json({"error": f"Chunk_id: {chunk_id} is out of range. It must be between 0 and {len(doc_state_manager.index) - 1}."}), 0.0, {}
        doc_state_manager.last_scanned_chunk_id = chunk_id
        return self._json({"retrieved_chunk": [doc_state_manager.index[chunk_id]], "chunk_id": chunk_id}), 0.0, {}


class ReadMultiChunksTool(_StateLMTool):
    MAX_CHUNKS_ONCE = 3

    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        doc_state_manager = kwargs.get("doc_state_manager")
        if not doc_state_manager:
            return self._json({"error": "Document State Manager not available"}), 0.0, {}
        if not doc_state_manager.index:
            return self._json({"error": "Index not built. Please call 'buildIndex' first."}), 0.0, {}
        raw = parameters.get("chunk_ids")
        if raw is None:
            return self._json({"error": "chunk_ids is required (a list of integers)."}), 0.0, {}
        if not isinstance(raw, list):
            return self._json({"error": "chunk_ids must be a list of integers."}), 0.0, {}
        if not raw:
            return self._json({"error": "chunk_ids cannot be empty."}), 0.0, {}

        try:
            normalized = [int(cid) for cid in raw]
        except (ValueError, TypeError):
            return self._json({"error": "chunk_ids must be integers."}), 0.0, {}

        honored_ids = normalized[: self.MAX_CHUNKS_ONCE]
        max_valid = len(doc_state_manager.index) - 1
        for cid in honored_ids:
            if cid < 0 or cid > max_valid:
                return self._json({"error": f"Chunk_id: {cid} is out of range. It must be between 0 and {max_valid}."}), 0.0, {}

        if honored_ids:
            doc_state_manager.last_scanned_chunk_id = honored_ids[-1]
        result = {
            "retrieved_chunks": [doc_state_manager.index[cid] for cid in honored_ids],
            "chunk_ids": honored_ids,
            "requested_count": len(normalized),
            "returned_count": len(honored_ids),
            "max_chunks_once": self.MAX_CHUNKS_ONCE,
            "truncated": len(normalized) > self.MAX_CHUNKS_ONCE,
        }
        if result["truncated"]:
            result["notice"] = f"At most {self.MAX_CHUNKS_ONCE} chunk_ids are accepted; the input exceeded this limit, so chunk_ids after the {self.MAX_CHUNKS_ONCE}-th are ignored."
        return self._json(result), 0.0, {}


class SearchEngineTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        doc_state_manager = kwargs.get("doc_state_manager")
        if not doc_state_manager:
            return self._json({"error": "Document State Manager not available"}), 0.0, {}
        if not doc_state_manager.index:
            return self._json({"error": "Index not built. Please call buildIndex first."}), 0.0, {}
        if not doc_state_manager._doc_id:
            return self._json({"error": "No active document for this run. Please call buildIndex first."}), 0.0, {}

        raw_kw = parameters.get("keyword", "")
        if isinstance(raw_kw, list):
            keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
        else:
            keywords = [k.strip() for k in str(raw_kw).split(",") if k.strip()]
        if not keywords:
            return self._json({"error": "keyword cannot be empty."}), 0.0, {}
        doc_state_manager.keywords_searched.update(keywords)

        mode = (parameters.get("mode") or "or").lower()
        size = int(parameters.get("size", 50))
        fragment_size = int(parameters.get("fragment_size", 512))
        num_frags = int(parameters.get("num_fragments", 2))
        no_match_size = int(parameters.get("no_match_size", 200))
        min_should = parameters.get("minimum_should_match", "1")

        def _clause(kw: str):
            if " " in kw:
                return {"match_phrase": {"content": {"query": kw, "slop": 2}}}
            return {"match": {"content": {"query": kw, "operator": "and"}}}

        if mode == "and":
            query = {"bool": {"must": [_clause(kw) for kw in keywords], "filter": [{"term": {"doc_id": doc_state_manager._doc_id}}]}}
        elif mode == "or":
            query = {"bool": {"should": [_clause(kw) for kw in keywords], "minimum_should_match": min_should, "filter": [{"term": {"doc_id": doc_state_manager._doc_id}}]}}
        else:
            return self._json({"error": f"Search mode '{mode}' not supported."}), 0.0, {}

        try:
            res = doc_state_manager._get_es().search(
                index=doc_state_manager._es_index_name,
                query=query,
                highlight={
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fields": {
                        "content": {
                            "type": "unified",
                            "fragment_size": fragment_size,
                            "number_of_fragments": num_frags,
                            "no_match_size": no_match_size,
                        }
                    },
                },
                _source=["chunk_id"],
                size=size,
                track_total_hits=False,
            )
        except Exception as exc:
            return self._json({"error": f"Elasticsearch query failed: {exc}"}), 0.0, {}

        hits = res.get("hits", {}).get("hits", [])
        items = []
        for hit in hits:
            src = hit.get("_source", {}) or {}
            chunk_id = src.get("chunk_id")
            if chunk_id is None:
                continue
            items.append(
                {
                    "chunk_id": chunk_id,
                    "relevance_score": round(float(hit.get("_score", 0.0)), 3),
                    "highlights": hit.get("highlight", {}).get("content", []),
                    "section_hint": (
                        doc_state_manager.index[chunk_id].get("section_hint")
                        if isinstance(chunk_id, int) and 0 <= chunk_id < len(doc_state_manager.index)
                        else None
                    ),
                }
            )
        items.sort(key=lambda item: item["relevance_score"], reverse=True)
        if not items:
            return self._json({"retrieved_chunks": [], "message": "No matching content found.", "keywords": keywords}), 0.0, {}
        total = len(items)
        max_results = 10
        if len(items) > max_results:
            items = items[:max_results]
            return self._json(
                {
                    "retrieved_chunks": items,
                    "message": f"Showing the most relevant {max_results}/{total} chunks.",
                    "keywords": keywords,
                }
            ), 0.0, {}
        return self._json({"retrieved_chunks": items, "keywords": keywords}), 0.0, {}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _ensure_memory_stores(agent_data):
    if not hasattr(agent_data, "memories"):
        agent_data.memories = getattr(agent_data, "notes", {})
    if not hasattr(agent_data, "simple_notes"):
        agent_data.simple_notes = {}
    agent_data.notes = agent_data.memories
    return agent_data.memories, agent_data.simple_notes


def _get_memory_state(agent_data) -> ContextPilotMemoryState:
    memories, _ = _ensure_memory_stores(agent_data)
    state = getattr(agent_data, "_contextpilot_memory_state", None)
    if not isinstance(state, ContextPilotMemoryState) or state.notes is not memories:
        state = ContextPilotMemoryState(memories)
        agent_data._contextpilot_memory_state = state
    return state


class MemorizeTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        key = parameters.get("key")
        if not key:
            return self._json({"error": "Memory key is required"}), 0.0, {}
        result = _get_memory_state(agent_data).add_memory(
            str(key),
            parameters.get("content"),
            parameters.get("summary"),
            parameters.get("entities"),
            parameters.get("episodes"),
            parameters.get("relations"),
            parameters.get("source"),
        )
        return self._json(result), 0.0, {}


class LoadMemoryTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        key = str(parameters.get("key", ""))
        if not key:
            return self._json({"error": "Memory key is required"}), 0.0, {}
        max_related = int(parameters.get("max_related", 5) or 5)
        return self._json(_get_memory_state(agent_data).read_memory(key, max_related)), 0.0, {}


class UpdateMemoryTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        key = str(parameters.get("key", ""))
        mode = str(parameters.get("mode", "")).lower()
        if not key:
            return self._json({"error": "Memory key is required"}), 0.0, {}
        result = _get_memory_state(agent_data).update_memory(
            key,
            mode,
            parameters.get("new_content"),
            parameters.get("new_summary"),
            parameters.get("new_entities"),
            parameters.get("new_episodes"),
            parameters.get("new_relations"),
            parameters.get("new_source"),
        )
        return self._json(result), 0.0, {}


class NoteTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        _, simple_notes = _ensure_memory_stores(agent_data)
        key = parameters.get("key")
        if not key:
            return self._json({"error": "Note key is required"}), 0.0, {}
        simple_notes[str(key)] = {"summary": _safe_str(parameters.get("summary")), "full_content": _safe_str(parameters.get("content")), "content": _safe_str(parameters.get("content"))}
        return self._json({"status": "success", "key": str(key)}), 0.0, {}


class ReadNoteTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        _, simple_notes = _ensure_memory_stores(agent_data)
        key = str(parameters.get("key", ""))
        if not key:
            return self._json({"error": "Note key is required"}), 0.0, {}
        note = simple_notes.get(key)
        if note is None:
            return self._json({"error": f"Note '{key}' not found!"}), 0.0, {}
        return self._json(copy.deepcopy(note)), 0.0, {}


class UpdateNoteTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        _, simple_notes = _ensure_memory_stores(agent_data)
        key = str(parameters.get("key", ""))
        mode = str(parameters.get("mode", "")).lower()
        if not key:
            return self._json({"error": "Note key is required"}), 0.0, {}
        if key not in simple_notes:
            return self._json({"error": f"Note '{key}' not found!"}), 0.0, {}
        if mode == "delete":
            del simple_notes[key]
            return self._json({"status": "success", "key": key, "message": f"Note '{key}' deleted."}), 0.0, {}
        new_content = _safe_str(parameters.get("new_content"))
        new_summary = _safe_str(parameters.get("new_summary"))
        if mode == "append":
            simple_notes[key]["full_content"] = (simple_notes[key].get("full_content", "") + "\n" + new_content).strip()
            simple_notes[key]["content"] = simple_notes[key]["full_content"]
            message = f"Note '{key}' appended."
        elif mode == "overwrite":
            simple_notes[key]["full_content"] = new_content
            simple_notes[key]["content"] = new_content
            message = f"Note '{key}' overwritten."
        else:
            return self._json({"error": f"Invalid mode '{mode}'. Use 'append', 'overwrite', or 'delete'."}), 0.0, {}
        simple_notes[key]["summary"] = new_summary
        return self._json({"status": "success", "key": key, "message": message}), 0.0, {}


class CheckBudgetTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        tokenizer = kwargs.get("tokenizer")
        tool_schemas = kwargs.get("tool_schemas", [])
        max_context_exp = kwargs.get("max_context_exp", 32000)
        max_output_tokens = kwargs.get("max_output_tokens", 2048)
        max_turns = kwargs.get("max_turns", 50)
        if not agent_data or not tokenizer:
            return self._json({"error": "Required context not available"}), 0.0, {}
        memories, simple_notes = _ensure_memory_stores(agent_data)
        messages = render_context(
            agent_data.full_history,
            memories,
            agent_data.deleted_msg_ids,
            simple_notes=simple_notes,
            summarized_msg_ids=getattr(agent_data, "summarized_msg_ids", {}),
            truncated_msg_ids=getattr(agent_data, "truncated_msg_ids", {}),
        )
        tokenized_messages = tokenizer.apply_chat_template(messages, tools=tool_schemas, add_generation_prompt=False, tokenize=True)
        conv_rounds = agent_data.assistant_turns
        return self._json(
            {
                "conv_rounds": conv_rounds,
                "available_tokens": max(max_context_exp - len(tokenized_messages) - max_output_tokens, 0),
                "available_rounds": max(max_turns - conv_rounds, 0),
            }
        ), 0.0, {}


class GetContextStatsTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        doc_state_manager = kwargs.get("doc_state_manager")
        if not agent_data or not doc_state_manager:
            return self._json({"error": "Required context not available"}), 0.0, {}
        memories, simple_notes = _ensure_memory_stores(agent_data)
        return self._json(
            {
                "total_memories": len(memories),
                "memory_keys": list(memories.keys()),
                "total_notes": len(simple_notes),
                "note_keys": list(simple_notes.keys()),
                "index_chunks": len(doc_state_manager.index),
                "document_size": len(doc_state_manager.encoded_doc["input_ids"]),
                "searched_keywords": list(doc_state_manager.keywords_searched),
            }
        ), 0.0, {}


def _resolve_msg_entry(full_history: list[dict[str, Any]], msg_id: int):
    for i, message in enumerate(full_history):
        if message.get("msg_id") == msg_id:
            return i, message
    return None, None


def _extract_context_text(entry: dict[str, Any]) -> str:
    role = entry.get("role")
    content = entry.get("content")
    if role == "tool":
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    if role == "assistant":
        if isinstance(content, str):
            return content
        return " ".join(block.get("text", "") for block in (content or []) if isinstance(block, dict))
    return _safe_str(content)


class DeleteContextTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available. deleteContext requires agent_data in kwargs."}), 0.0, {}
        try:
            msg_id = int(parameters.get("msg_id"))
        except (ValueError, TypeError):
            return self._json({"error": "msg_id is required and must be an integer"}), 0.0, {}
        _, entry = _resolve_msg_entry(agent_data.full_history, msg_id)
        if entry is None:
            return self._json({"error": f"msg_id {msg_id} not found"}), 0.0, {}
        role = entry.get("role")
        if role == "user":
            return self._json({"error": "Deleting user messages is not supported"}), 0.0, {}
        if role not in ("assistant", "tool"):
            return self._json({"error": f"Unsupported role '{role}' for deletion"}), 0.0, {}
        agent_data.deleted_msg_ids.add(msg_id)
        result = {"status": "success", "deleted_msg_id": msg_id, "deleted_role": role}
        return self._json(result), 0.0, {"deleted_msg_ids": [msg_id], "editor_msg_ids": [msg_id]}


class TruncateContextTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        try:
            msg_id = int(parameters.get("msg_id"))
        except (ValueError, TypeError):
            return self._json({"error": "msg_id is required and must be an integer"}), 0.0, {}
        start_sentence = parameters.get("start_sentence")
        stop_sentence = parameters.get("stop_sentence")
        if start_sentence is None or stop_sentence is None:
            return self._json({"error": "msg_id, start_sentence, and stop_sentence are all required."}), 0.0, {}
        _, entry = _resolve_msg_entry(agent_data.full_history, msg_id)
        if entry is None:
            return self._json({"error": f"msg_id {msg_id} not found"}), 0.0, {}
        if entry.get("role") not in ("tool", "assistant"):
            return self._json({"error": f"truncateContext does not support role '{entry.get('role')}'."}), 0.0, {}
        text = _extract_context_text(entry)
        start_idx = text.find(str(start_sentence))
        if start_idx == -1:
            return self._json({"error": f"start_sentence not found in message {msg_id}."}), 0.0, {}
        stop_idx = text.find(str(stop_sentence), start_idx + len(str(start_sentence)))
        if stop_idx == -1:
            return self._json({"error": f"stop_sentence not found after start_sentence in message {msg_id}."}), 0.0, {}
        truncated = text[start_idx : stop_idx + len(str(stop_sentence))]
        replacement = "[Original long context has been truncated to the following message to save space.]\nTruncated Content: \n" + truncated
        if not hasattr(agent_data, "truncated_msg_ids"):
            agent_data.truncated_msg_ids = {}
        agent_data.truncated_msg_ids[msg_id] = replacement
        result = {"status": "success", "msg_id": msg_id, "truncated_length": len(replacement)}
        return self._json(result), 0.0, {"editor_msg_ids": [msg_id]}


class SummarizeContextTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        try:
            msg_id = int(parameters.get("msg_id"))
        except (ValueError, TypeError):
            return self._json({"error": "msg_id is required and must be an integer"}), 0.0, {}
        summary = parameters.get("summary")
        if summary is None:
            return self._json({"error": "msg_id and summary are both required."}), 0.0, {}
        _, entry = _resolve_msg_entry(agent_data.full_history, msg_id)
        if entry is None:
            return self._json({"error": f"msg_id {msg_id} not found"}), 0.0, {}
        if entry.get("role") not in ("tool", "assistant"):
            return self._json({"error": f"summarizeContext does not support role '{entry.get('role')}'."}), 0.0, {}
        replacement = "[Original long context has been summarized to the following message to save space.]\nSummarized Content: \n" + str(summary)
        if not hasattr(agent_data, "summarized_msg_ids"):
            agent_data.summarized_msg_ids = {}
        agent_data.summarized_msg_ids[msg_id] = replacement
        if hasattr(agent_data, "compressed_msg_ids"):
            agent_data.compressed_msg_ids.discard(msg_id)
        result = {"status": "success", "msg_id": msg_id, "summary_length": len(replacement)}
        return self._json(result), 0.0, {"editor_msg_ids": [msg_id]}


class CompressContextTool(_StateLMTool):
    @staticmethod
    def _normalize_compression_rate(rate) -> float:
        if rate is None:
            return 10.0
        if isinstance(rate, str):
            rate = rate.strip().rstrip("%")
        try:
            value = float(rate)
        except (TypeError, ValueError):
            return 10.0
        return max(0.0, min(100.0, value))

    @staticmethod
    def _fallback_compress_text(text: str, compression_rate: float) -> str:
        words = str(text).split()
        if not words or compression_rate <= 0:
            return ""
        keep_count = max(1, int(len(words) * compression_rate / 100.0))
        keep_count = min(len(words), keep_count)
        sampled_indices = sorted(random.sample(range(len(words)), keep_count))
        return " ".join(words[i] for i in sampled_indices)

    @staticmethod
    def _compress_context_with_llmlingua2(text: str, compression_rate: float) -> str:
        if compression_rate <= 0:
            return ""
        config_path = os.getenv("CONTEXTPILOT_LLMLINGUA2_CONFIG")
        if not config_path:
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "infer", "openai_endpoint_llmlingua2.json")
            )
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        from openai import OpenAI

        base_url = config.get("OPENAI_BASE_URL")
        api_key = config.get("OPENAI_API_KEY") or "EMPTY"
        model_id = config.get("MODEL_ID", "LLMLingua-2")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120) if base_url else OpenAI(api_key=api_key, timeout=120)
        keep_words = max(1, int(len(str(text).split()) * compression_rate / 100.0))
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Compress the user's text. Return only the compressed text, with no explanation.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Keep about {compression_rate:.2f}% of the original text "
                        f"(approximately {keep_words} whitespace-delimited words).\n\n"
                        f"Text:\n{text}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=max(1, min(4096, keep_words * 3 + 64)),
        )
        return (response.choices[0].message.content or "").strip()

    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if not agent_data:
            return self._json({"error": "Agent data not available"}), 0.0, {}
        try:
            msg_id = int(parameters.get("msg_id"))
        except (ValueError, TypeError):
            return self._json({"error": "msg_id is required and must be an integer"}), 0.0, {}
        _, entry = _resolve_msg_entry(agent_data.full_history, msg_id)
        if entry is None:
            return self._json({"error": f"msg_id {msg_id} not found"}), 0.0, {}
        if entry.get("role") not in ("tool", "assistant"):
            return self._json({"error": f"compressContext does not support role '{entry.get('role')}'."}), 0.0, {}
        compression_rate = self._normalize_compression_rate(parameters.get("compression_rate", 10))
        fallback_used = False
        fallback_reason = None
        try:
            compressed_body = self._compress_context_with_llmlingua2(
                _extract_context_text(entry),
                compression_rate,
            )
        except Exception as exc:
            fallback_used = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            compressed_body = self._fallback_compress_text(_extract_context_text(entry), compression_rate)
        replacement = (
            "[Original long context has been compressed to the following message to save space.]\n"
            f"Compression Rate: {compression_rate:.2f}%\n"
            "Compressed Content: \n"
            + compressed_body
        )
        if not hasattr(agent_data, "summarized_msg_ids"):
            agent_data.summarized_msg_ids = {}
        if not hasattr(agent_data, "compressed_msg_ids"):
            agent_data.compressed_msg_ids = set()
        agent_data.summarized_msg_ids[msg_id] = replacement
        agent_data.compressed_msg_ids.add(msg_id)
        result = {
            "status": "success",
            "msg_id": msg_id,
            "compression_rate": compression_rate,
            "compressed_length": len(replacement),
            "fallback_used": fallback_used,
        }
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        return self._json(result), 0.0, {"editor_msg_ids": [msg_id]}


class FinishTool(_StateLMTool):
    @rollout_trace_op
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "No final answer provided.")
        return self._json({"final_answer": answer}), 0.0, {}
