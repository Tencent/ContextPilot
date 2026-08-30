import copy
import json
import os
import re
from datetime import datetime
from typing import Any, Optional

import numpy as np


class ContextPilotMemoryState:
    _REL_W_ENTITY = 3
    _REL_W_RELATION = 3
    _REL_W_TIME = 2
    _REL_W_FACET = 1
    _REL_VEC_THRESHOLD = 0.55
    _REL_VEC_MAX_BONUS = 5
    _TIME_TOKEN_RE = re.compile(r"\b\d{3,4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?\b")

    def __init__(self, notes: dict[str, dict[str, Any]]):
        self.notes = notes
        self._mem_signatures: dict[str, dict[str, Any]] = {}
        self._mem_embeddings: dict[str, np.ndarray] = {}
        self._mem_edges: dict[str, dict[str, dict[str, Any]]] = {}
        self._embedding_disabled = False
        self._vec_threshold = self._REL_VEC_THRESHOLD
        self._vec_max_bonus = self._REL_VEC_MAX_BONUS
        self._emb_client = None
        self._emb_model: Optional[str] = None
        for key in list(self.notes):
            self._reindex_memory(key, recompute_embedding=True)

    @staticmethod
    def _safe_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _safe_list(value: Any) -> list:
        return value if isinstance(value, list) else []

    @staticmethod
    def _dedupe_dicts(items: list[Any]) -> list[dict[str, Any]]:
        deduped = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            except TypeError:
                key = str(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _normalize_entities(self, entities: Any) -> str:
        if isinstance(entities, str):
            return entities
        normalized = []
        for entity in self._safe_list(entities):
            if not isinstance(entity, dict):
                continue
            name = self._safe_str(entity.get("name")).strip()
            if not name:
                continue
            normalized.append(
                {
                    "name": name,
                    "type": self._safe_str(entity.get("type"), "Thing") or "Thing",
                    "description": self._safe_str(entity.get("description")),
                    "aliases": [
                        self._safe_str(alias).strip()
                        for alias in self._safe_list(entity.get("aliases"))
                        if self._safe_str(alias).strip()
                    ],
                    "evidence": self._safe_str(entity.get("evidence")),
                }
            )
        if normalized:
            return json.dumps(self._dedupe_dicts(normalized), ensure_ascii=False)
        return self._safe_str(entities)

    def _normalize_facets(self, facets: Any) -> list[dict[str, str]]:
        normalized = []
        for facet in self._safe_list(facets):
            if not isinstance(facet, dict):
                continue
            label = self._safe_str(facet.get("label")).strip()
            description = self._safe_str(facet.get("description"))
            if label or description:
                normalized.append({"label": label, "description": description})
        return self._dedupe_dicts(normalized)

    def _normalize_episodes(self, episodes: Any) -> str:
        if isinstance(episodes, str):
            return episodes
        normalized = []
        for episode in self._safe_list(episodes):
            if not isinstance(episode, dict):
                continue
            title = self._safe_str(episode.get("title")).strip()
            summary = self._safe_str(episode.get("summary"))
            if not title and not summary:
                continue
            normalized.append(
                {
                    "title": title or summary[:80],
                    "summary": summary,
                    "facets": self._normalize_facets(episode.get("facets")),
                    "entities": [
                        self._safe_str(entity).strip()
                        for entity in self._safe_list(episode.get("entities"))
                        if self._safe_str(entity).strip()
                    ],
                    "timestamp": self._safe_str(episode.get("timestamp")),
                    "normalized_time": self._safe_str(episode.get("normalized_time")),
                    "location": self._safe_str(episode.get("location")),
                    "participants": [
                        self._safe_str(participant).strip()
                        for participant in self._safe_list(episode.get("participants"))
                        if self._safe_str(participant).strip()
                    ],
                    "event_type": self._safe_str(episode.get("event_type")),
                    "chunk_ids": episode.get("chunk_ids") if isinstance(episode.get("chunk_ids"), list) else [],
                }
            )
        if normalized:
            return json.dumps(self._dedupe_dicts(normalized), ensure_ascii=False)
        return self._safe_str(episodes)

    def _normalize_relations(self, relations: Any) -> list[dict[str, str]]:
        normalized = []
        for relation in self._safe_list(relations):
            if not isinstance(relation, dict):
                continue
            source = self._safe_str(relation.get("source")).strip()
            target = self._safe_str(relation.get("target")).strip()
            relation_name = self._safe_str(relation.get("relation")).strip()
            if not source or not target or not relation_name:
                continue
            normalized.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation_name,
                    "description": self._safe_str(relation.get("description")),
                    "evidence": self._safe_str(relation.get("evidence")),
                }
            )
        return self._dedupe_dicts(normalized)

    @staticmethod
    def _normalize_source(source: Any) -> dict[str, list]:
        if not isinstance(source, dict):
            return {"chunk_ids": [], "msg_ids": []}
        return {
            "chunk_ids": source.get("chunk_ids") if isinstance(source.get("chunk_ids"), list) else [],
            "msg_ids": source.get("msg_ids") if isinstance(source.get("msg_ids"), list) else [],
        }

    def _build_memory(
        self,
        key: str,
        content: Any,
        summary: Any,
        entities: Any = None,
        episodes: Any = None,
        relations: Any = None,
        source: Any = None,
    ) -> dict[str, Any]:
        timestamp = datetime.utcnow().isoformat() + "Z"
        return {
            "key": str(key),
            "summary": self._safe_str(summary),
            "full_content": self._safe_str(content),
            "content": self._safe_str(content),
            "entities": self._normalize_entities(entities),
            "episodes": self._normalize_episodes(episodes),
            "relations": self._normalize_relations(relations),
            "source": self._normalize_source(source),
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    @staticmethod
    def _norm_term(value: Any) -> str:
        text = ContextPilotMemoryState._safe_str(value).strip()
        return text.casefold() if text else ""

    @staticmethod
    def _parse_json_field(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        if not isinstance(value, str) or not value.strip():
            return []
        text = value.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass
        results = []
        for line in text.splitlines():
            try:
                parsed = json.loads(line.strip())
            except Exception:
                continue
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                results.append(parsed)
        return results

    def _entity_terms(self, memory: dict[str, Any]) -> set[str]:
        terms = set()
        for entity in self._parse_json_field(memory.get("entities", "")):
            if not isinstance(entity, dict):
                term = self._norm_term(entity)
                if term:
                    terms.add(term)
                continue
            for field in ("name", "canonical_name"):
                term = self._norm_term(entity.get(field))
                if term:
                    terms.add(term)
            for alias in entity.get("aliases") or []:
                term = self._norm_term(alias)
                if term:
                    terms.add(term)
        for episode in self._parse_json_field(memory.get("episodes", "")):
            if not isinstance(episode, dict):
                continue
            for entity in episode.get("entities") or []:
                term = self._norm_term(entity)
                if term:
                    terms.add(term)
            for participant in episode.get("participants") or []:
                term = self._norm_term(participant)
                if term:
                    terms.add(term)
            location = self._norm_term(episode.get("location"))
            if location:
                terms.add(location)
        for relation in memory.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            terms.add(self._norm_term(relation.get("source")))
            terms.add(self._norm_term(relation.get("target")))
        return {term for term in terms if term and len(term) >= 2}

    def _facet_terms(self, memory: dict[str, Any]) -> set[str]:
        terms = set()
        for episode in self._parse_json_field(memory.get("episodes", "")):
            if not isinstance(episode, dict):
                continue
            for facet in episode.get("facets") or []:
                term = self._norm_term(facet.get("label") if isinstance(facet, dict) else facet)
                if term:
                    terms.add(term)
            event_type = self._norm_term(episode.get("event_type"))
            if event_type:
                terms.add(event_type)
        return {term for term in terms if term and len(term) >= 2}

    def _time_terms(self, memory: dict[str, Any]) -> set[str]:
        terms = set()
        for episode in self._parse_json_field(memory.get("episodes", "")):
            if not isinstance(episode, dict):
                continue
            for field in ("timestamp", "normalized_time"):
                raw = self._safe_str(episode.get(field)).strip()
                if not raw:
                    continue
                terms.add(raw.casefold())
                terms.update(token.casefold() for token in self._TIME_TOKEN_RE.findall(raw))
        return {term for term in terms if term and len(term) >= 2}

    def _signature_text(self, memory: dict[str, Any]) -> str:
        parts = [
            self._safe_str(memory.get("summary")),
            self._safe_str(memory.get("entities")),
            self._safe_str(memory.get("episodes")),
            self._safe_str(memory.get("full_content"))[:1500],
        ]
        return "\n".join(part for part in parts if part)

    def _compute_signature(self, memory: dict[str, Any]) -> dict[str, Any]:
        relations = []
        for relation in memory.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            source = self._safe_str(relation.get("source")).casefold().strip()
            target = self._safe_str(relation.get("target")).casefold().strip()
            if source and target:
                relations.append({"src": source, "tgt": target, "raw": relation})
        return {
            "entities": self._entity_terms(memory),
            "facets": self._facet_terms(memory),
            "times": self._time_terms(memory),
            "relations": relations,
            "text": self._signature_text(memory),
        }

    def _get_embedding_client(self):
        if self._emb_client is None:
            api_key = os.getenv("EMB_OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("EMB_OPENAI_API_KEY is not configured")
            from openai import OpenAI

            base_url = os.getenv("EMB_OPENAI_BASE_URL")
            self._emb_model = os.getenv("EMB_MODEL_ID", "text-embedding-3-small")
            self._emb_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        return self._emb_client

    def _embed_memory_text(self, key: str, signature: dict[str, Any]) -> Optional[np.ndarray]:
        if not os.getenv("EMB_OPENAI_API_KEY") or self._embedding_disabled:
            return None
        text = signature.get("text") or ""
        if not text.strip():
            return None
        try:
            response = self._get_embedding_client().embeddings.create(model=self._emb_model, input=[text])
            vector = np.asarray(response.data[0].embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm < 1e-8:
                return None
            vector = vector / norm
            self._mem_embeddings[key] = vector
            return vector
        except Exception:
            self._embedding_disabled = True
            return None

    def _pair_score(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        left_vector: Optional[np.ndarray],
        right_vector: Optional[np.ndarray],
    ) -> Optional[dict[str, Any]]:
        shared_entities = sorted(left["entities"] & right["entities"])
        shared_facets = sorted(left["facets"] & right["facets"])
        shared_timestamps = sorted(left["times"] & right["times"])
        connections = []
        seen = set()
        for relations, host_entities, peer_entities in (
            (left["relations"], left["entities"], right["entities"]),
            (right["relations"], right["entities"], left["entities"]),
        ):
            for relation in relations:
                if not (
                    (relation["src"] in host_entities and relation["tgt"] in peer_entities)
                    or (relation["tgt"] in host_entities and relation["src"] in peer_entities)
                ):
                    continue
                raw = relation["raw"]
                try:
                    relation_key = json.dumps(raw, sort_keys=True, ensure_ascii=False)
                except TypeError:
                    relation_key = str(raw)
                if relation_key in seen:
                    continue
                seen.add(relation_key)
                connections.append(copy.deepcopy(raw))
        score = (
            len(shared_entities) * self._REL_W_ENTITY
            + len(connections) * self._REL_W_RELATION
            + len(shared_timestamps) * self._REL_W_TIME
            + len(shared_facets) * self._REL_W_FACET
        )
        vector_similarity = None
        if left_vector is not None and right_vector is not None:
            vector_similarity = max(-1.0, min(1.0, float(np.dot(left_vector, right_vector))))
            if vector_similarity >= self._vec_threshold:
                span = max(1e-6, 1.0 - self._vec_threshold)
                bonus = max(1, int(round((vector_similarity - self._vec_threshold) / span * self._vec_max_bonus)))
                score += bonus
        if score <= 0:
            return None
        return {
            "score": int(score),
            "breakdown": {
                "shared_entities": shared_entities,
                "shared_timestamps": shared_timestamps,
                "shared_facets": shared_facets,
                "connecting_relations": connections,
            },
            "vector_sim": vector_similarity,
        }

    def _remove_from_graph(self, key: str) -> None:
        key = str(key)
        self._mem_signatures.pop(key, None)
        self._mem_embeddings.pop(key, None)
        neighbours = self._mem_edges.pop(key, {})
        for other_key in list(neighbours):
            other_edges = self._mem_edges.get(other_key)
            if other_edges is not None:
                other_edges.pop(key, None)
                if not other_edges:
                    self._mem_edges.pop(other_key, None)

    def _reindex_memory(self, key: str, *, recompute_embedding: bool = True) -> None:
        key = str(key)
        memory = self.notes.get(key)
        if memory is None:
            self._remove_from_graph(key)
            return
        signature = self._compute_signature(memory)
        self._mem_signatures[key] = signature
        if recompute_embedding:
            self._mem_embeddings.pop(key, None)
            self._embed_memory_text(key, signature)
        old_neighbours = self._mem_edges.pop(key, {})
        for other_key in old_neighbours:
            other_edges = self._mem_edges.get(other_key)
            if other_edges is not None:
                other_edges.pop(key, None)
        left_vector = self._mem_embeddings.get(key)
        for other_key in self.notes:
            if other_key == key:
                continue
            other_signature = self._mem_signatures.get(other_key)
            if other_signature is None:
                other_signature = self._compute_signature(self.notes[other_key])
                self._mem_signatures[other_key] = other_signature
            edge = self._pair_score(
                signature,
                other_signature,
                left_vector,
                self._mem_embeddings.get(other_key),
            )
            if edge is not None:
                self._mem_edges.setdefault(key, {})[other_key] = edge
                self._mem_edges.setdefault(other_key, {})[key] = edge

    def _related_memories(self, key: str, max_related: int = 5) -> list[dict[str, Any]]:
        related = []
        for other_key, edge in (self._mem_edges.get(str(key)) or {}).items():
            other = self.notes.get(other_key)
            if other is None:
                continue
            related.append(
                {
                    "key": other_key,
                    "summary": other.get("summary", ""),
                    "episodes": copy.deepcopy(other.get("episodes", [])),
                    "relation_reason": copy.deepcopy(edge.get("breakdown", {})),
                    "score": int(edge.get("score", 0)),
                }
            )
        related.sort(key=lambda item: item["score"], reverse=True)
        return related[: max(0, int(max_related or 0))]

    def add_memory(
        self,
        key: str,
        content: Any,
        summary: Any,
        entities: Any = None,
        episodes: Any = None,
        relations: Any = None,
        source: Any = None,
    ) -> dict[str, Any]:
        key = str(key)
        self.notes[key] = self._build_memory(key, content, summary, entities, episodes, relations, source)
        self._reindex_memory(key, recompute_embedding=True)
        return {"status": "success", "key": key, "structured": True}

    def read_memory(self, key: str, max_related: int = 5) -> dict[str, Any]:
        key = str(key)
        memory = self.notes.get(key)
        if memory is None:
            return {"error": f"Memory '{key}' not found!"}
        return {
            "memory": copy.deepcopy(memory),
            "directly_related_memories": self._related_memories(key, max_related=max_related),
        }

    def update_memory(
        self,
        key: str,
        mode: str,
        new_content: Any = None,
        new_summary: Any = None,
        new_entities: Any = None,
        new_episodes: Any = None,
        new_relations: Any = None,
        new_source: Any = None,
    ) -> dict[str, Any]:
        key = str(key)
        if key not in self.notes:
            return {"error": f"Memory '{key}' not found!"}
        if mode == "delete":
            del self.notes[key]
            self._remove_from_graph(key)
            return {"status": "success", "key": key, "message": f"Memory '{key}' deleted."}
        if mode == "overwrite":
            if new_content is None:
                return {"error": "New memory content is required for overwrite mode."}
            self.notes[key] = self._build_memory(
                key,
                new_content,
                new_summary,
                new_entities,
                new_episodes,
                new_relations,
                new_source,
            )
            self._reindex_memory(key, recompute_embedding=True)
            return {
                "status": "success",
                "key": key,
                "message": f"Memory '{key}' overwritten.",
                "structured": True,
            }
        if mode == "append":
            memory = self.notes[key]
            if new_content:
                memory["full_content"] = (
                    self._safe_str(memory.get("full_content")) + "\n" + self._safe_str(new_content)
                ).strip()
                memory["content"] = memory["full_content"]
            if new_summary is not None:
                memory["summary"] = self._safe_str(new_summary)
            normalized_entities = self._normalize_entities(new_entities)
            if normalized_entities:
                memory["entities"] = (
                    self._safe_str(memory.get("entities")) + "\n" + normalized_entities
                ).strip()
            normalized_episodes = self._normalize_episodes(new_episodes)
            if normalized_episodes:
                memory["episodes"] = (
                    self._safe_str(memory.get("episodes")) + "\n" + normalized_episodes
                ).strip()
            memory["relations"] = self._dedupe_dicts(
                list(memory.get("relations", [])) + self._normalize_relations(new_relations)
            )
            if isinstance(new_source, dict):
                source = memory.setdefault("source", {"chunk_ids": [], "msg_ids": []})
                for field in ("chunk_ids", "msg_ids"):
                    values = new_source.get(field) if isinstance(new_source.get(field), list) else []
                    source[field] = list(dict.fromkeys(list(source.get(field, [])) + values))
            memory["updated_at"] = datetime.utcnow().isoformat() + "Z"
            self._reindex_memory(key, recompute_embedding=True)
            return {
                "status": "success",
                "key": key,
                "message": f"Memory '{key}' appended.",
                "structured": True,
            }
        return {"error": f"Invalid mode '{mode}'. Use 'append', 'overwrite', or 'delete'."}
