from datetime import datetime
import ast
import json, re, os, sys, time, uuid, random
import numpy as np
import requests
from openai import OpenAI
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set


from elasticsearch import Elasticsearch, helpers
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError


def read_json(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class StateManager:




    _REL_W_ENTITY = 3
    _REL_W_RELATION = 3
    _REL_W_TIME = 2
    _REL_W_FACET = 1
    _REL_VEC_THRESHOLD = 0.55
    _REL_VEC_MAX_BONUS = 5

    def __init__(self):
        self.notes = {}
        self.simple_notes = {}


















        self._mem_signatures: Dict[str, Dict[str, Any]] = {}
        self._mem_embeddings: Dict[str, np.ndarray] = {}
        self._mem_edges: Dict[str, Dict[str, Dict[str, Any]]] = {}



        self._embed_fn = None
        self._embedding_disabled: bool = False
        self._vec_threshold: float = self._REL_VEC_THRESHOLD
        self._vec_max_bonus: int = self._REL_VEC_MAX_BONUS

        print("[INFO] StateManager Initialized")




    def set_embedding_provider(self, embed_fn, *, vec_threshold: Optional[float] = None,
                               vec_max_bonus: Optional[int] = None) -> None:
        """Register a function ``embed_fn(List[str]) -> np.ndarray`` of shape
        ``(N, dim)``.  Re-indexes any memories that were inserted before the
        provider became available.

        ``vec_threshold`` / ``vec_max_bonus`` allow the caller to override the
        cosine cutoff and the maximum integer bonus contributed by the
        embedding signal (defaults: 0.55 and 5).
        """
        self._embed_fn = embed_fn
        self._embedding_disabled = False
        if vec_threshold is not None:
            self._vec_threshold = float(vec_threshold)
        if vec_max_bonus is not None:
            self._vec_max_bonus = int(vec_max_bonus)


        for k in list(self.notes.keys()):
            if k not in self._mem_embeddings:
                self._reindex_memory(k, recompute_embedding=True)

    @staticmethod
    def _safe_str(value, default=""):
        if value is None:
            return default
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _safe_list(value):
        return value if isinstance(value, list) else []

    @staticmethod
    def _dedupe_dicts(items):
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

    def _normalize_entities(self, entities):
        if isinstance(entities, str):
            return entities
        normalized = []
        for ent in self._safe_list(entities):
            if not isinstance(ent, dict):
                continue
            name = self._safe_str(ent.get("name")).strip()
            if not name:
                continue
            normalized.append({
                "name": name,
                "type": self._safe_str(ent.get("type"), "Thing") or "Thing",
                "description": self._safe_str(ent.get("description")),
                "aliases": [self._safe_str(a).strip() for a in self._safe_list(ent.get("aliases")) if self._safe_str(a).strip()],
                "evidence": self._safe_str(ent.get("evidence")),
            })
        if normalized:
            return json.dumps(self._dedupe_dicts(normalized), ensure_ascii=False)
        return self._safe_str(entities)

    def _normalize_facets(self, facets):
        normalized = []
        for facet in self._safe_list(facets):
            if not isinstance(facet, dict):
                continue
            label = self._safe_str(facet.get("label")).strip()
            description = self._safe_str(facet.get("description"))
            if label or description:
                normalized.append({"label": label, "description": description})
        return self._dedupe_dicts(normalized)

    def _normalize_episodes(self, episodes):
        if isinstance(episodes, str):
            return episodes
        normalized = []
        for ep in self._safe_list(episodes):
            if not isinstance(ep, dict):
                continue
            title = self._safe_str(ep.get("title")).strip()
            summary = self._safe_str(ep.get("summary"))
            if not title and not summary:
                continue
            normalized.append({
                "title": title or summary[:80],
                "summary": summary,
                "facets": self._normalize_facets(ep.get("facets")),
                "entities": [self._safe_str(e).strip() for e in self._safe_list(ep.get("entities")) if self._safe_str(e).strip()],
                "timestamp": self._safe_str(ep.get("timestamp")),
                "normalized_time": self._safe_str(ep.get("normalized_time")),
                "location": self._safe_str(ep.get("location")),
                "participants": [self._safe_str(p).strip() for p in self._safe_list(ep.get("participants")) if self._safe_str(p).strip()],
                "event_type": self._safe_str(ep.get("event_type")),
                "chunk_ids": ep.get("chunk_ids") if isinstance(ep.get("chunk_ids"), list) else [],
            })
        if normalized:
            return json.dumps(self._dedupe_dicts(normalized), ensure_ascii=False)
        return self._safe_str(episodes)

    def _normalize_relations(self, relations):
        normalized = []
        for rel in self._safe_list(relations):
            if not isinstance(rel, dict):
                continue
            source = self._safe_str(rel.get("source")).strip()
            target = self._safe_str(rel.get("target")).strip()
            relation = self._safe_str(rel.get("relation")).strip()
            if not source or not target or not relation:
                continue
            normalized.append({
                "source": source,
                "target": target,
                "relation": relation,
                "description": self._safe_str(rel.get("description")),
                "evidence": self._safe_str(rel.get("evidence")),
            })
        return self._dedupe_dicts(normalized)

    def _normalize_source(self, source):
        if not isinstance(source, dict):
            return {"chunk_ids": [], "msg_ids": []}
        return {
            "chunk_ids": source.get("chunk_ids") if isinstance(source.get("chunk_ids"), list) else [],
            "msg_ids": source.get("msg_ids") if isinstance(source.get("msg_ids"), list) else [],
        }

    def _build_memory(self, key, content, summary, entities=None, episodes=None, relations=None, source=None):
        return {
            "key": str(key),
            "summary": self._safe_str(summary),
            "full_content": self._safe_str(content),
            "content": self._safe_str(content),
            "entities": self._normalize_entities(entities),
            "episodes": self._normalize_episodes(episodes),
            "relations": self._normalize_relations(relations),
            "source": self._normalize_source(source),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

    def _text_terms(self, text):
        text = self._safe_str(text)
        terms = set()
        for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", text):
            terms.add(quoted.casefold().strip())
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*){0,4}\b", text):
            terms.add(token.casefold().strip())
        for token in re.findall(r"\b\d{3,4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?\b", text):
            terms.add(token.casefold().strip())
        return {t for t in terms if len(t) >= 2}

    @staticmethod
    def _norm_term(value) -> str:
        s = StateManager._safe_str(value).strip()
        return s.casefold() if s else ""

    @staticmethod
    def _parse_json_field(value):
        """The note schema stores ``entities`` / ``episodes`` as JSON-encoded
        strings (so they round-trip through the LLM tool args).  This helper
        decodes them back into Python objects when possible; bare strings or
        already-decoded lists/dicts are returned untouched.
        """
        if value is None:
            return []
        if isinstance(value, (list, dict)):
            return value
        if not isinstance(value, str):
            return []
        s = value.strip()
        if not s:
            return []


        results = []

        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                results.append(parsed)
        return results

    def _entity_terms(self, memory):
        """Extract canonical entity tokens from a memory.

        Pulls names from the structured ``entities`` field (and aliases),
        from each episode's ``entities`` / ``participants`` / ``location``,
        and from relation endpoints.  This intentionally avoids regex over
        the raw JSON text -- the previous heuristic happily picked up JSON
        keys like ``"name"`` or ``"type"``, which polluted every pair-wise
        comparison.
        """
        terms: Set[str] = set()
        for ent in self._parse_json_field(memory.get("entities", "")):
            if not isinstance(ent, dict):

                t = self._norm_term(ent)
                if t:
                    terms.add(t)
                continue
            for k in ("name", "canonical_name"):
                t = self._norm_term(ent.get(k))
                if t:
                    terms.add(t)
            for alias in ent.get("aliases") or []:
                t = self._norm_term(alias)
                if t:
                    terms.add(t)
        for ep in self._parse_json_field(memory.get("episodes", "")):
            if not isinstance(ep, dict):
                continue
            for e in ep.get("entities") or []:
                t = self._norm_term(e)
                if t:
                    terms.add(t)
            for p in ep.get("participants") or []:
                t = self._norm_term(p)
                if t:
                    terms.add(t)
            t = self._norm_term(ep.get("location"))
            if t:
                terms.add(t)
        for rel in memory.get("relations", []) or []:
            if not isinstance(rel, dict):
                continue
            terms.add(self._norm_term(rel.get("source")))
            terms.add(self._norm_term(rel.get("target")))
        return {t for t in terms if t and len(t) >= 2}

    def _facet_terms(self, memory):
        """Pull facet labels from each episode's ``facets[].label`` (and
        ``event_type``) -- these are the deterministic semantic foci that
        we want to intersect across memories.
        """
        terms: Set[str] = set()
        for ep in self._parse_json_field(memory.get("episodes", "")):
            if not isinstance(ep, dict):
                continue
            for f in ep.get("facets") or []:
                if isinstance(f, dict):
                    t = self._norm_term(f.get("label"))
                    if t:
                        terms.add(t)
                else:
                    t = self._norm_term(f)
                    if t:
                        terms.add(t)
            t = self._norm_term(ep.get("event_type"))
            if t:
                terms.add(t)
        return {t for t in terms if t and len(t) >= 2}

    _TIME_TOKEN_RE = re.compile(r"\b\d{3,4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?\b")

    def _time_terms(self, memory):
        """Collect timestamp tokens from each episode's ``timestamp`` /
        ``normalized_time`` fields.  Year / month / day-shaped numeric
        substrings are extracted independently so that "2021" matches
        "2021-05".
        """
        terms: Set[str] = set()
        for ep in self._parse_json_field(memory.get("episodes", "")):
            if not isinstance(ep, dict):
                continue
            for k in ("timestamp", "normalized_time"):
                raw = self._safe_str(ep.get(k)).strip()
                if not raw:
                    continue
                terms.add(raw.casefold())
                for tok in self._TIME_TOKEN_RE.findall(raw):
                    terms.add(tok.casefold())
        return {t for t in terms if t and len(t) >= 2}

    def _relation_connections(self, primary, other):
        primary_entities = self._entity_terms(primary)
        other_entities = self._entity_terms(other)
        connections = []
        for rel in primary.get("relations", []) + other.get("relations", []):
            source = rel.get("source", "").casefold()
            target = rel.get("target", "").casefold()
            if (source in primary_entities and target in other_entities) or (target in primary_entities and source in other_entities):
                connections.append(deepcopy(rel))
        return self._dedupe_dicts(connections)




    def _signature_text(self, memory: Dict[str, Any]) -> str:
        """Concatenate the most informative fields of a memory into a single
        string used as input to the embedding model.  Truncated to keep
        request payloads bounded.
        """
        parts = [
            self._safe_str(memory.get("summary")),
            self._safe_str(memory.get("entities")),
            self._safe_str(memory.get("episodes")),
            self._safe_str(memory.get("full_content"))[:1500],
        ]
        return "\n".join(p for p in parts if p)

    def _compute_signature(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-compute the deterministic feature sets used by the scorer so
        that pairwise comparisons are O(min(|A|,|B|)) instead of re-parsing
        the memory each time.
        """
        ents = self._entity_terms(memory)
        facets = self._facet_terms(memory)
        times = self._time_terms(memory)


        rels: List[Dict[str, str]] = []
        for rel in memory.get("relations", []) or []:
            if not isinstance(rel, dict):
                continue
            src = self._safe_str(rel.get("source")).casefold().strip()
            tgt = self._safe_str(rel.get("target")).casefold().strip()
            if not src or not tgt:
                continue
            rels.append({
                "src": src,
                "tgt": tgt,
                "raw": rel,
            })
        return {
            "entities": ents,
            "facets": facets,
            "times": times,
            "relations": rels,
            "text": self._signature_text(memory),
        }

    def _embed_memory_text(self, key: str, sig: Dict[str, Any]) -> Optional[np.ndarray]:
        """Run the registered embedding provider on ``sig['text']`` and cache
        the L2-normalised vector.  Silently degrades to ``None`` on any
        failure (and disables the vector channel for the rest of the
        session to avoid retry storms).
        """
        if self._embed_fn is None or self._embedding_disabled:
            return None
        text = sig.get("text") or ""
        if not text.strip():
            return None
        try:
            mat = self._embed_fn([text])
            if mat is None or len(mat) == 0:
                return None
            vec = np.asarray(mat[0], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                return None
            vec = vec / norm
            self._mem_embeddings[key] = vec
            return vec
        except Exception as e:
            print(f"[WARN] Memory embedding failed for key={key!r}: {e}. "
                  f"Vector channel disabled for the rest of this session.")
            self._embedding_disabled = True
            return None

    def _pair_score(self, sig_a: Dict[str, Any], sig_b: Dict[str, Any],
                    vec_a: Optional[np.ndarray],
                    vec_b: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Compute the relation score between two memories from their
        pre-computed signatures and (optional) embeddings.  Returns ``None``
        when the pair is not related (score <= 0).
        """
        shared_entities = sorted(sig_a["entities"] & sig_b["entities"])
        shared_facets = sorted(sig_a["facets"] & sig_b["facets"])
        shared_timestamps = sorted(sig_a["times"] & sig_b["times"])




        ents_a, ents_b = sig_a["entities"], sig_b["entities"]
        connections: List[Dict[str, Any]] = []
        seen = set()
        for rels, host_set, peer_set in (
            (sig_a["relations"], ents_a, ents_b),
            (sig_b["relations"], ents_b, ents_a),
        ):
            for r in rels:
                if (r["src"] in host_set and r["tgt"] in peer_set) or \
                   (r["tgt"] in host_set and r["src"] in peer_set):
                    raw = r["raw"]
                    try:
                        sig_key = json.dumps(raw, sort_keys=True, ensure_ascii=False)
                    except TypeError:
                        sig_key = str(raw)
                    if sig_key in seen:
                        continue
                    seen.add(sig_key)
                    connections.append(deepcopy(raw))

        score = (
            len(shared_entities) * self._REL_W_ENTITY
            + len(connections) * self._REL_W_RELATION
            + len(shared_timestamps) * self._REL_W_TIME
            + len(shared_facets) * self._REL_W_FACET
        )


        vec_sim: Optional[float] = None
        if vec_a is not None and vec_b is not None:
            try:
                cos = float(np.dot(vec_a, vec_b))
            except Exception:
                cos = 0.0
            vec_sim = max(-1.0, min(1.0, cos))
            if vec_sim >= self._vec_threshold:

                span = max(1e-6, 1.0 - self._vec_threshold)
                normed = (vec_sim - self._vec_threshold) / span
                bonus = int(round(normed * self._vec_max_bonus))
                if bonus < 1:
                    bonus = 1
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
            "vector_sim": vec_sim,
        }

    def _remove_from_graph(self, key: str) -> None:
        """Remove ``key`` and all incident edges from the graph caches."""
        k = str(key)
        self._mem_signatures.pop(k, None)
        self._mem_embeddings.pop(k, None)
        neighbours = self._mem_edges.pop(k, {})
        for other in list(neighbours.keys()):
            other_adj = self._mem_edges.get(other)
            if other_adj is not None:
                other_adj.pop(k, None)
                if not other_adj:
                    self._mem_edges.pop(other, None)

    def _reindex_memory(self, key: str, *, recompute_embedding: bool = True) -> None:
        """Recompute signature / embedding for ``key`` and rebuild every
        edge incident to it.  Called whenever a memory is added,
        overwritten, or appended.
        """
        k = str(key)
        memory = self.notes.get(k)
        if memory is None:
            self._remove_from_graph(k)
            return

        sig = self._compute_signature(memory)
        self._mem_signatures[k] = sig

        if recompute_embedding:
            self._mem_embeddings.pop(k, None)
            self._embed_memory_text(k, sig)



        old_neighbours = self._mem_edges.pop(k, {})
        for other in old_neighbours.keys():
            other_adj = self._mem_edges.get(other)
            if other_adj is not None:
                other_adj.pop(k, None)

        vec_a = self._mem_embeddings.get(k)
        for other_key in self.notes.keys():
            if other_key == k:
                continue
            other_sig = self._mem_signatures.get(other_key)
            if other_sig is None:


                other_sig = self._compute_signature(self.notes[other_key])
                self._mem_signatures[other_key] = other_sig
            vec_b = self._mem_embeddings.get(other_key)
            edge = self._pair_score(sig, other_sig, vec_a, vec_b)
            if edge is None:
                continue
            self._mem_edges.setdefault(k, {})[other_key] = edge
            self._mem_edges.setdefault(other_key, {})[k] = edge

    def _related_memories(self, key, max_related=5):
        """Return the persisted neighbours of ``key`` ranked by score.

        The output schema is identical to the previous query-time scorer so
        that ``loadMemory``'s response shape is fully backwards compatible.
        """
        k = str(key)
        if k not in self.notes:
            return []
        adj = self._mem_edges.get(k) or {}
        if not adj:
            return []

        related: List[Dict[str, Any]] = []
        for other_key, edge in adj.items():
            other = self.notes.get(other_key)
            if other is None:
                continue
            related.append({
                "key": other_key,
                "summary": other.get("summary", ""),
                "episodes": deepcopy(other.get("episodes", [])),
                "relation_reason": deepcopy(edge.get("breakdown", {})),
                "score": int(edge.get("score", 0)),
            })

        related.sort(key=lambda item: item["score"], reverse=True)
        return related[:max(0, int(max_related or 0))]

    def add_note(self, key, content, summary, entities=None, episodes=None, relations=None, source=None):
        k = str(key)
        self.notes[k] = self._build_memory(key, content, summary, entities, episodes, relations, source)


        self._reindex_memory(k, recompute_embedding=True)
        return {"status": "success", "key": key, "structured": True}

    def read_note(self, key, max_related=5):
        note = self.notes.get(str(key))
        if note is None:
            return {"error": f"Memory '{key}' not found!"}
        return {
            "memory": deepcopy(note),
            "directly_related_memories": self._related_memories(key, max_related=max_related),
        }

    def update_note(self, key, mode, new_content=None, new_summary=None, new_entities=None, new_episodes=None, new_relations=None, new_source=None):
        k = str(key)
        if k not in self.notes:
            return {"error": f"Memory '{key}' not found!"}

        if mode == "delete":
            del self.notes[k]
            self._remove_from_graph(k)
            return {"status": "success", "key": key, "message": f"Memory '{key}' deleted."}

        if mode == "overwrite":
            if new_content is None:
                return {"error": "New memory content is required for overwrite mode."}
            self.notes[k] = self._build_memory(key, new_content, new_summary, new_entities, new_episodes, new_relations, new_source)
            self._reindex_memory(k, recompute_embedding=True)
            return {"status": "success", "key": key, "message": f"Memory '{key}' overwritten.", "structured": True}

        if mode == "append":
            memory = self.notes[k]
            if new_content:
                existing_content = memory.get("full_content", "")
                memory["full_content"] = (existing_content + "\n" + self._safe_str(new_content)).strip()
                memory["content"] = memory["full_content"]
            if new_summary is not None:
                memory["summary"] = self._safe_str(new_summary)
            normalized_entities = self._normalize_entities(new_entities)
            if normalized_entities:
                memory["entities"] = (self._safe_str(memory.get("entities")) + "\n" + normalized_entities).strip()
            normalized_episodes = self._normalize_episodes(new_episodes)
            if normalized_episodes:
                memory["episodes"] = (self._safe_str(memory.get("episodes")) + "\n" + normalized_episodes).strip()
            memory["relations"] = self._dedupe_dicts(memory.get("relations", []) + self._normalize_relations(new_relations))
            if isinstance(new_source, dict):
                source = memory.setdefault("source", {"chunk_ids": [], "msg_ids": []})
                for field in ("chunk_ids", "msg_ids"):
                    source[field] = list(dict.fromkeys(source.get(field, []) + (new_source.get(field) if isinstance(new_source.get(field), list) else [])))
            memory["updated_at"] = datetime.utcnow().isoformat() + "Z"



            self._reindex_memory(k, recompute_embedding=True)
            return {"status": "success", "key": key, "message": f"Memory '{key}' appended.", "structured": True}

        return {"error": f"Invalid mode '{mode}'. Use 'append', 'overwrite', or 'delete'."}

    def merge_notes(self, keys, new_key=None, new_summary=None):
        """Merge Multiple Notes into One."""
        notes_to_merge = []
        for key in keys:
            k = str(key)
            if k in self.notes:
                notes_to_merge.append((k, self.notes[k]["summary"], self.notes[k]["full_content"]))
                del self.notes[k]



                self._remove_from_graph(k)

        if notes_to_merge:
            merged_key = new_key or "_".join([note[0] for note in notes_to_merge])
            existing_summary = new_summary or "  ".join([note[1] for note in notes_to_merge])
            merged_content = "\n".join([note[2] for note in notes_to_merge])

            self.notes[merged_key] = {"summary": str(existing_summary), "full_content": str(merged_content)}



            return {"status": "success", "new_key": merged_key, "merged_from": [note[0] for note in notes_to_merge]}

        return {"error": "No notes found to merge."}

    def get_notes_summary(self):

        if not self.notes: return "No memories recorded."
        return "\n".join([f"- **{key}**: {data['summary']}" for key, data in self.notes.items()])

    def add_simple_note(self, key, content, summary):
        self.simple_notes[str(key)] = {"summary": str(summary), "full_content": str(content)}
        return {"status": "success", "key": key}

    def read_simple_note(self, key):
        note = self.simple_notes.get(str(key))
        if note is None:
            return {"error": f"Note '{key}' not found!"}
        return deepcopy(note)

    def update_simple_note(self, key, mode, new_content=None, new_summary=None):
        k = str(key)
        if k not in self.simple_notes:
            return {"error": f"Note '{key}' not found!"}

        if mode == "delete":
            del self.simple_notes[k]
            return {"status": "success", "key": key, "message": f"Note '{key}' deleted."}

        if new_content is None:
            return {"error": "New note content is required."}

        existing_content = self.simple_notes[k]["full_content"]
        if mode == "append":
            full_content = existing_content + "\n" + str(new_content)
            msg = f"Note '{key}' appended."
        elif mode == "overwrite":
            full_content = str(new_content)
            msg = f"Note '{key}' overwritten."
        else:
            return {"error": f"Invalid mode '{mode}'. Use 'append', 'overwrite', or 'delete'."}

        self.simple_notes[k]["full_content"] = full_content
        if new_summary is not None:
            self.simple_notes[k]["summary"] = str(new_summary)
        return {"status": "success", "key": key, "message": msg}

    def get_simple_notes_summary(self):
        if not self.simple_notes: return "No notes recorded."
        return "\n".join([f"- **{key}**: {data['summary']}" for key, data in self.simple_notes.items()])

class ToolLibrary:




    _DEFAULT_BOUNDARY_PATTERNS = [
        r"\n\s*\n",
        r"[.!?][\"')\]]?\s",
        r"[\u3002\uff01\uff1f]",
        r"\n",
    ]


    _ALLOWED_TRUNCATE_SIDES = ("middle", "tail")

    @classmethod
    def _normalize_truncate_side(cls, side) -> str:
        """Normalize and validate the loadDocument truncate side.

        Accepts None / empty -> "middle" (legacy default). Any unknown value
        falls back to "middle" with a warning. Returns one of
        `_ALLOWED_TRUNCATE_SIDES`.
        """
        if side is None:
            return "middle"
        s = str(side).strip().lower()
        if not s:
            return "middle"
        if s not in cls._ALLOWED_TRUNCATE_SIDES:
            print(f"[WARN] Unknown load_document_truncate_side='{side}', "
                  f"expected one of {cls._ALLOWED_TRUNCATE_SIDES}; "
                  f"falling back to 'middle'.")
            return "middle"
        return s

    def __init__(self, state_manager, tokenizer, document_content,
                 embedding_config: Optional[Dict[str, str]] = None,
                 vllm_client: Optional[OpenAI] = None,
                 vllm_model_name: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 overlap: int = 0,
                 max_context: Optional[int] = None,
                 boundary_backtrack: int = 0,
                 boundary_patterns: Optional[List[str]] = None,
                 max_read_multi_chunks: int = 3,
                 reranker_config: Optional[Dict[str, str]] = None,
                 use_reranker: bool = False,
                 load_document_truncate_side: str = "middle"):
        self.state_manager = state_manager
        self.document = document_content
        self.tokenizer = tokenizer
        self._default_chunk_size = chunk_size
        self._default_overlap = overlap
        self._max_context = max_context



        self._load_document_truncate_side = self._normalize_truncate_side(load_document_truncate_side)



        self._max_read_multi_chunks = max(1, int(max_read_multi_chunks or 1))





        self._boundary_backtrack = max(0, int(boundary_backtrack or 0))







        self._default_highlight_fragment_size: Optional[int] = None
        self._default_highlight_num_fragments: Optional[int] = None
        self._default_highlight_no_match_size: Optional[int] = None




        self._search_engine_max_results: int = 20
        raw_patterns = boundary_patterns if boundary_patterns else self._DEFAULT_BOUNDARY_PATTERNS
        self._boundary_regexes = [re.compile(p) for p in raw_patterns]
        self._vllm_client = vllm_client
        self._vllm_model = vllm_model_name

        self.chunk_pointer = [-1, 0]
        self.last_scanned_chunk_id = -1
        self.scan_mode = False
        self.index = []
        self.keywords_searched = set()
        self._es: Optional[Elasticsearch] = None
        es_index_base = os.getenv('ES_INDEX_NAME', 'lc_agent_document')
        self._es_index_exact = os.getenv('ES_INDEX_NAME_EXACT', '').lower() in {'1', 'true', 'yes'}
        self._es_index_name = (
            es_index_base if self._es_index_exact
            else f"{es_index_base}_{os.getpid()}".lower()
        )
        self._es_num_shards = max(1, int(os.getenv('ES_INDEX_SHARDS', '1')))
        self._es_num_replicas = max(0, int(os.getenv('ES_INDEX_REPLICAS', '0')))
        self._es_delete_index_on_clear = (
            os.getenv('ES_DELETE_INDEX_ON_CLEAR', 'true').lower() in {'1', 'true', 'yes'}
            and not self._es_index_exact
        )
        self._es_host: str = os.getenv('ES_HOST', 'http://localhost:9200')
        self._es_user: Optional[str] = os.getenv('ES_USER')
        self._es_pass: Optional[str] = os.getenv('ES_PASS')
        self._es_api_key: Optional[str] = os.getenv('ES_API_KEY')
        self._es_ca_cert: Optional[str] = os.getenv('ES_CA_CERT')
        self.encoded_doc = self.tokenizer(self.document, return_offsets_mapping=True, add_special_tokens=False)
        self._doc_id: Optional[str] = None


        self._embedding_config = embedding_config
        self._emb_client: Optional[OpenAI] = None
        self._emb_model: Optional[str] = None
        self._chunk_embeddings: Optional[np.ndarray] = None
        self._emb_dim: Optional[int] = None





        self._reranker_config = reranker_config
        self._use_reranker = bool(use_reranker)









        if self._has_embedding_credentials():
            try:


                self.state_manager.set_embedding_provider(
                    lambda texts: self._embed_texts(list(texts), is_query=False)
                )
            except Exception as e:
                print(f"[WARN] Failed to register embedding provider with "
                      f"StateManager (memory-relation graph will fall back "
                      f"to deterministic signals only): {e}")

        print("[INFO] ToolLibrary Initialized")

    def _has_embedding_credentials(self) -> bool:
        """True when an embedding endpoint is reachable -- either via the
        explicit ``embedding_config`` argument or via the EMB_* environment
        variables.  Used to decide whether StateManager should attempt to
        embed memories at write time.
        """
        cfg = self._embedding_config or {}
        api_key = cfg.get("OPENAI_API_KEY") or os.getenv("EMB_OPENAI_API_KEY")
        return bool(api_key)




    def _get_es(self) -> Elasticsearch:
        if self._es is None:
            kwargs = {}

            if self._es_api_key:
                kwargs['api_key'] = self._es_api_key
            elif self._es_user and self._es_pass:
                kwargs['basic_auth'] = (self._es_user, self._es_pass)

            if self._es_ca_cert:
                kwargs['ca_certs'] = self._es_ca_cert
            self._es = Elasticsearch(self._es_host, **kwargs)
        return self._es

    def _ensure_es_index(self):
        es = self._get_es()
        if es.indices.exists(index=self._es_index_name):
            return
        es.indices.create(
            index=self._es_index_name,
            settings={
                'index': {
                    'number_of_shards': self._es_num_shards,
                    'number_of_replicas': self._es_num_replicas,
                    'analysis': {
                        'analyzer': {
                            'default': {'type': 'standard'}
                        }
                    }
                }
            },
            mappings={
                'properties': {
                    'doc_id':   {'type': 'keyword'},
                    'chunk_id': {'type': 'integer'},
                    'content':  {'type': 'text'},
                    'start_pos': {'type': 'integer'},
                    'end_pos':   {'type': 'integer'}
                }
            }
        )
        print(f"[INFO] Created Elasticsearch index '{self._es_index_name}'.")

    def _bulk_index_chunks(self):
        es = self._get_es()
        actions = ({
            '_op_type': 'index',
            '_index': self._es_index_name,
            '_id': f"{self._doc_id}:{c['chunk_id']}",
            'doc_id': self._doc_id,
            'chunk_id': c['chunk_id'],
            'content': c['content'],
            'start_pos': c['start_pos'],
            'end_pos': c['end_pos'],
        } for c in self.index)
        helpers.bulk(es, actions)
        es.indices.refresh(index=self._es_index_name)
        print(f"[INFO] Indexed {len(self.index)} chunks into Elasticsearch index '{self._es_index_name}'.")

    def clearCurrentDocument(self):
        if not self._doc_id:
            return {"message": "No active document to clear."}
        es = self._get_es()
        try:
            if self._es_delete_index_on_clear:
                es.indices.delete(index=self._es_index_name, ignore=[404])
            elif es.indices.exists(index=self._es_index_name):
                es.delete_by_query(
                    index=self._es_index_name,
                    query={"term": {"doc_id": self._doc_id}},
                    refresh=True,
                )
        except Exception as exc:
            if "index_not_found_exception" not in str(exc):
                raise

        self.index = []
        self.keywords_searched = set()
        self._doc_id = None
        self.chunk_pointer = [-1, 0]

        self._chunk_embeddings = None
        self._emb_dim = None
        return {"message": "Cleared current document from ES and local state."}




    def _get_emb_client(self) -> OpenAI:
        """Lazy-init an OpenAI client for the embedding API."""
        if self._emb_client is None:
            cfg = self._embedding_config or {}
            api_key = cfg.get("OPENAI_API_KEY") or os.getenv("EMB_OPENAI_API_KEY")
            base_url = cfg.get("OPENAI_BASE_URL") or os.getenv("EMB_OPENAI_BASE_URL")
            model_id = cfg.get("MODEL_ID") or os.getenv("EMB_MODEL_ID", "text-embedding-3-small")
            if not api_key:
                raise RuntimeError(
                    "Embedding API key is required. Provide embedding_config or set EMB_OPENAI_API_KEY."
                )
            self._emb_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            self._emb_model = model_id
        return self._emb_client

    _EMB_QUERY_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def _embed_texts(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        """Call the embedding API in batches and return an (N, dim) numpy array.

        Args:
            texts: list of strings to embed.
            is_query: if True, prepend an instruction prefix to each text so
                that instruction-aware embedding models (e.g. Qwen3-Embedding)
                produce query-optimised vectors.  Document texts should keep
                the default ``is_query=False``.
        """
        client = self._get_emb_client()
        all_vecs: List[List[float]] = []
        batch_size = 32

        if is_query:
            texts = [
                f"Instruct: {self._EMB_QUERY_INSTRUCTION}\nQuery:{t}"
                for t in texts
            ]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=self._emb_model, input=batch)
            for item in resp.data:
                all_vecs.append(item.embedding)
        arr = np.array(all_vecs, dtype=np.float32)
        return arr

    def buildEmbedding(self, params):
        """Build vector embeddings for all chunks in the index."""
        if not self.index:
            return {"error": "Index not built. Please call buildIndex first."}
        if self._chunk_embeddings is not None:
            return {
                "status": "already_built",
                "total_chunks_embedded": self._chunk_embeddings.shape[0],
                "embedding_dim": self._emb_dim,
                "model": self._emb_model,
            }
        try:
            texts = [chunk["content"] for chunk in self.index]
            print(f"[INFO] Building embeddings for {len(texts)} chunks ...")
            self._chunk_embeddings = self._embed_texts(texts)
            self._emb_dim = self._chunk_embeddings.shape[1]
            print(f"[INFO] Embeddings built: shape={self._chunk_embeddings.shape}, model={self._emb_model}")
            return {
                "status": "success",
                "total_chunks_embedded": len(texts),
                "embedding_dim": self._emb_dim,
                "model": self._emb_model,
            }
        except Exception as e:
            return {"error": f"Failed to build embeddings: {e}"}

    def semanticSearch(self, params):
        """Semantic (vector) search over chunk embeddings."""
        query = params.get("query", "").strip()
        top_k = int(params.get("top_k", 5))
        if not query:
            return {"error": "query is required."}
        if self._chunk_embeddings is None:
            return {"error": "Embeddings not built. Please call buildEmbedding first."}
        if top_k <= 0:
            return {"error": "top_k must be > 0."}

        try:

            q_vec = self._embed_texts([query], is_query=True)


            norms_chunks = np.linalg.norm(self._chunk_embeddings, axis=1, keepdims=True)
            norms_chunks = np.where(norms_chunks == 0, 1e-10, norms_chunks)
            normed_chunks = self._chunk_embeddings / norms_chunks

            norms_q = np.linalg.norm(q_vec, axis=1, keepdims=True)
            norms_q = np.where(norms_q == 0, 1e-10, norms_q)
            normed_q = q_vec / norms_q

            scores = (normed_chunks @ normed_q.T).squeeze()

            k = min(top_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:k]

            items = []
            for idx in top_indices:
                chunk = self.index[int(idx)]
                content = chunk["content"]

                if len(content) <= 600:
                    preview = content
                else:
                    preview = content[:300] + "\n...\n" + content[-300:]
                items.append({
                    "chunk_id": chunk["chunk_id"],
                    "relevance_score": round(float(scores[idx]), 4),
                    "content_preview": preview,
                })

            if not items or items[0]["relevance_score"] < 0.01:
                return {"retrieved_chunks": [], "message": "No semantically relevant chunks found.", "query": query}

            return {"retrieved_chunks": items, "query": query}
        except Exception as e:
            return {"error": f"Semantic search failed: {e}"}




    def _rerank(self, query: str, documents: List[str], top_k: int) -> List:
        """Call the cross-encoder reranker endpoint and return a list of
        ``(local_index, score)`` pairs sorted by score DESC, truncated to
        ``top_k``.

        ``local_index`` refers to the position inside the provided
        ``documents`` list (i.e. the BM25 candidate order).

        Uses the official Qwen3-Reranker prompt template (special tokens
        preserved verbatim) and posts to ``{OPENAI_BASE_URL}/rerank``.
        """
        if not self._reranker_config:
            raise RuntimeError("reranker_config is not set.")
        if not documents:
            return []

        base_url = (self._reranker_config.get("OPENAI_BASE_URL") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("Reranker OPENAI_BASE_URL is missing.")
        url = f"{base_url}/rerank"
        api_key = self._reranker_config.get("OPENAI_API_KEY") or "EMPTY"


        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        document_template = "<Document>: {doc}{suffix}"
        instruction = "Given a web search query, retrieve relevant passages that answer the query"

        formatted_query = query_template.format(
            prefix=prefix, instruction=instruction, query=query
        )
        formatted_docs = [
            document_template.format(doc=doc, suffix=suffix) for doc in documents
        ]

        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(
            url,
            json={"query": formatted_query, "documents": formatted_docs},
            headers=headers,
            timeout=300,
        )
        resp.raise_for_status()
        payload = resp.json()

        results = payload.get("results")
        if results is None:
            raise RuntimeError(f"Unexpected reranker response: {payload}")

        scored = []
        for r in results:
            idx = r.get("index")
            if idx is None:
                continue
            score = r.get("relevance_score", r.get("score", 0.0))
            scored.append((int(idx), float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None and top_k > 0:
            scored = scored[:top_k]
        return scored

    def _head_tail_snippet(self, text: str, head: int = 250, tail: int = 250) -> str:
        """Return "<first `head` tokens>\n....\n<last `tail` tokens>" of ``text``.

        Uses ``self.tokenizer`` for encode/decode so the token counting is
        consistent with the rest of the pipeline. If the text is short enough
        to fit entirely within ``head + tail`` tokens, the original text is
        returned unchanged. Used as a pseudo-highlight when hybridSearch
        bypasses Stage 1 and has no real BM25 highlight fragments available.
        """
        if not text:
            return ""
        try:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return text
        if len(token_ids) <= head + tail:
            return text
        head_text = self.tokenizer.decode(token_ids[:head], skip_special_tokens=True)
        tail_text = self.tokenizer.decode(token_ids[-tail:], skip_special_tokens=True)
        return f"{head_text}\n....\n{tail_text}"

    def hybridSearch(self, params):
        """Hybrid search: BM25 recall (top topk_bm25) then semantic rerank (top top_k).

        params:
        - query: str (required)
            Natural-language query used for semantic (vector) reranking.
            If ``keyword`` is not provided, it is split by whitespace to form
            the BM25 keyword list as a fallback.
        - keyword: str | list[str] (optional)
            Keyword(s) for the BM25 recall stage. If omitted, derived from
            ``query``.
        - topk_bm25: int (default 30)
            Number of candidates recalled by BM25 before reranking.
        - top_k: int (default 10)
            Final number of chunks returned after semantic reranking.
        - mode: "and" | "or" (default "or")
            BM25 matching mode.
        """
        query = str(params.get("query", "")).strip()
        if not query:
            return {"error": "query is required."}
        if not self.index:
            return {"error": "Index not built. Please call buildIndex first."}
        if not self._doc_id:
            return {"error": "No active document for this run. Please call buildIndex first."}
        if not self._use_reranker and self._chunk_embeddings is None:
            return {"error": "Embeddings not built. Please call buildEmbedding first."}

        try:
            topk_bm25 = int(params.get("topk_bm25", 30))
            top_k = int(params.get("top_k", 10))
        except (TypeError, ValueError):
            return {"error": "topk_bm25 and top_k must be integers."}
        if topk_bm25 <= 0 or top_k <= 0:
            return {"error": "topk_bm25 and top_k must be > 0."}


        raw_kw = params.get("keyword")
        if raw_kw is None or (isinstance(raw_kw, str) and not raw_kw.strip()):

            keywords = [k.strip() for k in query.split() if k.strip()]
            if not keywords:
                keywords = [query]
        elif isinstance(raw_kw, list):
            keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
        else:
            keywords = [k.strip() for k in str(raw_kw).split(",") if k.strip()]

        if not keywords:
            return {"error": "Could not derive any BM25 keyword from inputs."}

        self.keywords_searched.update(keywords)

        mode = (params.get("mode") or "or").lower()
        if mode not in ("and", "or"):
            return {"error": f"Unsupported mode '{mode}'. Use 'and' or 'or'."}

        es = self._get_es()

        def _clause(kw: str):
            if " " in kw:
                return {"match_phrase": {"content": {"query": kw, "slop": 2}}}
            return {"match": {"content": {"query": kw, "operator": "and"}}}

        if mode == "and":
            bm25_query = {
                "bool": {
                    "must": [_clause(kw) for kw in keywords],
                    "filter": [{"term": {"doc_id": self._doc_id}}],
                }
            }
        else:
            bm25_query = {
                "bool": {
                    "should": [_clause(kw) for kw in keywords],
                    "minimum_should_match": params.get("minimum_should_match", "1"),
                    "filter": [{"term": {"doc_id": self._doc_id}}],
                }
            }




        _fs_default = self._default_highlight_fragment_size if self._default_highlight_fragment_size is not None else 180
        _nf_default = self._default_highlight_num_fragments if self._default_highlight_num_fragments is not None else 3
        _nm_default = self._default_highlight_no_match_size if self._default_highlight_no_match_size is not None else 120
        fragment_size = int(params.get("fragment_size", _fs_default))
        num_frags = int(params.get("num_fragments", _nf_default))
        no_match_size = int(params.get("no_match_size", _nm_default))

        try:
            res = es.search(
                index=self._es_index_name,
                query=bm25_query,
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
                size=topk_bm25,
                track_total_hits=False,
            )
        except Exception as e:
            return {"error": f"Elasticsearch query failed: {e}"}

        hits = res.get("hits", {}).get("hits", [])





        cand_ids: List[int] = []
        bm25_scores: Dict[int, float] = {}
        bm25_highlights: Dict[int, list] = {}
        for h in hits:
            src = h.get("_source", {}) or {}
            cid = src.get("chunk_id")
            if cid is None:
                continue
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_int < 0 or cid_int >= len(self.index):
                continue
            if cid_int in bm25_scores:
                continue
            cand_ids.append(cid_int)
            bm25_scores[cid_int] = float(h.get("_score", 0.0))
            bm25_highlights[cid_int] = h.get("highlight", {}).get("content", [])










        num_chunks = len(self.index)
        threshold = min(top_k, num_chunks)
        if len(cand_ids) < threshold:
            cand_ids = list(range(num_chunks))
            bm25_scores = {cid: 0.0 for cid in cand_ids}
            bm25_highlights = {
                cid: [self._head_tail_snippet(self.index[cid]["content"], head=250, tail=250)]
                for cid in cand_ids
            }
            stage1_bypassed = True
        else:
            stage1_bypassed = False


        try:
            if self._use_reranker:

                cand_texts = [self.index[cid]["content"] for cid in cand_ids]
                k = min(top_k, len(cand_ids))
                try:
                    ranked = self._rerank(query, cand_texts, top_k=k)
                except Exception as e:
                    return {"error": f"Hybrid rerank failed: {e}"}

                items = []
                for local_idx, score in ranked:
                    if local_idx < 0 or local_idx >= len(cand_ids):
                        continue
                    cid = cand_ids[local_idx]
                    chunk = self.index[cid]
                    items.append({
                        "chunk_id": chunk["chunk_id"],
                        "semantic_score": round(float(score), 4),
                        "bm25_score": round(bm25_scores.get(cid, 0.0), 3),
                        "highlights": bm25_highlights.get(cid, []),
                    })
            else:

                q_vec = self._embed_texts([query], is_query=True)

                cand_arr = np.asarray(cand_ids, dtype=np.int64)
                cand_embs = self._chunk_embeddings[cand_arr]


                norms_chunks = np.linalg.norm(cand_embs, axis=1, keepdims=True)
                norms_chunks = np.where(norms_chunks == 0, 1e-10, norms_chunks)
                normed_chunks = cand_embs / norms_chunks

                norms_q = np.linalg.norm(q_vec, axis=1, keepdims=True)
                norms_q = np.where(norms_q == 0, 1e-10, norms_q)
                normed_q = q_vec / norms_q

                sem_scores = (normed_chunks @ normed_q.T).squeeze(axis=1)

                k = min(top_k, sem_scores.shape[0])
                top_order = np.argsort(sem_scores)[::-1][:k]

                items = []
                for pos in top_order:
                    local_idx = int(pos)
                    cid = int(cand_arr[local_idx])
                    chunk = self.index[cid]
                    items.append({
                        "chunk_id": chunk["chunk_id"],
                        "semantic_score": round(float(sem_scores[local_idx]), 4),
                        "bm25_score": round(bm25_scores.get(cid, 0.0), 3),
                        "highlights": bm25_highlights.get(cid, []),
                    })



















            SEM_THRESHOLD = 0.01
            kept = [it for it in items if it["semantic_score"] >= SEM_THRESHOLD]
            num_filtered_out = len(items) - len(kept)

            if len(kept) < top_k:
                present_cids = {it["chunk_id"] for it in kept}

                remaining = [
                    cid for cid in cand_ids if cid not in present_cids
                ]
                remaining.sort(key=lambda c: bm25_scores.get(c, 0.0), reverse=True)
                need = top_k - len(kept)
                for cid in remaining[:need]:
                    chunk = self.index[cid]
                    kept.append({
                        "chunk_id": chunk["chunk_id"],
                        "semantic_score": 0.0,
                        "bm25_score": round(bm25_scores.get(cid, 0.0), 3),
                        "highlights": bm25_highlights.get(cid, []),
                        "backfilled_by_bm25": True,
                    })

            if not kept:




                return {
                    "retrieved_chunks": [],
                    "message": "No chunks available for this query.",
                    "query": query,
                    "keywords": keywords,
                }

            return {
                "retrieved_chunks": kept,
                "query": query,
                "keywords": keywords,
                "bm25_candidates": len(cand_ids),
                "stage1_bypassed": stage1_bypassed,
                "semantic_filtered_out": num_filtered_out,
            }
        except Exception as e:
            return {"error": f"Hybrid search rerank failed: {e}"}

    def analyzeText(self, params):
        return {
            "file_name": "attached_document.txt",
            "total_tokens": len(self.encoded_doc["input_ids"])
        }

    def buildIndex(self, params):

        chunk_size = self._default_chunk_size if self._default_chunk_size is not None else params.get("chunk_size", 4000)
        overlap = self._default_overlap if self._default_overlap else params.get("overlap", 0)


        boundary_backtrack = self._boundary_backtrack if self._boundary_backtrack else int(params.get("boundary_backtrack", 0) or 0)

        if chunk_size <= 0: return {"error": "chunk_size must be > 0"}
        if overlap   <  0: return {"error": "overlap must be >= 0"}
        if overlap >= chunk_size: return {"error": "overlap must be < chunk_size"}









        if boundary_backtrack < 0:
            boundary_backtrack = 0
        max_safe_backtrack = max(0, chunk_size - max(1, overlap + 1))
        if boundary_backtrack > max_safe_backtrack:
            print(f"[WARN] boundary_backtrack={boundary_backtrack} is too large "
                  f"for chunk_size={chunk_size}, overlap={overlap}; "
                  f"clamping to {max_safe_backtrack}.")
            boundary_backtrack = max_safe_backtrack

        input_ids = self.encoded_doc["input_ids"]
        offsets = self.encoded_doc["offset_mapping"]

        self.index = []
        self._doc_id = uuid.uuid4().hex

        start_token = 0
        chunk_id = 0
        total_tokens = len(input_ids)
        current_section_hint = None
        section_heading_re = re.compile(
            r"(?im)^\s*((?:chapter|book|part|volume)\s+"
            r"(?:[0-9]+|[ivxlcdm]+|[a-z][a-z '\-]{1,60}))\s*$"
        )
        while start_token < total_tokens:
            hard_end = min(start_token + chunk_size, total_tokens)













            end_token = hard_end
            if boundary_backtrack > 0 and hard_end < total_tokens:
                lo = max(start_token + 1, hard_end - boundary_backtrack)
                hard_char_end = offsets[hard_end - 1][1]


                for t in range(hard_end - 1, lo - 1, -1):
                    tok_char_start = offsets[t][0]
                    tok_char_end = offsets[t][1]
                    if tok_char_end <= tok_char_start:

                        continue




                    segment = self.document[tok_char_start:hard_char_end]
                    hit = False
                    for rgx in self._boundary_regexes:
                        m = rgx.search(segment)
                        if not m:
                            continue



                        if m.start() <= (tok_char_end - tok_char_start - 1):
                            hit = True
                            break
                    if hit:
                        end_token = t + 1
                        break




            chunk_offsets = offsets[start_token:end_token]
            char_start = chunk_offsets[0][0]
            char_end = chunk_offsets[-1][1]


            chunk_content = self.document[char_start:char_end]
            heading_window = self.document[max(0, char_start - 256):char_end]
            headings = section_heading_re.findall(heading_window)
            if headings:
                current_section_hint = " ".join(headings[-1].split())[:120]
            chunk_data = {
                "chunk_id": chunk_id,
                "content": chunk_content,
                "start_pos": start_token,
                "end_pos": end_token,
                "section_hint": current_section_hint,
            }
            self.index.append(chunk_data)

            chunk_id += 1



            next_start = end_token - overlap

            if next_start <= start_token:
                next_start = start_token + 1
            start_token = next_start

        self.keywords_searched = set()
        self.chunk_pointer = [-1, 0]


        try:
            self._ensure_es_index()
            self._bulk_index_chunks()
        except Exception as e:
            return {'error': f'Failed to (re)build Elasticsearch index: {e}'}

        return {
            "index_id": "document_index",
            "total_chunks": len(self.index),
            "first_chunk_id": 0,
            "last_chunk_id": len(self.index) - 1,
        }

    def loadDocument(self, params):
        """Load the full document content. If the document exceeds max_context
        tokens, truncate it according to `self._load_document_truncate_side`:

        - "middle": keep the first half and the last half, drop the middle
          (legacy default -- useful when answers tend to be at either end).
        - "tail":   keep only the first `_max_context` tokens, drop everything
          after it (useful when the beginning carries the most relevant info).
        """
        if not self.document:
            return {"error": "Document content is empty."}

        if self._max_context is not None:
            doc_tokens = self.tokenizer.encode(self.document, add_special_tokens=False)
            if len(doc_tokens) > self._max_context:
                side = self._load_document_truncate_side or "middle"
                truncated_count = len(doc_tokens) - self._max_context

                if side == "tail":
                    head_tokens = doc_tokens[: self._max_context]
                    head_text = self.tokenizer.decode(head_tokens, skip_special_tokens=True)
                    truncated_doc = (
                        head_text
                        + f"\n\n[... {truncated_count} tokens truncated from the tail ...]"
                    )
                    print(f"[loadDocument] Document truncated: {len(doc_tokens)} tokens -> "
                          f"{self._max_context} tokens (dropped {truncated_count} from tail)")
                else:
                    half = self._max_context // 2
                    head_tokens = doc_tokens[:half]
                    tail_tokens = doc_tokens[-half:]
                    head_text = self.tokenizer.decode(head_tokens, skip_special_tokens=True)
                    tail_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)
                    truncated_doc = (
                        head_text
                        + f"\n\n[... {truncated_count} tokens truncated from the middle ...]\n\n"
                        + tail_text
                    )
                    print(f"[loadDocument] Document truncated: {len(doc_tokens)} tokens -> "
                          f"{self._max_context} tokens (dropped {truncated_count} from middle)")

                return {
                    "document_content": truncated_doc,
                    "truncated": True,
                    "truncate_side": side,
                    "original_tokens": len(doc_tokens),
                    "kept_tokens": self._max_context,
                }

        return {
            "document_content": self.document
        }

    def readChunk(self, params):
        """
        Retrieve the full text of a document chunk.

        Supports two modes:
        1. Explicit mode (enable_scan=false): Retrieve by chunk_id
        2. Scan mode (enable_scan=true): Retrieve next chunk sequentially
        """

        if params.get("enable_scan") is not None:
            self.scan_mode = params.get("enable_scan")

        if not self.index:
            return {"error": "Index not built. Please call 'buildIndex' first."}

        if self.scan_mode:

            next_chunk_id = self.last_scanned_chunk_id + 1
            if next_chunk_id >= len(self.index):
                return {"error": f"No more chunks available. Last chunk ({len(self.index)-1}) already retrieved."}

            self.last_scanned_chunk_id = next_chunk_id
            return {
                "retrieved_chunk": [self.index[next_chunk_id]],
                "chunk_id": next_chunk_id,
                "reading_progress": f"{next_chunk_id + 1}/{len(self.index)}"
            }
        else:

            chunk_id = params.get("chunk_id")
            if chunk_id is None:
                return {"error": "chunk_id is required when enable_scan is false."}

            try:
                chunk_id = int(chunk_id)
            except (ValueError, TypeError):
                return {"error": "chunk_id must be an integer."}

            if chunk_id < 0 or chunk_id > (len(self.index)-1):
                return {"error": f"Chunk_id: {chunk_id} is out of range. It must be between 0 and {len(self.index)-1}."}


            self.last_scanned_chunk_id = chunk_id
            return {"retrieved_chunk": [self.index[chunk_id]], "chunk_id": chunk_id}

    def readMultiChunks(self, params):
        """
        Retrieve the full text of multiple document chunks in a single call.

        Behavior:
        - Accepts a list via `chunk_ids`.
        - Honors only the first `self._max_read_multi_chunks` entries; any
          chunk_ids after the K-th are ignored.
        - When the provided list exceeds the server-side limit, a short
          warning is returned under the `notice` field of the form:
              "At most K chunk_ids are accepted; the input exceeded this
               limit, so chunk_ids after the K-th are ignored."
        - The payload mirrors `readChunk`'s shape: chunk content is carried
          only inside `retrieved_chunks` (each item is
          {"chunk_id": N, "content": "..."}), so the text is not duplicated
          into a separate concatenated string.
        """
        if not self.index:
            return {"error": "Index not built. Please call 'buildIndex' first."}

        raw = params.get("chunk_ids")
        if raw is None:
            return {"error": "chunk_ids is required (a list of integers)."}
        if not isinstance(raw, list):
            return {"error": "chunk_ids must be a list of integers."}
        if len(raw) == 0:
            return {"error": "chunk_ids cannot be empty."}


        normalized: List[int] = []
        for cid in raw:
            try:
                normalized.append(int(cid))
            except (ValueError, TypeError):
                return {"error": f"chunk_ids must be integers, got invalid entry: {cid!r}."}

        k = self._max_read_multi_chunks
        original_len = len(normalized)
        truncated_flag = original_len > k
        honored_ids = normalized[:k]


        max_valid = len(self.index) - 1
        for cid in honored_ids:
            if cid < 0 or cid > max_valid:
                return {"error": f"Chunk_id: {cid} is out of range. It must be between 0 and {max_valid}."}


        retrieved_chunks: List[Dict[str, Any]] = []
        for cid in honored_ids:
            chunk_entry = self.index[cid]
            retrieved_chunks.append(chunk_entry)



        if honored_ids:
            self.last_scanned_chunk_id = honored_ids[-1]

        result: Dict[str, Any] = {
            "retrieved_chunks": retrieved_chunks,
            "chunk_ids": honored_ids,
            "requested_count": original_len,
            "returned_count": len(honored_ids),
            "max_chunks_once": k,
            "truncated": truncated_flag,
        }
        if truncated_flag:
            result["notice"] = (
                f"At most {k} chunk_ids are accepted; the input exceeded this "
                f"limit, so chunk_ids after the {k}-th are ignored."
            )
        return result

    def searchEngine(self, params):
        """
        params:
        - keyword: str | list[str]
            Examples:
                "blue mountain, haze"
                ["blue mountain", "haze"]
        - mode: "and" | "or" (default: "or")
            "and": all keywords must appear
            "or": any keyword may appear (default)
        - fragment_size: int (default 180)
        - num_fragments: int (default 3)
        - no_match_size: int (default 120)
        - size: int (default 50)
        - minimum_should_match: str | int (only used for "or", default "1")
        """
        raw_kw = params.get("keyword", "")
        if not self.index:
            return {"error": "Index not built. Please call buildIndex first."}
        if not self._doc_id:
            return {"error": "No active document for this run. Please call buildIndex first."}


        if isinstance(raw_kw, list):
            keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
        else:
            keywords = [k.strip() for k in str(raw_kw).split(",") if k.strip()]

        if not keywords:
            return {"error": "keyword cannot be empty."}

        self.keywords_searched.update(keywords)




        mode = (params.get("mode") or "or").lower()
        size = int(params.get("size", 50))
        _fs_default = self._default_highlight_fragment_size if self._default_highlight_fragment_size is not None else 180
        _nf_default = self._default_highlight_num_fragments if self._default_highlight_num_fragments is not None else 3
        _nm_default = self._default_highlight_no_match_size if self._default_highlight_no_match_size is not None else 120
        fragment_size = int(params.get("fragment_size", _fs_default))
        num_frags = int(params.get("num_fragments", _nf_default))
        no_match_size = int(params.get("no_match_size", _nm_default))
        min_should = params.get("minimum_should_match", "1")

        es = self._get_es()

        def _clause(kw: str):

            if " " in kw:
                return {"match_phrase": {"content": {"query": kw, "slop": 2}}}

            return {"match": {"content": {"query": kw, "operator": "and"}}}


        if mode == "and":
            query = {
                "bool": {
                    "must": [_clause(kw) for kw in keywords],
                    "filter": [{"term": {"doc_id": self._doc_id}}],
                }
            }
        elif mode == "or":
            query = {
                "bool": {
                    "should": [_clause(kw) for kw in keywords],
                    "minimum_should_match": min_should,
                    "filter": [{"term": {"doc_id": self._doc_id}}],
                }
            }

        else:
            raise NotImplementedError(f"Search mode '{mode}' not supported.")

        try:
            res = es.search(
                index=self._es_index_name,
                query=query,
                highlight={
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fields": {
                        "content": {
                            "type": "unified",
                            "fragment_size": fragment_size,
                            "number_of_fragments": num_frags,
                            "no_match_size": no_match_size
                        }
                    }
                },
                _source=["chunk_id"],
                size=size,
                track_total_hits=False
            )
        except Exception as e:
            return {"error": f"Elasticsearch query failed: {e}"}

        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            return {"retrieved_chunks": [], "message": "No matching content found.", "keywords": keywords}

        items = []
        for h in hits:
            src = h.get("_source", {})
            chunk_id = src.get("chunk_id")
            score = h.get("_score", 0.0)
            highlights = h.get("highlight", {}).get("content", [])
            items.append({
                "chunk_id": chunk_id,
                "relevance_score": round(float(score), 3),
                "highlights": highlights,
                "section_hint": (
                    self.index[chunk_id].get("section_hint")
                    if isinstance(chunk_id, int) and 0 <= chunk_id < len(self.index)
                    else None
                ),
            })

        items.sort(key=lambda x: x["relevance_score"], reverse=True)
        total = len(items)
        max_results = max(1, int(self._search_engine_max_results or 20))
        if len(items) > max_results:
            items = items[:max_results]
            return {
                "retrieved_chunks": items,
                "message": f"Showing the most relevant {max_results}/{total} chunks.",
                "keywords": keywords
            }
        return {"retrieved_chunks": items, "keywords": keywords}

    def checkBudget(self, params):
        raise NotImplementedError

    def memorize(self, params):
        return self.state_manager.add_note(
            key=params['key'],
            content=params.get('content', ''),
            summary=params.get('summary', ''),
            entities=params.get('entities'),
            episodes=params.get('episodes'),
            relations=params.get('relations'),
            source=params.get('source')
        )

    def loadMemory(self, params):
        return self.state_manager.read_note(
            key=params['key'],
            max_related=params.get('max_related', 5)
        )

    def updateMemory(self, params):
        return self.state_manager.update_note(
            key=params['key'],
            mode=params.get('mode', '').lower(),
            new_content=params.get('new_content'),
            new_summary=params.get('new_summary'),
            new_entities=params.get('new_entities'),
            new_episodes=params.get('new_episodes'),
            new_relations=params.get('new_relations'),
            new_source=params.get('new_source')
        )

    def note(self, params):
        return self.state_manager.add_simple_note(
            key=params['key'],
            content=params.get('content', ''),
            summary=params.get('summary', '')
        )

    def readNote(self, params):
        return self.state_manager.read_simple_note(key=params['key'])

    def updateNote(self, params):
        return self.state_manager.update_simple_note(
            key=params['key'],
            mode=params.get('mode', '').lower(),
            new_content=params.get('new_content'),
            new_summary=params.get('new_summary')
        )

    def mergeNotes(self, params):
        return self.state_manager.merge_notes(
            keys=params['keys'],
            new_key=params.get('new_key'),
            new_summary=params.get('new_summary')
        )

    def deleteContext(self, params):
        raise NotImplementedError

    def getContextStats(self, params):
        """Get context statistics."""
        return {
            "total_notes": len(self.state_manager.notes),
            "notes_keys": list(self.state_manager.notes.keys()),
            "index_chunks": len(self.index),
            "document_size": len(self.encoded_doc["input_ids"]),
            "searched_keywords": list(self.keywords_searched),
        }

    def finish(self, params):
        return {"final_answer": params.get("answer", "No final answer provided.")}


class ExecLogger:
    """Save query logs, trajectories, and final answers."""
    def __init__(self, log_dir="logs", results_dir="results"):
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.ensure_output_dir()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.api_calls = []

    def ensure_output_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def log_api_call(self, api_input, api_output, call_index):
        api_call = {
            "timestamp": datetime.now().isoformat(),
            "call_index": call_index,
            "session_id": self.session_id,
            "api_input": api_input,
            "api_output": api_output
        }
        self.api_calls.append(api_call)
        print(f"[INFO] API call {call_index} has been recorded")

    def save_query_log(self, query, document_info, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "query": query,
            "document_info": document_info,
            "api_calls_count": len(self.api_calls)
        }

        log_file = os.path.join(self.log_dir, f"query_log_{self.session_id}.json")

        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Query log saved to: {log_file}")

    def save_inference_result(self, query, agent, result_info=None, prefix_tag=None):
        timestamp = datetime.now().isoformat()

        sanitized_history = agent._sanitize_for_json(agent.full_history)
        sanitized_result_info = agent._sanitize_for_json(result_info or {})
        sanitized_snapshots = agent._sanitize_for_json(
            getattr(agent, "snapshots", [])
        )

        inference_trace = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "system_prompt": agent.system_prompt,
            "query": query,
            "result_info": sanitized_result_info,
            "full_history": sanitized_history,
            "snapshots": sanitized_snapshots,
        }

        result_file = os.path.join(self.log_dir, f"inference_result_{self.session_id}.json")
        if prefix_tag:
            result_file = os.path.join(self.log_dir, f"{prefix_tag}_inference_result_{self.session_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(inference_trace, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Inference results saved to: {result_file}")
        return result_file


    def save_api_calls_log(self):
        """Save API calls log separately"""
        if not self.api_calls:
            return

        api_log_file = os.path.join(self.log_dir, f"api_calls_{self.session_id}.json")
        with open(api_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.api_calls, f, ensure_ascii=False, indent=2)

        print(f"[INFO] API calls log saved to: {api_log_file}")

    def save_final_result(self, agent, question, expected_answer, meta_info=None, filename=None):
        """Save the final answer and metadata."""
        full_history = agent.full_history
        final_answer = None
        for msg in reversed(full_history):
            if msg.get("role") == "tool":
                final_answer = msg.get("content", {}).get("final_answer")
                if final_answer:
                    break
        if not final_answer:
            final_answer = "No final answer found."
        result_info = {
            "session_id": self.session_id,
            "question": question,
            "final_answer": final_answer,
            "expected_answer": expected_answer,
            "meta_info": meta_info or {}
        }
        if filename:
            result_file = os.path.join(self.results_dir, filename)
        else:
            sample_id = (meta_info or {}).get("sample_id", "id_unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.results_dir, f"{sample_id}_final_result_{timestamp}.json")

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Final results saved to: {result_file}")
        return result_file, final_answer


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks from text (for API payload only)."""
    return _THINK_RE.sub("", text).strip()


class ContextPilot:
    def __init__(self,
                 vllm_config: Dict[str, Any],
                 document_content: str,
                 temperature: float,
                 tokenizer: Any,
                 logger: Optional[ExecLogger] = None,
                 max_context_exp: int = 30720,
                 max_turns_exp: int = 50,
                 max_output_tokens: int = 4096,
                 system_prompt_name: Optional[str] = None,
                 tool_config_path: Optional[str] = None,
                 topp: float = 1.0,
                 topk: int = None,
                 delete_assistant_tool_call_only: bool = False,
                 mflow_project_dir: Optional[str] = None,
                 enable_graph: bool = False,
                 allow_text_tool_call_fallback: bool = False,
                 embedding_config: Optional[Dict[str, str]] = None,
                 chunk_size: Optional[int] = None,
                 overlap: int = 0,
                 boundary_backtrack: int = 0,
                 max_chunks_once: int = 3,
                 reranker_config: Optional[Dict[str, str]] = None,
                 use_reranker: bool = False,
                ):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._boundary_backtrack = boundary_backtrack
        self._max_chunks_once = max(1, int(max_chunks_once or 1))
        print("[INFO] Setting up OpenAI Client...")


        self.model_name = (
            vllm_config.get("MODEL_ID", "ContextPilot")
        )

        openai_base = vllm_config.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        openai_key  = vllm_config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is required. Set env or pass via vllm_config['OPENAI_API_KEY'].")




        openai_bases = vllm_config.get("OPENAI_BASE_URLS")
        worker_rank = vllm_config.get("_worker_rank", 0)
        if openai_bases and isinstance(openai_bases, list) and len(openai_bases) > 0:
            chosen = openai_bases[worker_rank % len(openai_bases)]
            print(f"[INFO] OPENAI_BASE_URLS has {len(openai_bases)} endpoints, "
                  f"worker_rank={worker_rank} -> using {chosen}")
            openai_base = chosen

        extra_headers = (
            vllm_config.get("OPENAI_EXTRA_HEADERS")
            or vllm_config.get("EXTRA_HEADERS")
            or os.getenv("OPENAI_EXTRA_HEADERS")
        )
        if isinstance(extra_headers, str):
            extra_headers = json.loads(extra_headers) if extra_headers.strip() else None
        if extra_headers is not None and not isinstance(extra_headers, dict):
            raise TypeError("OPENAI_EXTRA_HEADERS must be a JSON object/dict when provided.")

        extra_body = (
            vllm_config.get("OPENAI_EXTRA_BODY")
            or vllm_config.get("EXTRA_BODY")
            or os.getenv("OPENAI_EXTRA_BODY")
        )
        if isinstance(extra_body, str):
            extra_body = json.loads(extra_body) if extra_body.strip() else None
        if extra_body is not None and not isinstance(extra_body, dict):
            raise TypeError("OPENAI_EXTRA_BODY must be a JSON object/dict when provided.")
        self.openai_extra_body = extra_body or {}

        request_timeout = int(os.getenv("OPENAI_TIMEOUT", "600"))
        client_kwargs = {"api_key": openai_key, "timeout": request_timeout}
        if openai_base:
            client_kwargs["base_url"] = openai_base
        if extra_headers:
            client_kwargs["default_headers"] = {
                str(key): str(value) for key, value in extra_headers.items()
            }
        self.vllm_client = OpenAI(**client_kwargs)

        self.system_prompt = self._get_system_prompt_text(system_prompt_name)

        self.tools = self._get_tool_config(tool_config_path)
        print(f"[INFO] OpenAI Client ready for model '{self.model_name}' with {len(self.tools)} tools configured.")
        self.tool_names = [item['function']['name'] for item in self.tools]






        if enable_graph or mflow_project_dir:
            print("[WARN] `enable_graph` / `mflow_project_dir` are deprecated "
                  "and ignored: M-Flow MCP graph tools have been removed.")


        self.state_manager = StateManager()
        self.tool_library = ToolLibrary(self.state_manager, tokenizer, document_content,
                                          embedding_config=embedding_config,
                                          vllm_client=self.vllm_client,
                                          vllm_model_name=self.model_name,
                                          chunk_size=chunk_size,
                                          overlap=overlap,
                                          boundary_backtrack=boundary_backtrack,
                                          max_read_multi_chunks=self._max_chunks_once,
                                          reranker_config=reranker_config,
                                          use_reranker=use_reranker)
        self.tokenizer = tokenizer
        self.full_history: List[Dict[str, Any]] = []
        self.ctx_counter = 0
        self.deleted_msg_ids = set()
        self.summarized_msg_ids = {}
        self.truncated_msg_ids = {}
        self.compressed_msg_ids = set()





        self.restorable_msg_ids: Set[int] = set()
        self.snapshots: List[List[Dict[str, Any]]] = []

        self.logger = logger
        self.api_call_counter = 0
        self.temperature = temperature
        self.max_context_exp = max_context_exp
        self.max_turns = max_turns_exp
        self.max_output_tokens = max_output_tokens
        self.topp = topp
        self.topk = topk
        self.delete_assistant_tool_call_only = delete_assistant_tool_call_only
        self.allow_text_tool_call_fallback = allow_text_tool_call_fallback
        self.max_search_calls = None
        self.search_call_counter = 0
        self.max_no_tool_retries = 3
        self.max_token_window = None
        self.auto_delete_on_context_overflow = True
        self._retry_hint = None



        self._no_tool_nudge_active = False




        self._search_limit_nudge_active = False

    def _get_system_prompt_text(self, system_prompt_name=None):
        if system_prompt_name is None:
            system_prompt_name = "FSM_PLAN_BM25_MC_PROMPT"
        from infer.configs import prompts
        system_prompt = getattr(prompts, system_prompt_name)
        print(f"[INFO] Using system prompt: {system_prompt_name}")
        return system_prompt

    def _get_tool_config(self, tool_config_path=None):
        if tool_config_path:
            print(f"[INFO] Using custom tool config: {tool_config_path}")
            return read_json(tool_config_path)
        default_path = os.path.join(os.path.dirname(__file__), "..", "tools", "context-shaper_tools.json")
        print(f"[INFO] Using default tool config: {default_path}")
        return read_json(default_path)

    def _resolve_msg_entry(self, msg_id: int):
        for i, m in enumerate(self.full_history):
            if m.get("msg_id") == msg_id:
                return i, m
        return None, None


    _TOOL_CALL_TAG_RE = re.compile(
        r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL
    )

    @staticmethod
    def _try_parse_json_lenient(raw: str):
        """Try json.loads, stripping trailing excess '}' on failure."""
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        while raw.endswith("}"):
            raw = raw[:-1].rstrip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _parse_tool_arguments_lenient(raw):
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ContextPilot._try_parse_json_lenient(text)
            if parsed is None:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _try_parse_legacy_tool_call(raw: str):
        name_match = re.match(r"^\s*([A-Za-z_][\w.-]*)", raw.strip())
        if not name_match:
            return None
        args = {}
        pair_re = re.compile(
            r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
            re.DOTALL,
        )
        for key, value in pair_re.findall(raw[name_match.end():]):
            if key.strip():
                args[key.strip()] = value.strip()
        return {"name": name_match.group(1), "arguments": args}

    def _parse_llm_output(self, resp):
        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        msg = choice.message
        text = (msg.content or "").strip()
        tool_calls = msg.tool_calls or []






        _rc = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)
        if _rc:
            think_block = f"<think>\n{_rc}\n</think>"
            text = (think_block + "\n" + text) if text else think_block



        if tool_calls:
            call = tool_calls[0]

            return (
                text,
                call.function.name,
                self._parse_tool_arguments_lenient(call.function.arguments),
                call.id,
                "tool_calls",
            )




        if self.allow_text_tool_call_fallback and not tool_calls and text and "<tool_call>" in text:
            for m in self._TOOL_CALL_TAG_RE.finditer(text):
                raw_json = m.group(1)
                obj = self._try_parse_json_lenient(raw_json)
                if obj is None or not isinstance(obj, dict):
                    obj = self._try_parse_legacy_tool_call(raw_json)
                    if obj is None or not isinstance(obj, dict):
                        continue
                name = obj.get("name", "unknown")
                args = obj.get("arguments", {})
                tc_id = f"chatcmpl-tool-{uuid.uuid4().hex[:32]}"

                clean_text = self._TOOL_CALL_TAG_RE.sub("", text).strip()
                print(f"    [INFO] Parsed tool call from assistant text fallback: {name}")
                return (
                    clean_text if clean_text else text,
                    name,
                    args if isinstance(args, dict) else {},
                    tc_id,
                    "tool_calls",
                )

        return text, None, None, None, finish_reason

    def _build_api_payload(self, *, keep_think: bool = False, inject_retry_hint: bool = True):
        """
        Build a list[dict] `messages` that conforms to the OpenAI / vLLM format.

        * role == "system"  ➜ first element, contains the long system prompt
        * user / assistant  ➜ strings; assistant may also include `tool_calls`
        * tool results      ➜ role == "tool" (must carry the matching tool_call_id)

        Args:
            keep_think: Retained for backward compatibility with existing
                snapshot call sites. Assistant content is now preserved
                verbatim in both API payloads and snapshots, including any
                <think>…</think> blocks.
            inject_retry_hint: If True (default), inject _retry_hint into the
                first user message. Set to False for snapshots so the hint
                does not persist in recorded trajectories.
        """
        STUB_MESSAGE = "Content has been deleted to save space."

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        external_memory_summary = (
            f"\n\n<external_memory>\n## Available Memories\n"
            f"{self.state_manager.get_notes_summary()}"
            f"\n</external_memory>"
        )
        external_note_summary = (
            f"\n\n<external_note>\n## Available Notes\n"
            f"{self.state_manager.get_simple_notes_summary()}"
            f"\n</external_note>"
        )
        external_context_summary = external_memory_summary + external_note_summary

        for idx, msg in enumerate(self.full_history):
            role = msg.get("role")

            if role == "user":
                text = msg["content"]

                if inject_retry_hint and idx == 0 and self._retry_hint:
                    text += self._retry_hint
                text += (external_context_summary if idx == 0 else "")
                messages.append({"role": "user", "content": text})

            elif role == "assistant":
                msg_id = msg["msg_id"]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    normalized_tool_calls = []
                    for tc in tool_calls:
                        if hasattr(tc, "model_dump"):
                            normalized_tool_calls.append(tc.model_dump())
                        elif isinstance(tc, dict):
                            normalized_tool_calls.append(tc)
                        else:
                            normalized_tool_calls.append(json.loads(json.dumps(tc, default=str)))


                replacement_text = None
                if msg_id in self.deleted_msg_ids:
                    replacement_text = STUB_MESSAGE
                elif msg_id in self.summarized_msg_ids:
                    replacement_text = self.summarized_msg_ids[msg_id]
                elif msg_id in self.truncated_msg_ids:
                    replacement_text = self.truncated_msg_ids[msg_id]

                if replacement_text is not None:
                    tool_calls = msg.get("tool_calls") or []
                    if tool_calls:



                        is_deleted = msg_id in self.deleted_msg_ids
                        args_message = STUB_MESSAGE if is_deleted else replacement_text


                        stubs = []
                        for tc in tool_calls:
                            if hasattr(tc, "model_dump"):
                                tcd = tc.model_dump()
                            elif isinstance(tc, dict):
                                tcd = tc
                            else:
                                tcd = json.loads(json.dumps(tc, default=str))

                            fn = tcd.get("function") or {}
                            name = fn.get("name") or ""

                            stubs.append({
                                "id": tcd.get("id"),
                                "type": "function",
                                "function": {
                                    "name": name,

                                    "arguments": json.dumps(
                                        {"message": args_message},
                                        ensure_ascii=False,
                                    ),
                                },
                            })
                        if is_deleted and self.delete_assistant_tool_call_only:
                            raw_text = " ".join(
                                blk.get("text", "")
                                for blk in (msg.get("content") or [])
                                if isinstance(blk, dict) and blk.get("type") == "text"
                            )
                        else:

                            raw_text = STUB_MESSAGE

                        messages.append({
                            "role": "assistant",
                            "content": raw_text.strip() if raw_text.strip() else "",
                            "tool_calls": stubs
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": replacement_text,
                        })

                    continue


                content_blocks = msg.get("content") or []
                raw_text = " ".join(
                    blk.get("text", "")
                    for blk in content_blocks
                    if isinstance(blk, dict) and blk.get("type") == "text"
                )


                cleaned_text = raw_text.strip()
                assistant_msg = {
                    "role": "assistant",

                    "content": (cleaned_text if cleaned_text else None),
                }

                if tool_calls:
                    assistant_msg["tool_calls"] = normalized_tool_calls
                messages.append(assistant_msg)

            elif role == "tool":
                msg_id = msg["msg_id"]
                msg_id_ia = msg["msg_id(invoking_assistant)"]
                tool_use_id = msg["tool_use_id"]
                tool_result_content_cp = deepcopy(msg["content"])
                tool_result_content_cp["msg_id"] = msg_id
                tool_result_content_cp["msg_id(invoking_assistant)"] = msg_id_ia

                tool_replacement = None
                if msg_id in self.deleted_msg_ids:
                    tool_replacement = STUB_MESSAGE
                elif msg_id in self.summarized_msg_ids:
                    tool_replacement = self.summarized_msg_ids[msg_id]
                elif msg_id in self.truncated_msg_ids:
                    tool_replacement = self.truncated_msg_ids[msg_id]

                if tool_replacement is not None:
                    tool_name = msg.get("tool_name", "unknown")
                    if msg_id in self.deleted_msg_ids and tool_name not in ["nextChunk", "readChunk", "memorize", "updateMemory", "note", "updateNote"]:
                        print(f"[INFO] Attempting to delete {msg.get('tool_name', 'unknown')}")
                    tool_result_content_cp = {
                        "msg_id": msg_id,
                        "msg_id(invoking_assistant)": msg_id_ia,
                        "status": "success",
                        "message": tool_replacement,
                        "original_tool": msg.get("tool_name", "unknown")
                    }
                    if msg.get("tool_name") in ["nextChunk", "readChunk"]:
                        if "reading_progress" in msg["content"]:
                            tool_result_content_cp["reading_progress"] = msg["content"]["reading_progress"]
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result_content_cp, ensure_ascii=False),
                        "tool_call_id": tool_use_id,
                    }
                )






        if inject_retry_hint and self._no_tool_nudge_active:
            messages.append({
                "role": "user",
                "content": self.NO_TOOL_NUDGE_USER_PROMPT,
            })






        if inject_retry_hint and self._search_limit_nudge_active:
            messages.append({
                "role": "user",
                "content": self.SEARCH_LIMIT_NUDGE_USER_PROMPT,
            })

        return messages

    def _call_llm_api(self, messages, tools=None):
        body_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools if tools is not None else self.tools,
            "temperature": self.temperature,
            "top_p": self.topp,
            "max_tokens": self.max_output_tokens,

        }
        seed_base = getattr(self, "_sampling_seed_base", None)
        if seed_base is not None:
            body_kwargs["seed"] = (int(seed_base) + self.api_call_counter) & 0x7FFFFFFF
        extra_body = dict(self.openai_extra_body)
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
            except (APIError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
                status_code = getattr(e, "status_code", None)
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise
                if tries >= max_tries:
                    raise
                wait = 5 * (2**tries)
                print(f"[API] {e} - retrying in {wait}s")
                time.sleep(wait)
                tries += 1



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
        keep_count = int(len(words) * compression_rate / 100.0)
        if keep_count <= 0:
            keep_count = 1
        keep_count = min(len(words), keep_count)
        sampled_indices = sorted(random.sample(range(len(words)), keep_count))
        return " ".join(words[i] for i in sampled_indices)

    @staticmethod
    def _extract_context_text(entry: Dict[str, Any]) -> str:
        role = entry.get("role")
        if role == "tool":
            return json.dumps(entry.get("content", {}), ensure_ascii=False)
        if role == "assistant":
            return " ".join(
                blk.get("text", "")
                for blk in (entry.get("content") or [])
                if isinstance(blk, dict) and blk.get("type") == "text"
            )
        return ""

    def _compress_context_with_llmlingua2(self, text: str, compression_rate: float) -> str:
        if compression_rate <= 0:
            return ""
        cfg_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "openai_endpoint_llmlingua2.json"
        ))
        cfg = read_json(cfg_path)
        base_url = cfg.get("OPENAI_BASE_URL")
        api_key = cfg.get("OPENAI_API_KEY") or "EMPTY"
        model_id = cfg.get("MODEL_ID", "LLMLingua-2")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120) if base_url else OpenAI(api_key=api_key, timeout=120)
        keep_words = max(1, int(len(str(text).split()) * compression_rate / 100.0)) if compression_rate > 0 else 0
        resp = client.chat.completions.create(
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
        return (resp.choices[0].message.content or "").strip()

    def _execute_tool(self, action, params):
        if action == "checkBudget" or action == "getContextStats":
            messages = self._build_api_payload()
            tokenized_messages = self.tokenizer.apply_chat_template(
                self._convert_payload_for_chat_template(messages),
                tools=self.tools, add_generation_prompt=False, tokenize=True
            )
            conv_rounds = len(self.full_history) // 2
            message_len = len(tokenized_messages)
            budget_info =  {
                "conv_rounds": conv_rounds,
                "available_tokens": max(self.max_context_exp - message_len - self.max_output_tokens, 0),
                "available_rounds": max(self.max_turns - conv_rounds, 0),
            }
            if action == "checkBudget":
                return budget_info
            else:
                context_stats = self.tool_library.getContextStats(params or {})
                context_stats.update(budget_info)
                return context_stats

        if action == "deleteContext":
            msg_id = params.get("msg_id")
            if msg_id is None:
                return {"error": "msg_id is required"}
            idx, entry = self._resolve_msg_entry(int(msg_id))
            if entry is None:
                return {"error": f"msg_id {msg_id} not found"}
            role = entry.get("role")
            if role == "user":
                return {"error": "Deleting user messages is not supported"}
            elif role in ("assistant", "tool"):
                self.deleted_msg_ids.add(int(msg_id))

                self.restorable_msg_ids.add(int(msg_id))
                return {"status": "success", "deleted_msg_id": int(msg_id), "deleted_role": role}
            return {"error": f"Unsupported role '{role}' for deletion"}

        if action == "truncateContext":
            msg_id = params.get("msg_id")
            start_sentence = params.get("start_sentence")
            stop_sentence = params.get("stop_sentence")
            if msg_id is None or start_sentence is None or stop_sentence is None:
                return {"error": "msg_id, start_sentence, and stop_sentence are all required."}
            idx, entry = self._resolve_msg_entry(int(msg_id))
            if entry is None:
                return {"error": f"msg_id {msg_id} not found"}
            role = entry.get("role")

            if role == "tool":
                text = json.dumps(entry.get("content", {}), ensure_ascii=False)
            elif role == "assistant":
                text = " ".join(
                    blk.get("text", "")
                    for blk in (entry.get("content") or [])
                    if isinstance(blk, dict) and blk.get("type") == "text"
                )
            else:
                return {"error": f"truncateContext does not support role '{role}'."}

            start_idx = text.find(start_sentence)
            if start_idx == -1:
                return {"error": f"start_sentence not found in message {msg_id}."}

            search_from = start_idx + len(start_sentence)
            stop_idx = text.find(stop_sentence, search_from)
            if stop_idx == -1:
                return {"error": f"stop_sentence not found after start_sentence in message {msg_id}."}

            truncated = text[start_idx:stop_idx + len(stop_sentence)]

            truncated = "[Original long context has been truncated to the following message to save space.]\nTruncated Content: \n" + truncated

            self.truncated_msg_ids[int(msg_id)] = truncated

            self.restorable_msg_ids.add(int(msg_id))
            return {"status": "success", "msg_id": int(msg_id), "truncated_length": len(truncated)}

        if action == "summarizeContext":
            msg_id = params.get("msg_id")
            summary = params.get("summary")
            if msg_id is None or summary is None:
                return {"error": "msg_id and summary are both required."}
            idx, entry = self._resolve_msg_entry(int(msg_id))
            if entry is None:
                return {"error": f"msg_id {msg_id} not found"}
            role = entry.get("role")
            if role not in ("tool", "assistant"):
                return {"error": f"summarizeContext does not support role '{role}'."}

            summary = "[Original long context has been summarized to the following message to save space.]\nSummarized Content: \n" + summary

            self.summarized_msg_ids[int(msg_id)] = summary
            self.compressed_msg_ids.discard(int(msg_id))

            self.restorable_msg_ids.add(int(msg_id))
            return {"status": "success", "msg_id": int(msg_id), "summary_length": len(summary)}

        if action == "compressContext":
            msg_id = params.get("msg_id")
            if msg_id is None:
                return {"error": "msg_id is required."}
            try:
                msg_id_int = int(msg_id)
            except (ValueError, TypeError):
                return {"error": f"msg_id must be an integer, got {msg_id!r}"}
            idx, entry = self._resolve_msg_entry(msg_id_int)
            if entry is None:
                return {"error": f"msg_id {msg_id} not found"}
            role = entry.get("role")
            if role not in ("tool", "assistant"):
                return {"error": f"compressContext does not support role '{role}'."}
            text = self._extract_context_text(entry)
            compression_rate = self._normalize_compression_rate(params.get("compression_rate", 10))
            fallback_used = False
            error_message = None
            try:
                compressed_body = self._compress_context_with_llmlingua2(text, compression_rate)
            except Exception as exc:
                fallback_used = True
                error_message = f"{type(exc).__name__}: {exc}"
                compressed_body = self._fallback_compress_text(text, compression_rate)
            compressed = (
                "[Original long context has been compressed to the following message to save space.]\n"
                f"Compression Rate: {compression_rate:.2f}%\n"
                "Compressed Content: \n" + compressed_body
            )
            self.summarized_msg_ids[msg_id_int] = compressed
            self.compressed_msg_ids.add(msg_id_int)
            self.restorable_msg_ids.add(msg_id_int)
            result = {
                "status": "success",
                "msg_id": msg_id_int,
                "compression_rate": compression_rate,
                "compressed_length": len(compressed),
                "fallback_used": fallback_used,
            }
            if error_message:
                result["fallback_reason"] = error_message
            return result

        if action == "restoreContext":









            msg_id = params.get("msg_id")
            if msg_id is None:
                return {"error": "msg_id is required"}
            try:
                msg_id_int = int(msg_id)
            except (ValueError, TypeError):
                return {"error": f"msg_id must be an integer, got {msg_id!r}"}
            if msg_id_int not in self.restorable_msg_ids:
                return {"error": (
                    f"msg_id {msg_id_int} is not in the list of restorable msg_ids. "
                    f"You can only restore a msg_id that was previously pruned by "
                    f"deleteContext / truncateContext / summarizeContext and has not "
                    f"been restored yet. Restorable msg_ids: "
                    f"{sorted(self.restorable_msg_ids)}."
                )}
            restored_from = []
            if msg_id_int in self.deleted_msg_ids:
                self.deleted_msg_ids.discard(msg_id_int)
                restored_from.append("deleteContext")
            if msg_id_int in self.truncated_msg_ids:
                self.truncated_msg_ids.pop(msg_id_int, None)
                restored_from.append("truncateContext")
            if msg_id_int in self.summarized_msg_ids:
                self.summarized_msg_ids.pop(msg_id_int, None)
                if msg_id_int in self.compressed_msg_ids:
                    restored_from.append("compressContext")
                else:
                    restored_from.append("summarizeContext")
            self.compressed_msg_ids.discard(msg_id_int)
            self.restorable_msg_ids.discard(msg_id_int)
            return {
                "status": "success",
                "msg_id": msg_id_int,
                "restored_from": restored_from,
            }

        if hasattr(self.tool_library, action):
            return getattr(self.tool_library, action)(params or {})
        return {"error": f"Tool '{action}' not found."}











    def _execute_plan_tool(self, params: Dict) -> Dict:
        """Execute the `plan` tool.

        The caller is expected to have already parsed ``params`` out of the
        model's tool-call arguments. The only required argument is
        ``strategy``: a non-empty natural-language plan that reflects on
        progress and states the concrete next actions. The tool itself does
        NOT produce any planning text — it just validates the argument and
        returns a short success message so the pipeline can advance.
        """
        strategy = params.get("strategy") if isinstance(params, dict) else None
        if not isinstance(strategy, str) or not strategy.strip():
            return {
                "error": "plan tool requires a non-empty 'strategy' argument "
                         "containing your reflection and next-step plan."
            }
        return {
            "status": "success",
            "message": "Plan recorded. Please proceed according to your strategy above.",
        }


    def _auto_delete_earliest_readchunk(self) -> bool:
        """Auto-recovery when context length is exceeded.

        Strategy:
        - Count undiscarded readChunk/nextChunk tool responses.
        - If more than 1 remain, delete the earliest one.
        - If 1 or fewer remain, delete the earliest undiscarded searchEngine
          tool response instead (readChunk results are too precious to lose).

        Deletion is performed by injecting a synthetic ``deleteContext``
        tool-call turn (assistant + tool response) into ``full_history``.

        Returns True if a turn was injected, False if nothing to delete.
        """

        readchunk_msg_ids = []
        for msg in self.full_history:
            if (msg.get("role") == "tool"
                    and msg.get("tool_name") in ("readChunk", "nextChunk")
                    and msg["msg_id"] not in self.deleted_msg_ids):
                readchunk_msg_ids.append(int(msg["msg_id"]))


        target_msg_id = None
        target_tool_name = None

        if len(readchunk_msg_ids) > 1:

            target_msg_id = readchunk_msg_ids[0]
            target_tool_name = "readChunk/nextChunk"
        else:


            for msg in self.full_history:
                if (msg.get("role") == "tool"
                        and msg.get("tool_name") == "searchEngine"
                        and msg["msg_id"] not in self.deleted_msg_ids):
                    target_msg_id = int(msg["msg_id"])
                    target_tool_name = "searchEngine"
                    break

        if target_msg_id is None:
            return False


        self.ctx_counter += 1
        asst_msg_id = self.ctx_counter
        synthetic_tool_use_id = f"recovery_delete_{target_msg_id}"

        self.full_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": synthetic_tool_use_id,
                "type": "function",
                "function": {
                    "name": "deleteContext",
                    "arguments": json.dumps({"msg_id": target_msg_id}),
                },
            }],
            "msg_id": asst_msg_id,
        })


        result = self._execute_tool("deleteContext", {"msg_id": target_msg_id})


        self.ctx_counter += 1
        tool_msg_id = self.ctx_counter
        self.full_history.append({
            "role": "tool",
            "content": result,
            "msg_id": tool_msg_id,
            "msg_id(invoking_assistant)": asst_msg_id,
            "tool_use_id": synthetic_tool_use_id,
            "tool_name": "deleteContext",
        })

        print(f"[RECOVERY] Injected deleteContext turn: "
              f"target {target_tool_name} tool msg_id={target_msg_id}, "
              f"synthetic assistant msg_id={asst_msg_id}, "
              f"synthetic tool msg_id={tool_msg_id}, "
              f"result={result}")
        return True


    def set_max_search_calls(self, limit: int):
        """Set the maximum number of searchEngine calls allowed."""
        self.max_search_calls = limit
        print(f"[INFO] searchEngine call limit set to {limit}")

    def set_max_no_tool_retries(self, limit: int):
        """Set the maximum number of retries when the model fails to call a tool."""
        self.max_no_tool_retries = limit
        print(f"[INFO] max no-tool retries set to {limit}")

    def set_max_token_window(self, limit: int):
        """Set the hard token window limit for API payloads."""
        self.max_token_window = limit
        print(f"[INFO] max token window set to {limit}")

    def set_auto_delete_on_context_overflow(self, enabled: bool):
        """Enable or disable automatic readChunk deletion after overflow."""
        self.auto_delete_on_context_overflow = bool(enabled)
        print(
            "[INFO] auto delete on context overflow set to "
            f"{self.auto_delete_on_context_overflow}"
        )

    def set_max_chunks_once(self, limit: int):
        """Set the maximum number of chunk_ids honored per readMultiChunks call.

        Extra chunk_ids beyond this limit are silently ignored at tool-call
        time, and the returned message is prefixed with a notice explaining
        that the tail was truncated.
        """
        limit = max(1, int(limit or 1))
        self._max_chunks_once = limit
        if hasattr(self, "tool_library") and self.tool_library is not None:
            self.tool_library._max_read_multi_chunks = limit
        print(f"[INFO] max chunks per readMultiChunks call set to {limit}")

    def _get_tools_for_api(self, force_finish: bool = False):
        """Return the tool list for the current API call.

        If *force_finish* is True, only the ``finish`` tool is returned so
        the model is forced to produce a final answer.

        Otherwise, if ``max_search_calls`` is set and the counter has
        reached the limit, the ``searchEngine`` tool is excluded as a
        fallback safeguard (normally force_finish is already True).
        """
        if force_finish:
            return [t for t in self.tools if t["function"]["name"] == "finish"]
        if (self.max_search_calls is not None
                and self.search_call_counter >= self.max_search_calls):
            return [t for t in self.tools
                    if t["function"]["name"] not in ("searchEngine", "semanticSearch", "hybridSearch")]
        return self.tools

    FORCE_FINISH_USER_PROMPT = (
        "Time is up. Based on the information you've already gathered, "
        "call `finish` now with your best answer. Do not call any other tool."
    )




    NO_TOOL_NUDGE_USER_PROMPT = (
        "Your previous response ended without calling a tool. "
        "Keep your thinking brief and you MUST call exactly one tool in your next response."
    )




    SEARCH_LIMIT_NUDGE_USER_PROMPT = (
        "You have reached the maximum number of search-tool calls. "
        "Based on the information you've already gathered, call `finish` now with your best answer."
    )

    def _inject_force_finish_messages(self):
        """Append an empty-assistant turn and a user nudge message to
        ``self.full_history`` so the next LLM call sees a direct instruction
        to invoke ``finish`` immediately.

        These messages are persisted into ``full_history`` (and therefore
        into snapshots via the normal append path), but no snapshot is cut
        here -- the caller is expected to let the main loop record the
        snapshot at the *next* assistant turn (i.e. the one that responds
        to this nudge), so the split lands on the finish-response turn.
        """
        self.ctx_counter += 1
        empty_assistant_msg_id = self.ctx_counter
        self.full_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "msg_id": empty_assistant_msg_id,
        })
        self.full_history.append({
            "role": "user",
            "content": self.FORCE_FINISH_USER_PROMPT,
        })

    def _convert_payload_for_chat_template(self, messages):
        """Convert OpenAI-format messages when the active template needs it.

        Qwen's Jinja2 chat template expects ``tool_call.arguments`` to be a
        **dict** and ``tool_call.name`` at the top level, whereas the OpenAI
        SDK format nests them under ``function`` with ``arguments`` as a JSON
        string. Gemma 4's native template expects the OpenAI ``function``
        wrapper and must not receive that Qwen-only flattening. Detect the
        native OpenAI shape from the loaded template before converting.
        """
        chat_template = str(getattr(self.tokenizer, "chat_template", "") or "")
        if ("tool_call['function']" in chat_template
                or 'tool_call["function"]' in chat_template):
            return messages

        converted = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                new_tool_calls = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    new_tool_calls.append({
                        "name": fn.get("name", "unknown"),
                        "arguments": args if isinstance(args, dict) else {},
                    })
                new_msg["tool_calls"] = new_tool_calls
                converted.append(new_msg)
            else:
                converted.append(msg)
        return converted

    def _enforce_token_window(self, api_payload, tools):
        """Stub-out the earliest assistant+tool turn pairs until the
        tokenised payload fits within ``self.max_token_window``.

        Messages are kept in ``full_history`` verbatim; we only add their
        ``msg_id`` to ``self.deleted_msg_ids`` so that ``_build_api_payload``
        renders them as the standard ``STUB_MESSAGE`` (same mechanism as
        ``deleteContext``). Because ``deleted_msg_ids`` is persistent, the
        stub is carried forward into every subsequent payload and snapshot
        produced from this turn onwards -- but previously captured entries
        in ``self.snapshots`` are left untouched, so historical snapshots
        continue to reflect exactly what the model saw at the time they
        were recorded.

        Only turns *after* the first user message (index 0) are eligible.
        Turns are stubbed in chronological order, always as an
        (assistant, tool) pair so the conversation stays well-formed.
        """
        def _token_len(payload, tls):
            return len(self.tokenizer.apply_chat_template(
                self._convert_payload_for_chat_template(payload),
                tools=tls, add_generation_prompt=True, tokenize=True
            ))

        tok_len = _token_len(api_payload, tools)
        if tok_len <= self.max_token_window:
            return api_payload

        print(f"[TOKEN_WINDOW] Payload length {tok_len} exceeds limit "
              f"{self.max_token_window}. Stubbing out earliest turns...")

        stubbed_ids: set = set()
        attempted_asst_indices: set[int] = set()
        while tok_len > self.max_token_window:


            asst_idx = None
            for i, msg in enumerate(self.full_history):
                if i == 0:
                    continue
                if i in attempted_asst_indices:
                    continue
                if msg.get("role") != "assistant":
                    continue
                if msg.get("msg_id") in self.deleted_msg_ids:
                    continue
                asst_idx = i
                break

            if asst_idx is None:
                print("[TOKEN_WINDOW] No more assistant turns available to stub.")
                break

            asst_msg = self.full_history[asst_idx]
            asst_msg_id = asst_msg.get("msg_id")
            attempted_asst_indices.add(asst_idx)


            tool_msg_id = None
            tool_name = "unknown"
            if (asst_idx + 1 < len(self.full_history)
                    and self.full_history[asst_idx + 1].get("role") == "tool"):
                tool_msg = self.full_history[asst_idx + 1]
                tool_msg_id = tool_msg.get("msg_id")
                tool_name = tool_msg.get("tool_name", "unknown")


            if asst_msg_id is not None:
                self.deleted_msg_ids.add(int(asst_msg_id))
                stubbed_ids.add(int(asst_msg_id))
            if tool_msg_id is not None:
                self.deleted_msg_ids.add(int(tool_msg_id))
                stubbed_ids.add(int(tool_msg_id))

            if tool_msg_id is not None:
                print(f"[TOKEN_WINDOW] Stubbed assistant(msg_id={asst_msg_id}) + "
                      f"tool/{tool_name}(msg_id={tool_msg_id})")
            else:
                print(f"[TOKEN_WINDOW] Stubbed standalone assistant(msg_id={asst_msg_id})")




            self._sync_fsm_state_on_stub(asst_msg_id, tool_msg_id, tool_name)


            api_payload = self._build_api_payload()
            new_tok_len = _token_len(api_payload, tools)



            if new_tok_len >= tok_len:
                print(f"[TOKEN_WINDOW] Stubbing had no effect ({tok_len} -> "
                      f"{new_tok_len}); trying the next eligible turn.")
            tok_len = new_tok_len












        print(f"[TOKEN_WINDOW] Stubbed {len(stubbed_ids)} messages. "
              f"New payload length: {tok_len}")
        return api_payload

    def _sync_fsm_state_on_stub(self, asst_msg_id, tool_msg_id, tool_name):
        """Keep FSM bookkeeping in sync when _enforce_token_window forcibly
        stubs out an (assistant, tool) pair.

        The FSM subclass tracks pending search / plan results via several
        sets/counters so that S4 can decide whether to enter S4.5 / S4.6,
        and so S5 / S4.5 / S4.6 can validate which msg_ids the model is
        allowed to target. When the token-window guard silently stubs a
        search/plan/readChunk tool response (and its assistant parent),
        those tracking structures would otherwise drift out of sync with
        what the model actually sees.

        This method:
          * discards the stubbed msg_ids from every ``_allowed_msg_ids_for_*``
            set and from ``_search_tool_msg_ids`` / ``_plan_assistant_msg_ids``
            / ``_readchunk_tool_msg_ids`` so later validation won't point
            the model at a msg_id whose content is now a stub;
          * bumps ``_search_delete_count`` / ``_plan_delete_count`` when
            the stubbed tool was a search / plan call, so the
            ``pending_search`` / ``pending_plan`` bookkeeping reflects
            reality (and S4.5 / S4.6 aren't triggered for messages the
            model can no longer meaningfully delete).

        All fields are accessed via ``getattr`` with a default, so the
        base class (which has no FSM fields) continues to work.
        """
        def _discard_from(attr_name, value):
            container = getattr(self, attr_name, None)
            if container is None or value is None:
                return
            try:
                if isinstance(container, (set, list)):
                    if value in container:
                        if isinstance(container, set):
                            container.discard(value)
                        else:
                            try:
                                container.remove(value)
                            except ValueError:
                                pass
            except TypeError:
                pass


        try:
            asst_id = int(asst_msg_id) if asst_msg_id is not None else None
        except (ValueError, TypeError):
            asst_id = None
        try:
            tool_id = int(tool_msg_id) if tool_msg_id is not None else None
        except (ValueError, TypeError):
            tool_id = None





        for attr in ("_allowed_msg_ids_for_s5",
                     "_allowed_msg_ids_for_s5_5",
                     "_allowed_msg_ids_for_s4_5",
                     "_allowed_msg_ids_for_s4_6"):
            _discard_from(attr, asst_id)

        if tool_id is None:
            return


        for attr in ("_allowed_msg_ids_for_s5",
                     "_allowed_msg_ids_for_s5_5",
                     "_allowed_msg_ids_for_s4_5",
                     "_allowed_msg_ids_for_s4_6"):
            _discard_from(attr, tool_id)

        search_tools = {"searchEngine", "semanticSearch", "hybridSearch"}
        read_tools = {"readChunk", "readMultiChunks"}

        search_ids = getattr(self, "_search_tool_msg_ids", None)
        plan_ids = getattr(self, "_plan_assistant_msg_ids", None)
        read_ids = getattr(self, "_readchunk_tool_msg_ids", None)

        if tool_name in search_tools and isinstance(search_ids, set) \
                and tool_id in search_ids:
            search_ids.discard(tool_id)


            cur = getattr(self, "_search_delete_count", None)
            if cur is not None:
                self._search_delete_count = cur + 1
            print(f"[TOKEN_WINDOW] FSM sync: search msg_id={tool_id} "
                  f"removed; search_delete_count="
                  f"{getattr(self, '_search_delete_count', 'n/a')}")
        elif tool_name == "plan" and isinstance(plan_ids, set) \
                and asst_id is not None and asst_id in plan_ids:





            plan_ids.discard(asst_id)
            cur = getattr(self, "_plan_delete_count", None)
            if cur is not None:
                self._plan_delete_count = cur + 1
            print(f"[TOKEN_WINDOW] FSM sync: plan assistant msg_id={asst_id} "
                  f"removed; plan_delete_count="
                  f"{getattr(self, '_plan_delete_count', 'n/a')}")
        elif tool_name in read_tools and isinstance(read_ids, list) \
                and tool_id in read_ids:
            try:
                read_ids.remove(tool_id)
            except ValueError:
                pass
            print(f"[TOKEN_WINDOW] FSM sync: readChunk msg_id={tool_id} "
                  f"removed from tracking list.")

    def run(self, user_query, max_turns_to_fail=80):
        """Run the ContextPilot interaction loop."""

        self.full_history.append({"role": "user", "content": user_query})
        self.ctx_counter = 0
        self.search_call_counter = 0
        turn = 0
        force_finish = False
        no_tool_retry_counter = 0
        try:
            while turn <= max_turns_to_fail:
                print(f"\n--- Round {turn} (Max {max_turns_to_fail} rounds, expected within {self.max_turns} rounds) ---")
                api_payload = self._build_api_payload()

                current_tools = self._get_tools_for_api(force_finish=force_finish)


                if self.max_token_window is not None:
                    api_payload = self._enforce_token_window(api_payload, current_tools)

                try:
                    resp = self._call_llm_api(api_payload, tools=current_tools)
                except Exception as e:
                    err = f"LLM API failed after retries: {type(e).__name__}: {e}"
                    print("[ERROR]", err)



                    if self.auto_delete_on_context_overflow and (
                        "maximum context length" in str(e)
                        or "context_length_exceeded" in str(e)
                    ):
                        freed = self._auto_delete_earliest_readchunk()
                        if freed:
                            print("[RECOVERY] Deleted earliest readChunk turn to free context space. Retrying...")
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

                self.full_history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": thought or ""}],
                    "tool_calls": raw_tool_calls,
                    "msg_id": msg_id
                })
                print("[RUN] Assistant:", thought)


                self.snapshots.append(self._build_api_payload(keep_think=True, inject_retry_hint=False))

                if stop_reason == 'tool_calls':
                    print(f"[RUN] Assistant action: Call tool `{action}`, parameters: {params}")
                    no_tool_retry_counter = 0
                    self._no_tool_nudge_active = False


                    if action not in self.tool_names:
                        result = {"error": f"Tool '{action}' not found."}
                    else:
                        try:
                            if action == "plan":





                                result = self._execute_plan_tool(params)
                            else:
                                result = self._execute_tool(action, params)
                        except Exception as e:
                            result = {"error": f"Tool '{action}' execution failed: {type(e).__name__}: {e}"}
                            print(f"[WARN] Tool '{action}' raised {type(e).__name__}: {e}")

                    self.ctx_counter += 1
                    msg_id_tool = self.ctx_counter
                    self.full_history.append({
                        "role": "tool",
                        "content": deepcopy(result),
                        "msg_id": msg_id_tool,
                        "msg_id(invoking_assistant)": msg_id,
                        "tool_use_id": tool_use_id,
                        "tool_name": action
                    })

                    result_preview = json.dumps(result, ensure_ascii=False)
                    if len(result_preview) > 200:
                        result_preview = result_preview[:200] + "..."
                    print(f"[RUN] Tool result (ID: {msg_id_tool}): {result_preview}")


                    if action in ("searchEngine", "semanticSearch", "hybridSearch"):
                        self.search_call_counter += 1
                        if (self.max_search_calls is not None
                                and self.search_call_counter >= self.max_search_calls
                                and not self._search_limit_nudge_active):
                            print(f"[INFO] search call limit reached ({self.max_search_calls}). "
                                  f"Activating payload-only search-limit nudge.")




                            self._search_limit_nudge_active = True

                    if action == "finish":
                        print(f"\n--- Final Answer --- \n{result.get('final_answer', 'No final answer provided.')}")
                        break

                else:


                    no_tool_retry_counter += 1
                    if no_tool_retry_counter <= self.max_no_tool_retries:
                        print(f"[WARN] Model did not call a tool (stop_reason='{stop_reason}'). "
                              f"Retry {no_tool_retry_counter}/{self.max_no_tool_retries}. "
                              f"Removing last assistant message and re-prompting.")

                        self.full_history.pop()
                        self.ctx_counter -= 1

                        if self.snapshots:
                            self.snapshots.pop()




                        self._no_tool_nudge_active = True
                        continue
                    else:
                        print(f"[INFO] Model failed to call a tool after {self.max_no_tool_retries} retries. "
                              f"Stopping (stop_reason='{stop_reason}').")
                        self._no_tool_nudge_active = False
                        break

                turn += 1






                if turn >= self.max_turns and not force_finish:
                    print(f"[INFO] Reached expected max rounds ({self.max_turns}). "
                          f"Switching to force-finish mode.")
                    force_finish = True
                    self._inject_force_finish_messages()

            if turn > max_turns_to_fail:
                print(f"[INFO] Reached hard max rounds {max_turns_to_fail}, stopping execution.")

            self.snapshots.append(self._build_api_payload(keep_think=True, inject_retry_hint=False))
            self.tool_library.clearCurrentDocument()
            return self._build_api_payload()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl-C). Exiting gracefully...")
            self.tool_library.clearCurrentDocument()
            return self._build_api_payload()



    def _extract_final_answer(self):

        for msg in reversed(self.full_history):
            if msg.get("role") == "tool" and msg.get("tool_name") == "finish":
                content = msg.get("content", {})
                if isinstance(content, dict) and "final_answer" in content:
                    return content.get("final_answer")

        for msg in reversed(self.full_history):
            if msg.get("role") == "assistant" and msg.get("msg_id") not in self.deleted_msg_ids:
                text = ""
                for blk in (msg.get("content") or []):
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        text += blk.get("text") or ""
                if (text := text.strip()):
                    return text
        return ""


    def _sanitize_for_json(self, obj):
        """Recursively convert SDK / complex objects to plain JSON-safe types."""

        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        elif hasattr(obj, "dict"):
            obj = obj.dict()


        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in obj]


        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except Exception:
                return str(obj)


        try:
            json.dumps(obj, ensure_ascii=False)
            return obj
        except TypeError:
            return str(obj)


    def save_trajectory(self, out_dir="logs", filename=None, correct_answer=None, meta_info=None):
        snapshot = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "model": getattr(self, "model_name", "Qwen3-8B-Agentic"),
            "session_notes_summary": self.state_manager.get_notes_summary(),
            "api_call_count": self.api_call_counter,
            "final_answer": self._extract_final_answer(),
            "correct_answer": correct_answer,
            "full_history": self._sanitize_for_json(self.full_history),
            "deleted_msg_ids": sorted(self.deleted_msg_ids),
            "summarized_msg_ids": {str(k): v for k, v in self.summarized_msg_ids.items()},
            "truncated_msg_ids": {str(k): v for k, v in self.truncated_msg_ids.items()},
            "compressed_msg_ids": sorted(self.compressed_msg_ids),
            "restorable_msg_ids": sorted(self.restorable_msg_ids),
            "snapshots": self.snapshots,
            "meta_info": meta_info,
        }

        os.makedirs(out_dir or ".", exist_ok=True)
        if not filename:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"trajectory_{ts}.json"
        path = os.path.join(out_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Trajectory saved to: {path}")
        return path
