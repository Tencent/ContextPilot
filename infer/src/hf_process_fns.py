"""Hugging Face/local dataset adapters for the supported FSM domains."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List


def infinitebench_longbook_choice_eng_i2q(item: Dict[str, Any]) -> str:
    lines = [item["input"].strip()]
    lines.extend(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item["options"]))
    lines.append("Select the best answer from the options above.")
    return "\n".join(lines)


def infinitebench_longbook_choice_eng_i2c(item: Dict[str, Any]) -> str:
    return item["context"].strip()


def infinitebench_longbook_choice_eng_i2a(item: Dict[str, Any]) -> str:
    return chr(65 + item["options"].index(item["answer"][0]))


def novelqa_i2q(item: Dict[str, Any]) -> str:
    lines = [item["question"].strip()]
    lines.extend(f"{key}. {item['options'][key]}" for key in sorted(item["options"]))
    lines.append("Select the best answer from the options above.")
    return "\n".join(lines)


def novelqa_i2c(item: Dict[str, Any]) -> str:
    if item.get("context"):
        return item["context"]
    if item.get("context_path"):
        source_path = Path(item["context_path"])
        candidates = [source_path]
        bundle_root = Path(__file__).resolve().parents[1] / "data" / "NovelQA"
        if source_path.is_absolute():
            candidates.append(
                bundle_root / "Books" / "CopyrightProtected" / source_path.name
            )
        else:
            candidates.append(bundle_root / source_path)
        context_root = os.getenv("NOVELQA_CONTEXT_ROOT")
        if context_root:
            candidates.append(Path(context_root).expanduser() / source_path.name)
        context_path = next((path for path in candidates if path.is_file()), None)
        if context_path is None:
            raise FileNotFoundError(
                f"NovelQA context is unavailable for {item['context_path']!r}. "
                "Set NOVELQA_CONTEXT_ROOT to the directory containing the "
                "copyrighted book .txt files."
            )
        with context_path.open("r", encoding="utf-8") as source:
            return source.read()
    raise KeyError("NovelQA item is missing both 'context' and 'context_path'.")


def novelqa_i2a(item: Dict[str, Any]) -> str:
    for key in ("gold", "Gold", "answer", "Answer"):
        if item.get(key):
            return item[key]
    return ""


def novelqa_i2meta(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "book_id", "question_id", "split", "copyright", "title", "author",
        "aspect", "complexity", "context_path",
    )
    return {key: item[key] for key in keys if key in item}


def _clean_longmemeval_session(session: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {key: value for key, value in turn.items() if key != "has_answer"}
        if isinstance(turn, dict) else turn
        for turn in session
    ]


def longmemevals_i2q(item: Dict[str, Any]) -> str:
    return f"Current Date: {item['question_date'].strip()}\nQuestion: {item['question'].strip()}"


def longmemevals_i2c(item: Dict[str, Any]) -> str:
    sessions = []
    for index, (date, session) in enumerate(
        zip(item["haystack_dates"], item["haystack_sessions"]), start=1
    ):
        sessions.append(
            f"### Session {index}:\nSession Date: {date}\nSession Content:\n"
            f"{json.dumps(_clean_longmemeval_session(session), ensure_ascii=False)}\n"
        )
    body = "\n".join(sessions).strip()
    return f"History Chats:\n\n{body}" if body else "History Chats:"


def longmemevals_i2a(item: Dict[str, Any]) -> str:
    return item["answer"]


def longmemevals_i2meta(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_id": item["question_id"],
        "answer_session_ids": item.get("answer_session_ids", []),
    }


def bc_plus_i2q(item: Dict[str, Any]) -> str:
    return item["query"]


def bc_plus_i2c(item: Dict[str, Any]) -> str:
    rng = random.Random(42)
    evidence_doc_text = [doc["text"] for doc in item["evidence_docs"]]
    negative_doc_text = [doc["text"] for doc in item["negative_docs"]]
    all_docs_text = evidence_doc_text + negative_doc_text
    rng.shuffle(all_docs_text)
    return "\n\n".join(all_docs_text)


def bc_plus_i2a(item: Dict[str, Any]) -> str:
    return item["answer"]


def bc_plus_i2meta(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": str(item["query_id"]),
        "gold_doc_ids": [doc["docid"] for doc in item.get("gold_docs", [])],
        "evidence_doc_ids": [
            doc["docid"] for doc in item.get("evidence_docs", [])
        ],
    }
