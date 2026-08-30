"""Hugging Face/local dataset scorers for the supported FSM domains."""

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import openai
from openai import OpenAI
from tqdm import tqdm


LONGMEMEVAL_QA_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def iter_jsonl(path: str, cnt: int = None):
    with open(path, "r", encoding="utf-8") as source:
        for index, line in enumerate(line for line in source if line.strip()):
            if cnt is not None and index >= cnt:
                break
            yield json.loads(line)


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_openai_endpoint_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as source:
        config = json.load(source)
    api_key_env = config.pop("OPENAI_API_KEY_ENV", None)
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        config["OPENAI_API_KEY"] = api_key
    return config


def _resolve_judge_client(judge_model: str, endpoint_config: str = None):
    if judge_model != "endpoint-config":
        raise ValueError("Only endpoint-config judges are supported.")
    if not endpoint_config:
        raise ValueError("endpoint_config is required")
    config = load_openai_endpoint_config(endpoint_config)
    client_kwargs = {
        "api_key": config["OPENAI_API_KEY"],
        "base_url": config["OPENAI_BASE_URL"],
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "600")),
    }
    extra_headers = config.get("OPENAI_EXTRA_HEADERS") or config.get("EXTRA_HEADERS")
    if extra_headers:
        client_kwargs["default_headers"] = {
            str(key): str(value) for key, value in extra_headers.items()
        }
    request_kwargs = {}
    if config.get("OPENAI_EXTRA_BODY"):
        request_kwargs["extra_body"] = config["OPENAI_EXTRA_BODY"]
    request_kwargs["max_tokens_field"] = config.get(
        "OPENAI_MAX_TOKENS_FIELD", "max_tokens"
    )
    request_kwargs["send_temperature"] = config.get(
        "OPENAI_SEND_TEMPERATURE", True
    )
    return OpenAI(**client_kwargs), config["MODEL_ID"], request_kwargs


def chat_completions_with_backoff(
    client, max_retries: int = 10, base_delay: float = 1.0, **kwargs
):
    last_error = None
    for retry_index in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as error:
            last_error = error
            if retry_index == max_retries - 1:
                raise
            delay = min(30.0, base_delay * (2**retry_index)) + random.random()
            time.sleep(delay)
        except openai.APIStatusError as error:
            if error.status_code < 500:
                raise
            last_error = error
            if retry_index == max_retries - 1:
                raise
            delay = min(30.0, base_delay * (2**retry_index)) + random.random()
            time.sleep(delay)
    raise last_error


_TEXT_KEYS = {"content", "text", "reasoning_content", "reasoning", "output_text"}
_METADATA_KEYS = {
    "role", "name", "refusal", "tool_call_id", "function_call", "tool_calls",
    "type", "id", "object", "model", "created", "finish_reason",
}


def _collect_text(obj):
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj.strip()] if obj.strip() else []
    if isinstance(obj, list):
        return [text for item in obj for text in _collect_text(item)]
    if isinstance(obj, dict):
        texts = [text for key in _TEXT_KEYS if key in obj for text in _collect_text(obj[key])]
        texts.extend(
            text
            for key, value in obj.items()
            if key not in _TEXT_KEYS and key not in _METADATA_KEYS
            for text in _collect_text(value)
        )
        return texts
    if hasattr(obj, "model_dump"):
        return _collect_text(obj.model_dump())
    return []


def extract_judge_response_text(completion) -> str:
    if not completion.choices:
        return ""
    message = completion.choices[0].message
    direct = []
    for name in _TEXT_KEYS:
        direct.extend(_collect_text(getattr(message, name, None)))
    candidates = direct or _collect_text(message) or _collect_text(completion)
    return "\n".join(
        text for text in candidates
        if text.strip().lower() not in {"assistant", "user", "system", "tool", "function"}
    )


def _parallel_judge(
    client,
    model: str,
    prompts,
    max_tokens: int,
    workers: int,
    desc: str,
    request_kwargs: dict = None,
):
    results = [None] * len(prompts)
    request_kwargs = dict(request_kwargs or {})
    max_tokens_field = request_kwargs.pop("max_tokens_field", "max_tokens")
    send_temperature = request_kwargs.pop("send_temperature", True)

    def invoke(index: int, prompt: str):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            max_tokens_field: max_tokens,
            **request_kwargs,
        }
        if send_temperature:
            payload["temperature"] = 0
        completion = chat_completions_with_backoff(client, **payload)
        return index, extract_judge_response_text(completion).strip()

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(prompts)))) as pool:
        futures = [pool.submit(invoke, index, prompt) for index, prompt in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            index, result = future.result()
            results[index] = result
    return results


def _choice_correct(prediction, label) -> bool:
    if isinstance(prediction, list):
        prediction = prediction[0] if prediction else ""
    prediction = str(prediction).strip()
    label = str(label)
    match = re.search(r"\b[A-D]\b(?!.*\b[A-D]\b)", prediction)
    if match and match.group(0) in label:
        return True
    if not prediction:
        return False
    if prediction[0] in "ABCD":
        return prediction[0] in label
    if prediction in label:
        return True
    for character in ("\n", '"', "'", ".", ",", "?", "!", "{", "}"):
        prediction = prediction.replace(character, " ")
    while "  " in prediction:
        prediction = prediction.replace("  ", " ")
    for prefix in ("answer is:", "answer:", "answer is", "option is"):
        index = prediction.find(prefix)
        if index < 0:
            continue
        if len(prediction) < index + len(prefix) + 1:
            return False
        suffix = prediction[index + len(prefix) + 1:]
        return any(suffix.startswith(option) for option in label)
    return any(word in "ABCD" and word in label for word in prediction.split())


def score_choice_file(
    preds_path: str,
    results_output: str,
    model_name: str = "unknown",
    label_key: str = "correct_answer",
    pred_key: str = "final_answer",
):
    rows = list(iter_jsonl(preds_path))
    correct = sum(_choice_correct(row.get(pred_key, ""), row.get(label_key, "")) for row in rows)
    score = correct / len(rows) if rows else 0.0
    ensure_parent_dir(results_output)
    with open(results_output, "w", encoding="utf-8") as output:
        output.write(f"Model: {model_name}\n")
        output.write(f"Results Path: {preds_path}\n")
        output.write(f"Task: longbook_choice_eng ({len(rows)} examples)\n")
        output.write(f"Score: {score}\n--------------------\n")
    print(f"Correct Count: {correct}")
    print(f"Total Count: {len(rows)}")
    print(f"Score: {score}")


def evaluate_choice_file(
    file_path: str,
    pred_key: str,
    label_key: str,
    output_key: str = "correct_list",
    task_name: str = "longbook_choice_eng",
    model_name: str = "unknown",
    results_output: str = "",
):
    """Score multiple-choice predictions and annotate each row."""
    rows = list(iter_jsonl(file_path))
    scores = []
    for row in rows:
        score = _choice_correct(row.get(pred_key, ""), row.get(label_key, ""))
        row[output_key] = [bool(score)]
        scores.append(bool(score))
    with open(file_path, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    correct = sum(scores)
    score = correct / len(scores) if scores else 0.0
    if results_output:
        ensure_parent_dir(results_output)
        with open(results_output, "w", encoding="utf-8") as output:
            output.write(f"Model: {model_name}\n")
            output.write(f"Results Path: {file_path}\n")
            output.write(f"Task: {task_name} ({len(rows)} examples)\n")
            output.write(f"Correct Count: {correct}\n")
            output.write(f"Total Count: {len(rows)}\n")
            output.write(f"Score: {score}\n--------------------\n")
    print(f"Correct Count: {correct}")
    print(f"Total Count: {len(rows)}")
    print(f"Score: {score}")


def build_longmemeval_judge_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    if not abstention:
        if task in {"single-session-user", "single-session-assistant", "multi-session"}:
            instructions = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no."
            )
            reference_block = f"Correct Answer: {answer}"
        elif task == "temporal-reasoning":
            instructions = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
                "(e.g., predicting 19 days when the answer is 18), the model's response is still correct."
            )
            reference_block = f"Correct Answer: {answer}"
        elif task == "knowledge-update":
            instructions = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the required answer."
            )
            reference_block = f"Correct Answer: {answer}"
        elif task == "single-session-preference":
            instructions = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal information correctly."
            )
            reference_block = f"Rubric: {answer}"
        else:
            raise NotImplementedError(f"Unsupported LongMemEval task type: {task}")
        final_question = "Is the model response correct?"
    else:
        instructions = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given but the asked information is not."
        )
        reference_block = f"Explanation: {answer}"
        final_question = "Does the model correctly identify the question as unanswerable?"

    return (
        f"{instructions}\n\nQuestion: {question}\n\n{reference_block}\n\n"
        f"Model Response: {response}\n\n{final_question} Return JSON only with this schema: "
        '{"label": true, "reasoning": "brief explanation", '
        '"matched_span": "short quoted span from the response or empty string"}.'
    )


def _coerce_bool_label(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "correct", "1"}:
            return True
        if normalized in {"false", "no", "incorrect", "0"}:
            return False
    return None


def _extract_first_json_object(text: str):
    if not text:
        return None
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.I | re.S)
    if fence:
        stripped = fence.group(1).strip()
    for candidate in [stripped, *re.findall(r"\{.*?\}", stripped, re.S)]:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def parse_longmemeval_judge_response(text: str) -> dict:
    result = {
        "label": False,
        "reasoning": "",
        "matched_span": "",
        "parse_error": False,
        "raw_text": text,
    }
    parsed_json = _extract_first_json_object(text)
    if isinstance(parsed_json, dict):
        for key in ("label", "is_correct", "correct", "verdict"):
            if key in parsed_json:
                label = _coerce_bool_label(parsed_json[key])
                if label is not None:
                    result["label"] = label
                    break
        reasoning = (
            parsed_json.get("reasoning")
            or parsed_json.get("explanation")
            or parsed_json.get("why")
            or ""
        )
        matched_span = (
            parsed_json.get("matched_span")
            or parsed_json.get("answer_span")
            or parsed_json.get("evidence")
            or ""
        )
        result["reasoning"] = (
            reasoning
            if isinstance(reasoning, str)
            else json.dumps(reasoning, ensure_ascii=False)
        )
        result["matched_span"] = (
            matched_span
            if isinstance(matched_span, str)
            else json.dumps(matched_span, ensure_ascii=False)
        )
        if any(key in parsed_json for key in ("label", "is_correct", "correct", "verdict")):
            return result
    patterns = (
        r'"label"\s*:\s*(true|false|"yes"|"no"|"true"|"false")',
        r'"is_correct"\s*:\s*(true|false|"yes"|"no"|"true"|"false")',
        r'"correct"\s*:\s*(true|false|"yes"|"no"|"true"|"false")',
        r'"verdict"\s*:\s*"?(yes|no|true|false)"?',
        r'\blabel\s*[:=]\s*(yes|no|true|false)\b',
        r'\b(?:final\s+)?verdict\s*[:=]\s*(yes|no|true|false)\b',
        r'\bcorrect\s*[:=]\s*(yes|no|true|false)\b',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            label = _coerce_bool_label(match.group(1).strip('"'))
            if label is not None:
                result["label"] = label
                return result
    matches = re.findall(r"\b(yes|no|true|false)\b", text.lower())
    if matches:
        result["label"] = matches[-1] in {"yes", "true"}
        return result
    result["parse_error"] = True
    return result


def _load_longmemeval_references(path: str):
    try:
        with open(path, "r", encoding="utf-8") as source:
            data = json.load(source)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return list(iter_jsonl(path))


def annotate_longmemeval_file(
    preds_path: str,
    ref_path: str,
    output_path: str,
    results_output: str,
    endpoint_config: str,
    pred_key: str = "final_answer",
    judge_model: str = "endpoint-config",
    max_samples: int = None,
    max_judge_tokens: int = 4096,
    num_workers: int = 32,
):
    predictions = list(iter_jsonl(preds_path, cnt=max_samples))
    references = {row["question_id"]: row for row in _load_longmemeval_references(ref_path)}
    client, model, request_kwargs = _resolve_judge_client(
        judge_model, endpoint_config
    )
    selected = []
    prompts = []
    for row in predictions:
        question_id = row.get("question_id")
        if question_id not in references:
            continue
        reference = references[question_id]
        prediction = row.get(pred_key, "")
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else ""
        prompts.append(build_longmemeval_judge_prompt(
            reference["question_type"], reference["question"], reference["answer"],
            prediction, abstention="_abs" in question_id,
        ))
        selected.append((row, reference))

    responses = _parallel_judge(
        client,
        model,
        prompts,
        max_judge_tokens,
        num_workers,
        "Scoring LongMemEval",
        request_kwargs,
    )
    counts = {task: [] for task in LONGMEMEVAL_QA_TYPES}
    abstentions = []
    scored_rows = []
    for (row, reference), response in zip(selected, responses):
        judge_result = parse_longmemeval_judge_response(response or "")
        label = judge_result["label"]
        scored = dict(row)
        scored["autoeval_label"] = {
            "model": model,
            "label": label,
            "judge_response": response,
            "reasoning": judge_result["reasoning"],
            "matched_span": judge_result["matched_span"],
            "parse_error": judge_result["parse_error"],
        }
        scored_rows.append(scored)
        counts[reference["question_type"]].append(int(label))
        if "_abs" in row["question_id"]:
            abstentions.append(int(label))

    values = [value for task in LONGMEMEVAL_QA_TYPES for value in counts[task]]
    overall = sum(values) / len(values) if values else 0.0
    abstention_score = sum(abstentions) / len(abstentions) if abstentions else 0.0

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as output:
        for row in scored_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    ensure_parent_dir(results_output)
    with open(results_output, "w", encoding="utf-8") as output:
        output.write(f"Judge Model: {model}\nResults Path: {preds_path}\nReference Path: {ref_path}\n")
        for task in LONGMEMEVAL_QA_TYPES:
            if counts[task]:
                output.write(
                    f"{task}: {sum(counts[task]) / len(counts[task]):.4f} "
                    f"({sum(counts[task])}/{len(counts[task])})\n"
                )
        output.write(
            f"Abstention Score: {abstention_score:.4f} "
            f"({sum(abstentions)}/{len(abstentions)})\n"
        )
        output.write(f"Correct Count: {sum(values)}\n")
        output.write(f"Total Count: {len(values)}\n")
        output.write(f"Score: {overall}\n--------------------\n")
    print(f"Judge Model: {model}")
    print(f"Correct Count: {sum(values)}")
    print(f"Total Count: {len(values)}")
    print(f"Overall Accuracy: {overall:.4f}")


BROWSECOMP_PLUS_GRADER_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


def _parse_browsecomp_judge(text: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not text:
        result["parse_error"] = True
        return result

    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", text, re.I | re.S
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", text, re.I | re.S
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)", text, re.I | re.S
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        text,
        re.I | re.S,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            text,
            re.I | re.S,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)", text, re.I | re.S
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", text, re.I)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", text, re.I)
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", text, re.I)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    confidence_match = re.search(
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", text, re.I
    )
    if not confidence_match:
        confidence_match = re.search(
            r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", text, re.I
        )
    if not confidence_match:
        confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", text, re.I)
    if confidence_match:
        result["confidence"] = min(100.0, float(confidence_match.group(1)))

    if result["correct"] is None:
        result["parse_error"] = True
    return result


def annotate_browsecomp_plus_file(
    preds_path: str,
    ref_path: str,
    output_path: str,
    results_output: str,
    endpoint_config: str,
    pred_key: str = "final_answer",
    judge_model: str = "endpoint-config",
    max_samples: int = None,
    max_judge_tokens: int = 4096,
    num_workers: int = 32,
):
    predictions = list(iter_jsonl(preds_path, cnt=max_samples))
    references = {
        str(row["query_id"]): (row["query"], row["answer"])
        for row in iter_jsonl(ref_path)
    }
    client, model, request_kwargs = _resolve_judge_client(
        judge_model, endpoint_config
    )
    selected = []
    prompts = []
    for row in predictions:
        query_id = str(row.get("query_id", ""))
        if query_id not in references:
            continue
        question, answer = references[query_id]
        prediction = row.get(pred_key, "")
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else ""
        prompts.append(BROWSECOMP_PLUS_GRADER_PROMPT.format(
            question=question, response=prediction, correct_answer=answer
        ))
        selected.append(row)

    responses = _parallel_judge(
        client,
        model,
        prompts,
        max_judge_tokens,
        num_workers,
        "Scoring BrowseComp+",
        request_kwargs,
    )
    scored_rows = []
    for row, prompt, response in zip(selected, prompts, responses):
        scored = dict(row)
        scored["autoeval_judge"] = {
            "model": model,
            "judge_prompt": prompt,
            "judge_response": response,
            **_parse_browsecomp_judge(response),
        }
        scored_rows.append(scored)
    valid = [row["autoeval_judge"] for row in scored_rows if not row["autoeval_judge"]["parse_error"]]
    correct = sum(bool(row["correct"]) for row in valid)
    score = correct / len(valid) if valid else 0.0

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as output:
        for row in scored_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    ensure_parent_dir(results_output)
    with open(results_output, "w", encoding="utf-8") as output:
        output.write(f"Judge Model: {model}\nResults Path: {preds_path}\nReference Path: {ref_path}\n")
        output.write(f"Correct Count: {correct}\nTotal Count: {len(valid)}\nScore: {score}\n--------------------\n")
    print(f"Correct Count: {correct}")
    print(f"Total Count: {len(valid)}")
    print(f"Score: {score}")


if __name__ == "__main__":
    fire.Fire({
        "score_choice_file": score_choice_file,
        "evaluate_choice_file": evaluate_choice_file,
        "annotate_longmemeval_file": annotate_longmemeval_file,
        "annotate_browsecomp_plus_file": annotate_browsecomp_plus_file,
    })
