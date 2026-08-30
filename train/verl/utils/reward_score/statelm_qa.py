"""
Reward function for ContextPilot QA training.

Terminal reward components implemented here:
- outcome reward R_out in {0, 1}
- format reward R_fmt in {-0.5, 0}

The execution/budget penalty R_pen in {-0.5, 0} is applied in the
agent-loop layer because it depends on trajectory-level tool execution and
context-budget signals.
"""

import json
import logging
import os
import re

from openai import OpenAI
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

logger = logging.getLogger(__name__)

LLM_JUDGE_PROMPT_TEMPLATE = """
Given a problem, its correct answer, and a student's answer below, your task is to review the student's answer and determine if it is correct by comparing it to the correct answer. If the student's answer is incomplete or ambiguous, assume it is incorrect.

{problem}

{answer}

{model_ans}

Please put your final answer (True or False) in \\boxed{{}}. Specifically, if the student's answer is correct, the final answer should be \\boxed{{True}}; otherwise, the final answer should be \\boxed{{False}}.
""".strip()


def has_valid_tool_call(solution_str: str) -> bool:
    """Check if the solution string contains a proper tool call."""
    tool_call_pattern = r'\<tool_call\>(.*?)</tool_call>'
    match = re.search(tool_call_pattern, solution_str, re.DOTALL)
    if match:
        try:
            tool_call = json.loads(match.group(1))
            return tool_call.get("name") == "finish" and "answer" in tool_call.get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            return False
    return False


def extract_tool_call_answer(solution_str: str) -> tuple[str, bool]:
    """Extract the answer from the tool call in the model output.
    A decoded solution string may look like this:

    assistant
    I find out who married Rosamund. Let me finish this question.
    <tool_call>
    {"name": "finish", "arguments": {"answer": "A"}}
    </tool_call>

    Returns:
        tuple: (extracted_answer, has_valid_tool_call)
    """
    last_assistant_idx = solution_str.rfind("assistant")
    if last_assistant_idx != -1:
        solution_str = solution_str[last_assistant_idx + len("assistant"):]
    
    if has_valid_tool_call(solution_str):
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(tool_call_pattern, solution_str, re.DOTALL)
        if match:
            try:
                tool_call = json.loads(match.group(1))
                final_answer = tool_call.get("arguments").get("answer")
                return final_answer, True
            except (json.JSONDecodeError, AttributeError):
                return "", False
    return "", False


def is_mcq(answer_str: str) -> bool:
    """Determine if the answer of MCQ format (i.e., ABCD)."""
    ans = answer_str.strip()
    return len(ans) == 1 and ans in "ABCD"


def count_sentences(text: str) -> int:
    """Count the number of sentences in text."""
    text = text.strip()
    if not text:
        return 0
    sentences = re.split(r'[.!?]+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def llm_judge_answer(prediction: str, ground_truth: str, question: str = None) -> bool:
    """Use OpenAI API to judge if the prediction is correct.
    
    Args:
        prediction: The predicted answer
        ground_truth: The correct answer
        question: Optional question text for context
        
    Returns:
        True if LLM judges the answer as correct, False otherwise
    """
    import time
    from openai import RateLimitError, APIConnectionError

    MAX_RETRIES = 3
    BASE_DELAY = 2
    
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"), 
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    model_name = os.environ.get("OPENAI_MODEL")
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": LLM_JUDGE_PROMPT_TEMPLATE.format(
                        problem=question, 
                        answer=ground_truth, 
                        model_ans=prediction
                    )}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            result = last_boxed_only_string(response.choices[0].message.content)
            if result is not None:
                correctness = remove_boxed(result).strip().lower() == "true"
                return correctness
            
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("LLM judge returned no boxed result; retrying in %d seconds", delay)
                time.sleep(delay)
            else:
                raise ValueError(f"No boxed result found after {MAX_RETRIES} attempts: {response.choices[0].message.content}")
                
        except (RateLimitError, APIConnectionError) as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("LLM judge API error; retrying in %d seconds: %s", delay, e)
                time.sleep(delay)
            else:
                logger.warning("LLM judge failed after %d attempts: %s", MAX_RETRIES, e)
                pred_normalized = prediction.lower().strip()
                gt_normalized = ground_truth.lower().strip()
                return pred_normalized == gt_normalized
                
        except Exception as e:
            logger.warning("LLM judge error: %s", e)
            pred_normalized = prediction.lower().strip()
            gt_normalized = ground_truth.lower().strip()
            return pred_normalized == gt_normalized


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """Compute the terminal QA score following the paper decomposition.
    
    Components implemented in this reward function:
    - R_out = 1 if the submitted answer matches the ground truth, else 0
    - R_fmt = -0.5 if any assistant turn emitted plain text instead of a valid tool call, else 0

    The trajectory-level R_pen term is applied by the agent loop, where tool
    execution errors and context-budget violations are available.
    
    Args:
        solution_str: The model's output string
        ground_truth: The correct answer (letter for MCQ, or text for Open-Ended)
        **kwargs: Additional keyword arguments:
            - question: Optional question text for LLM judge context
    
    Returns:
        float: R_out + R_fmt, in {-0.5, 0.0, 1.0}
    """
    predicted, has_valid_finish_tool_call = extract_tool_call_answer(solution_str)
    
    if not isinstance(predicted, str):
        predicted = str(predicted) if predicted else ""
    
    extra_info = kwargs.get("extra_info", {}) or {}
    messages = extra_info.get("raw_prompt", [])
    for msg in messages:
        if msg.get("role") == "user":
            question = msg.get("content")
            break
    else:
        question = ""
        logger.warning("No user prompt found; using an empty question for QA scoring")
    
    is_mcq_question = is_mcq(ground_truth)
    
    if is_mcq_question:
        is_correct = has_valid_finish_tool_call and (
            (predicted.strip() == ground_truth) or (predicted.strip().startswith(f"{ground_truth}."))
        )
    else:
        is_correct = has_valid_finish_tool_call and llm_judge_answer(predicted, ground_truth, question)

    had_format_violation = bool(extra_info.get("had_format_violation", not has_valid_finish_tool_call))
    r_out = 1.0 if is_correct else 0.0
    r_fmt = -0.5 if had_format_violation else 0.0
    score = r_out + r_fmt

    return score
