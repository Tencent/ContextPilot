"""
Reward function for Multiple Choice Questions (MCQ).
Extracts the answer letter (A, B, C, D) from model output and compares with ground truth.
"""
import re
import json

def extract_tool_call_answer(solution_str: str) -> str:
    """Extract the answer from the tool call in the model output.
    A decoded solution string may look like this:

    assistant
    I find out who married Rosamund. Let me finish this question.
    <tool_call>
    {"name": "finish", "arguments": {"answer": "A"}}
    </tool_call>

    """
    last_assistant_idx = solution_str.rfind("assistant")
    if last_assistant_idx != -1:
        solution_str = solution_str[last_assistant_idx + len("assistant "):]
    else:
        return solution_str
    
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(tool_call_pattern, solution_str)
    if match:
        tool_call = json.loads(match.group(1))
        if tool_call.get("name") == "finish" and tool_call.get("arguments").get("answer"):
            solution_str = tool_call.get("arguments").get("answer")
     
    return solution_str


def match_mcq_answer(pred: str, label: str) -> bool:
    pred = pred.strip()
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"

    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred in label:
            return True
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False

def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """Compute the score for a multiple choice question.
    
    Args:
        solution_str: The model's output string
        ground_truth: The correct answer letter (A, B, C, or D)
        **kwargs: Additional keyword arguments (ignored, for compatibility with reward framework)
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    predicted = extract_tool_call_answer(solution_str)
    return 1.0 if match_mcq_answer(predicted, ground_truth) else 0.0
