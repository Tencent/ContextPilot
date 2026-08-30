"""System prompt used by the four-domain FSM evaluator."""

from pathlib import Path


_PROMPT_PATH = Path(__file__).with_name("fsm_plan_bm25_mc_prompt.txt")
FSM_PLAN_BM25_MC_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()
