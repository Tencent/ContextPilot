#!/usr/bin/env python3
"""Hugging Face and local-dataset evaluation runner for ContextPilot."""

import importlib
import importlib.util
import hashlib
import json
import multiprocessing as mp
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Callable, Optional

import fire
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def _import_contextpilot(version: str):
    if version == "fsm":
        try:
            from infer.src.contextpilot import ExecLogger
            from infer.src.contextpilot_fsm import ContextPilotFSM
        except Exception as e:
            raise ImportError(
                "Could not import ContextPilotFSM. "
                f"Original error: {e}"
            ) from e
        return ContextPilotFSM, ExecLogger

    try:
        from infer.src.contextpilot import ContextPilot, ExecLogger
    except Exception as e:
        raise ImportError(
            "Could not import ContextPilot. "
            f"Original error: {e}"
        ) from e
    return ContextPilot, ExecLogger

def _load_callable(spec: str) -> Callable:
    """Load a callable from `module:function` or `/path/file.py:function`."""
    if ":" not in spec:
        raise ValueError(f"Function spec must be 'module_or_path:func', got: {spec}")
    mod_part, func_name = spec.split(":", 1)

    if mod_part.endswith(".py") or "/" in mod_part or mod_part.startswith("."):
        path = Path(mod_part).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Function file not found: {path}")
        module_name = path.stem + "_dyn"
        spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)

    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"'{func_name}' not found or not callable in {mod_part}")
    return fn

def read_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_local_json_data(file_path: str):
    if file_path.endswith(".json"):
        data = read_json(file_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in local dataset file: {file_path}")
        return data

    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    raise ValueError(f"Unsupported local dataset format: {file_path}")


def _derive_sampling_seed(base_seed: str, sample_id: str, mode: str) -> int:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "global":
        return int(base_seed) & 0x7FFFFFFF
    if normalized_mode == "sample_hash":
        material = f"{base_seed}:{sample_id}".encode("utf-8")
    elif normalized_mode == "constant_hash":
        material = str(base_seed).encode("utf-8")
    else:
        raise ValueError(
            "CONTEXTPILOT_SEED_MODE must be one of: global, sample_hash, "
            f"constant_hash; got {mode!r}"
        )
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF

def _collect_done_ids(result_dir: str, log_dir: str, shard_fp: str) -> set:
    """Collect completed sample IDs from results, shards, and traces."""
    done = set()

    if result_dir and os.path.isdir(result_dir):
        for fp in glob(os.path.join(result_dir, "*_final_result_*.json")):
            base = os.path.basename(fp)
            sid = base.split("_final_result_")[0]
            if sid:
                done.add(str(sid))

    if shard_fp and os.path.exists(shard_fp):
        with open(shard_fp, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("sample_id")
                if sid is not None:
                    done.add(str(sid))

    if log_dir and os.path.isdir(log_dir):
        for fp in glob(os.path.join(log_dir, "*_inference_result_*.json")):
            base = os.path.basename(fp)
            sid = base.split("_inference_result_")[0]
            if sid:
                done.add(str(sid))

    return done

def _worker_run(
    rank: int,
    n_proc: int,
    vllm_cfg_path: str,
    temperature: float,
    top_p: float,
    top_k: int,
    tool_config_path: Optional[str],
    system_prompt_name: Optional[str],
    max_turns_exp: int,
    max_context_exp: int,
    max_context: int,
    dataset_name: str,
    dataset_split: str,
    item_to_question: str,
    item_to_context: str,
    item_to_answer: str,
    item_to_meta: str,
    correct_answer_key: str,
    model_answer_key: str,
    trajectory_dir: str,
    result_dir: str,
    output_fp: str,
    tokenizer_path: str,
    max_turns_to_fail: int,
    quiet_progress: bool,
    version: str,
    max_items: int = None,
    resume: bool = False,
    max_output_tokens: int = 4096,
    mflow_project_dir: Optional[str] = None,
    enable_graph: bool = False,
    allow_text_tool_call_fallback: bool = False,
    embedding_cfg_path: Optional[str] = None,
    reranker_cfg_path: Optional[str] = None,
    use_reranker: bool = False,
    max_search_calls: int = None,
    max_no_tool_retries: int = None,
    max_token_window: int = None,
    chunk_size: int = None,
    overlap: int = None,
    boundary_backtrack: int = None,
    max_chunks_once: int = None,
    failed_samples_file: Optional[str] = None,
    retry_with_hint: bool = False,
    use_more_plan: bool = False,
    use_required_tool_choice: bool = True,
    load_document_truncate_side: str = "middle",
    highlight_fragment_size: int = None,
    highlight_num_fragments: int = None,
    highlight_no_match_size: int = None,
    search_engine_max_results: int = None,
    save_last_payload: bool = True,
    auto_delete_on_context_overflow: bool = True,
    work_counter=None,
    work_counter_lock=None,
) -> None:
    agent_cls, logger_cls = _import_contextpilot(version=version)

    vllm_cfg = read_json(vllm_cfg_path)
    vllm_cfg["_worker_rank"] = rank
    embedding_config = read_json(embedding_cfg_path) if embedding_cfg_path else None
    reranker_config = read_json(reranker_cfg_path) if reranker_cfg_path else None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if dataset_name.endswith((".json", ".jsonl")) and dataset_split == "local":
        data = read_local_json_data(dataset_name)
    else:
        data = load_dataset(dataset_name, split=dataset_split)

    if max_items is not None:
        if max_items <= 0:
            data = [] if isinstance(data, list) else data.select([])
        else:
            cap = min(max_items, len(data))
            data = data[:cap] if isinstance(data, list) else data.select(range(cap))

    item_to_question_fn = _load_callable(item_to_question)
    item_to_context_fn  = _load_callable(item_to_context)
    item_to_answer_fn   = _load_callable(item_to_answer)
    item_to_meta_fn     = _load_callable(item_to_meta) if item_to_meta else None

    failed_samples_map = {}
    if failed_samples_file and os.path.exists(failed_samples_file):
        failed_samples_map = read_json(failed_samples_file)
        print(f"[retry][rank {rank}] loaded {len(failed_samples_map)} failed samples from {failed_samples_file}")

    rank_suffix = f".rank{rank}"
    out_fp_rank = (
        output_fp.replace(".jsonl", f"{rank_suffix}.jsonl")
        if output_fp.endswith(".jsonl") else
        f"{output_fp}{rank_suffix}.jsonl"
    )

    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    out_dir = os.path.dirname(out_fp_rank)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done_ids = set()
    if resume:
        done_ids = _collect_done_ids(result_dir=result_dir, log_dir=trajectory_dir, shard_fp=out_fp_rank)
        print(f"[resume][rank {rank}] found {len(done_ids)} completed sample_id(s).")
    fout_mode = "a" if (resume and os.path.exists(out_fp_rank)) else "w"

    skipped = 0
    processed = 0
    with open(out_fp_rank, fout_mode, encoding="utf-8") as fout:
        if work_counter is not None and work_counter_lock is not None:
            def dynamic_indices():
                while True:
                    with work_counter_lock:
                        idx = int(work_counter.value)
                        work_counter.value += 1
                    if idx >= len(data):
                        return
                    yield idx
            assigned_indices = dynamic_indices()
            progress_total = None
            schedule_name = "queue"
        else:
            assigned_indices = range(rank, len(data), n_proc)
            progress_total = len(range(rank, len(data), n_proc))
            schedule_name = "stride"

        for idx in tqdm(assigned_indices,
                        total=progress_total,
                        desc=f"Rank {rank}",
                        position=rank,
                        disable=quiet_progress):
            sample = data[idx]

            if "_id" in sample:
                sample_id = str(sample["_id"])
            elif "id" in sample:
                sample_id = str(sample["id"])
            else:
                sample_id = str(idx)

            if resume and (sample_id in done_ids):
                skipped += 1
                continue

            if failed_samples_map and (sample_id not in failed_samples_map):
                skipped += 1
                continue

            context = item_to_context_fn(sample)
            question = item_to_question_fn(sample)
            correct_ans = item_to_answer_fn(sample)
            meta_dict = item_to_meta_fn(sample) if item_to_meta_fn else {}

            retry_hint = None
            if retry_with_hint and failed_samples_map and sample_id in failed_samples_map:
                wrong_ans = failed_samples_map[sample_id].get("wrong_answer", "")
                if wrong_ans:
                    retry_hint = f"\n\n[Note: A previous attempt answered \"{wrong_ans}\" which was incorrect. Please try a different approach and search more thoroughly.]"

            question_type = sample.get("question_type", "Long Context QA")

            logger = logger_cls(log_dir=trajectory_dir, results_dir=result_dir)
            delete_assistant_tool_call_only = version == "niah"
            print(f"[INFO] Setting delete_assistant_tool_call_only: {delete_assistant_tool_call_only}")
            agent = agent_cls(
                vllm_config=vllm_cfg,
                document_content=context,
                temperature=temperature,
                topp=top_p,
                topk=top_k,
                tool_config_path=tool_config_path,
                system_prompt_name=system_prompt_name,
                tokenizer=tokenizer,
                max_turns_exp=max_turns_exp,
                max_context_exp=max_context_exp,
                max_output_tokens=max_output_tokens,
                delete_assistant_tool_call_only=delete_assistant_tool_call_only,
                mflow_project_dir=mflow_project_dir,
                enable_graph=enable_graph,
                allow_text_tool_call_fallback=allow_text_tool_call_fallback,
                embedding_config=embedding_config,
                reranker_config=reranker_config,
                use_reranker=use_reranker,
            )
            deterministic_seed = os.getenv("CONTEXTPILOT_DETERMINISTIC_SEED")
            if deterministic_seed is not None:
                seed_mode = os.getenv("CONTEXTPILOT_SEED_MODE", "global")
                agent._sampling_seed_base = _derive_sampling_seed(
                    deterministic_seed, sample_id, seed_mode
                )
                agent._sampling_seed_mode = seed_mode.strip().lower()
            if chunk_size is not None:
                agent.tool_library._default_chunk_size = chunk_size
            if overlap is not None:
                agent.tool_library._default_overlap = overlap
            if boundary_backtrack is not None and boundary_backtrack >= 0:
                agent.tool_library._boundary_backtrack = int(boundary_backtrack)
            if max_context is not None and max_context > 0:
                agent.tool_library._max_context = max_context
            if load_document_truncate_side is not None:
                agent.tool_library._load_document_truncate_side = (
                    agent.tool_library._normalize_truncate_side(load_document_truncate_side)
                )
            if highlight_fragment_size is not None and int(highlight_fragment_size) > 0:
                agent.tool_library._default_highlight_fragment_size = int(highlight_fragment_size)
            if highlight_num_fragments is not None and int(highlight_num_fragments) > 0:
                agent.tool_library._default_highlight_num_fragments = int(highlight_num_fragments)
            if highlight_no_match_size is not None and int(highlight_no_match_size) >= 0:
                agent.tool_library._default_highlight_no_match_size = int(highlight_no_match_size)
            if search_engine_max_results is not None and int(search_engine_max_results) > 0:
                agent.tool_library._search_engine_max_results = int(search_engine_max_results)
            if max_search_calls is not None and max_search_calls > 0:
                agent.set_max_search_calls(max_search_calls)
            if max_no_tool_retries is not None and max_no_tool_retries >= 0:
                agent.set_max_no_tool_retries(max_no_tool_retries)
            if max_token_window is not None and max_token_window > 0:
                agent.set_max_token_window(max_token_window)
            if hasattr(agent, "set_auto_delete_on_context_overflow"):
                agent.set_auto_delete_on_context_overflow(bool(auto_delete_on_context_overflow))
            if max_chunks_once is not None and max_chunks_once > 0:
                agent.set_max_chunks_once(max_chunks_once)
            if hasattr(agent, "set_use_more_plan"):
                agent.set_use_more_plan(bool(use_more_plan))
            if hasattr(agent, "set_use_required_tool_choice"):
                agent.set_use_required_tool_choice(bool(use_required_tool_choice))
            if retry_hint:
                agent._retry_hint = retry_hint
            try:
                last_payload = agent.run(question, max_turns_to_fail=max_turns_to_fail)
            except Exception as e:
                print(f"[ERROR][rank {rank}] sample {idx}: {e}")
                last_payload = {"error": str(e)}

            stored_last_payload = last_payload if save_last_payload else None
            meta_info = {
                "question_type": question_type,
                "sample_id": sample_id,
                "message_count": getattr(agent, "ctx_counter", None),
                "notes_count": len(getattr(agent.state_manager, "notes", []))
                               if hasattr(agent, "state_manager") else None,
                "last_payload": stored_last_payload,
            }
            res_file, final_answer = logger.save_final_result(agent, question, correct_ans, meta_info)

            result_info = {
                "question_type": question_type,
                correct_answer_key: correct_ans,
                model_answer_key: final_answer,
                "message_count": getattr(agent, "ctx_counter", None),
                "notes_count": len(getattr(agent.state_manager, "notes", []))
                               if hasattr(agent, "state_manager") else None,
                "last_payload": stored_last_payload,
            }
            inf_file = logger.save_inference_result(question, agent, result_info, prefix_tag=sample_id)

            result = {
                "dataset": dataset_name,
                "split": dataset_split,
                "model": "ContextPilot",
                "sample_id": sample_id,
                "question": question,
                "question_type": question_type,
                correct_answer_key: correct_ans,
                model_answer_key: final_answer,
                "meta_info": {
                    "api_call_count": getattr(agent, "api_call_counter", 0),
                    "rank": rank,
                    "n_proc": n_proc,
                    "inference_result_path": inf_file,
                    "final_result_path": res_file,
                    "max_turns_to_fail": max_turns_to_fail,
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "sampling_seed_base": getattr(agent, "_sampling_seed_base", None),
                    "sampling_seed_mode": getattr(agent, "_sampling_seed_mode", None),
                    "system_prompt_name": system_prompt_name,
                    "timestamp": datetime.now().isoformat()
                }
            }

            if meta_dict:
                result.update(meta_dict)
                if "task_name" in meta_dict:
                    result['others'] = meta_dict
                    result['input'] = question

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

    print(
        f"[rank {rank}] wrote {out_fp_rank} "
        f"(processed={processed}, skipped={skipped}, scheduling={schedule_name})"
    )

def eval_hfds_contextpilot(
    vllm_cfg: str,
    temperature: float,
    max_turns_exp: int,
    max_context_exp: int,
    max_context: int,
    tool_config_path: Optional[str],
    system_prompt_name: Optional[str],
    dataset_name: str,
    dataset_split: str,
    item_to_question: str,
    item_to_context: str,
    item_to_answer: str,
    trajectory_dir: str,
    result_dir: str,
    output_fp: str,
    tokenizer_path: str = "Qwen/Qwen3-8B",
    max_turns_to_fail: int = 80,
    top_p: float = 1.0,
    top_k: int = None,
    max_output_tokens: int = 4096,
    item_to_meta: str = None,
    output_postprocess: str = None,
    model_answer_key: str = 'final_answer',
    correct_answer_key: str = 'correct_answer',
    n_proc: int = 1,
    max_items: int = None,
    resume: bool = True,
    merge_after: bool = True,
    version: str = "v4",
    mflow_project_dir: str = None,
    enable_graph: bool = False,
    allow_text_tool_call_fallback: bool = False,
    embedding_cfg: str = None,
    reranker_cfg: str = None,
    use_reranker: bool = False,
    max_search_calls: int = None,
    max_no_tool_retries: int = None,
    max_token_window: int = None,
    chunk_size: int = None,
    overlap: int = None,
    boundary_backtrack: int = None,
    max_chunks_once: int = None,
    failed_samples_file: str = None,
    retry_with_hint: bool = False,
    use_more_plan: bool = False,
    use_required_tool_choice: bool = True,
    load_document_truncate_side: str = "middle",
    highlight_fragment_size: int = None,
    highlight_num_fragments: int = None,
    highlight_no_match_size: int = None,
    search_engine_max_results: int = None,
    save_last_payload: bool = True,
    auto_delete_on_context_overflow: bool = True,
    scheduling: str = "stride",
) -> None:

    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    if output_fp:
        out_dir = os.path.dirname(output_fp)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    resume_boolean = str(resume).lower() == 'true'
    enable_graph_boolean = str(enable_graph).lower() == 'true'
    allow_text_tool_call_fallback_boolean = str(allow_text_tool_call_fallback).lower() == 'true'
    use_reranker_boolean = str(use_reranker).lower() == 'true'
    use_more_plan_boolean = str(use_more_plan).lower() == 'true'
    use_required_tool_choice_boolean = str(use_required_tool_choice).lower() == 'true'
    retry_with_hint_boolean = str(retry_with_hint).lower() == 'true'
    save_last_payload_boolean = str(save_last_payload).lower() == 'true'
    auto_delete_on_context_overflow_boolean = str(auto_delete_on_context_overflow).lower() == 'true'
    scheduling = str(scheduling).strip().lower()
    if scheduling not in {"stride", "queue"}:
        raise ValueError(f"scheduling must be 'stride' or 'queue', got {scheduling!r}")
    if n_proc <= 1:
        _worker_run(
            rank=0,
            n_proc=1,
            vllm_cfg_path=vllm_cfg,
            temperature=temperature,
            max_turns_exp=max_turns_exp,
            max_context_exp=max_context_exp,
            max_context=max_context,
            tool_config_path=tool_config_path,
            system_prompt_name=system_prompt_name,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            item_to_question=item_to_question,
            item_to_context=item_to_context,
            item_to_answer=item_to_answer,
            item_to_meta=item_to_meta,
            correct_answer_key=correct_answer_key,
            model_answer_key=model_answer_key,
            trajectory_dir=trajectory_dir,
            result_dir=result_dir,
            output_fp=output_fp,
            tokenizer_path=tokenizer_path,
            max_turns_to_fail=max_turns_to_fail,
            quiet_progress=False,
            max_items=max_items,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            resume=resume_boolean,
            version=version,
            mflow_project_dir=mflow_project_dir,
            enable_graph=enable_graph_boolean,
            allow_text_tool_call_fallback=allow_text_tool_call_fallback_boolean,
            embedding_cfg_path=embedding_cfg,
            reranker_cfg_path=reranker_cfg,
            use_reranker=use_reranker_boolean,
            max_search_calls=max_search_calls,
            max_no_tool_retries=max_no_tool_retries,
            max_token_window=max_token_window,
            chunk_size=chunk_size,
            overlap=overlap,
            boundary_backtrack=boundary_backtrack,
            max_chunks_once=max_chunks_once,
            failed_samples_file=failed_samples_file,
            retry_with_hint=retry_with_hint_boolean,
            use_more_plan=use_more_plan_boolean,
            use_required_tool_choice=use_required_tool_choice_boolean,
            load_document_truncate_side=load_document_truncate_side,
            highlight_fragment_size=highlight_fragment_size,
            highlight_num_fragments=highlight_num_fragments,
            highlight_no_match_size=highlight_no_match_size,
            search_engine_max_results=search_engine_max_results,
            save_last_payload=save_last_payload_boolean,
            auto_delete_on_context_overflow=auto_delete_on_context_overflow_boolean,
        )
        if merge_after and output_fp.endswith(".jsonl"):
            shard0 = output_fp.replace(".jsonl", ".rank0.jsonl")
            if os.path.exists(shard0) and shard0 != output_fp:
                os.replace(shard0, output_fp)
                print(f"[merge] single shard → {output_fp}")
        print(f"Final output saved to {output_fp}")

        if output_postprocess:
            output_postprocess_fn = _load_callable(output_postprocess)
            output_postprocess_fn(output_fp)
        return

    mp.set_start_method("spawn", force=True)
    work_counter = mp.Value("q", 0) if scheduling == "queue" else None
    work_counter_lock = mp.Lock() if scheduling == "queue" else None
    print(f"[scheduler] mode={scheduling}, workers={n_proc}")
    procs = []
    for r in range(n_proc):
        p = mp.Process(
            target=_worker_run,
            kwargs=dict(
                rank=r,
                n_proc=n_proc,
                vllm_cfg_path=vllm_cfg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                max_turns_exp=max_turns_exp,
                max_context_exp=max_context_exp,
                max_context=max_context,
                tool_config_path=tool_config_path,
                system_prompt_name=system_prompt_name,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                item_to_question=item_to_question,
                item_to_context=item_to_context,
                item_to_answer=item_to_answer,
                item_to_meta=item_to_meta,
                correct_answer_key=correct_answer_key,
                model_answer_key=model_answer_key,
                trajectory_dir=trajectory_dir,
                result_dir=result_dir,
                output_fp=output_fp,
                tokenizer_path=tokenizer_path,
                max_turns_to_fail=max_turns_to_fail,
                quiet_progress=(n_proc > 8),
                max_items=max_items,
                resume=resume_boolean,
                version=version,
                mflow_project_dir=mflow_project_dir,
                enable_graph=enable_graph_boolean,
                allow_text_tool_call_fallback=allow_text_tool_call_fallback_boolean,
                embedding_cfg_path=embedding_cfg,
                reranker_cfg_path=reranker_cfg,
                use_reranker=use_reranker_boolean,
                max_search_calls=max_search_calls,
                max_no_tool_retries=max_no_tool_retries,
                max_token_window=max_token_window,
                chunk_size=chunk_size,
                overlap=overlap,
                boundary_backtrack=boundary_backtrack,
                max_chunks_once=max_chunks_once,
                failed_samples_file=failed_samples_file,
                retry_with_hint=retry_with_hint_boolean,
                use_more_plan=use_more_plan_boolean,
                use_required_tool_choice=use_required_tool_choice_boolean,
                load_document_truncate_side=load_document_truncate_side,
                highlight_fragment_size=highlight_fragment_size,
                highlight_num_fragments=highlight_num_fragments,
                highlight_no_match_size=highlight_no_match_size,
                search_engine_max_results=search_engine_max_results,
                save_last_payload=save_last_payload_boolean,
                auto_delete_on_context_overflow=auto_delete_on_context_overflow_boolean,
                work_counter=work_counter,
                work_counter_lock=work_counter_lock,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed_exit_codes = [p.exitcode for p in procs if p.exitcode != 0]
    if failed_exit_codes:
        raise RuntimeError(
            f"Inference workers failed with exit codes: {failed_exit_codes}"
        )

    if merge_after and output_fp.endswith(".jsonl"):
        shards = []
        for r in range(n_proc):
            shard = output_fp.replace(".jsonl", f".rank{r}.jsonl")
            if os.path.exists(shard):
                shards.append(shard)
        if shards:
            results = []
            for shard in shards:
                with open(shard, "r", encoding="utf-8") as fin:
                    for line in fin:
                        results.append(json.loads(line))
            results.sort(key=lambda x: x.get("sample_id", -1))
            with open(output_fp, "w", encoding="utf-8") as fout:
                for r in results:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[merge] merged {len(shards)} shard(s) → {output_fp} (sorted by sample_id)")
            for shard in shards:
                os.remove(shard)
                print(f"[cleanup] deleted {shard}")
        else:
            print("[merge] no shard files found; nothing to merge.")

    if output_postprocess:
        output_postprocess_fn = _load_callable(output_postprocess)
        output_postprocess_fn(output_fp)

if __name__ == "__main__":
    fire.Fire()
