"""Search-R1 retrieval environment (Ray actor).

Implements `EnvironmentInterface.step()` for the multi-turn rollout loop in
`run_multi_turn_rollout` (rollouts.py:403). Each turn:

1. Look at the latest assistant message in each sample's message_log.
2. If it contains `<answer>...</answer>` → terminal: compute reward against
   `golden_answers` (carried in metadata as `ground_truth`), emit empty obs.
3. Else if it contains a search query (`<tool_call><function=search>
   <parameter=query>...` for `qwen_native`; `<search>...</search>` for `paper`)
   → POST to the local retriever, format docs as a tool/observation message,
   continue.
4. Else if turn_count ≥ max_turns → terminate with reward=0.
5. Else → no tool call and no answer in this turn; we don't have the model's
   output yet so we let the rollout continue (turn_count++) and the next
   `env.step` decides.

Retrieval is batched: all samples that emitted a search query in this turn go
to the retriever in one HTTP call (via the `query: list[str]` shape of
`/batch_search` — VERL_REFERENCE.md §1, retriever_serving.py:111).

The observation `content` is appended raw to the message log without re-running
the chat template (rollouts.py:498-501). For `paper` we emit
`<information>{docs}</information>` (mirrors `CodeEnvironment.format_result`'s
`<result>...</result>` raw-text pattern). For `qwen_native` we hand-construct
the Qwen3.5 chat-template markers: `<|im_end|>\n<|im_start|>user\n
<tool_response>...\n</tool_response><|im_end|>\n<|im_start|>assistant\n` so
the next-turn generation context is well-formed.

Reward uses the byte-identical M1 scorer at `training/src/rewards/search_r1.py`.
For the qwen_native arm, the paper-tag-keyed `is_valid_sequence` /
`is_retrieval_correct` checks always fail (different tag set), so the reward
collapses to EM-only via the `score - structure_format_score` branch — that's
the M2 baseline by design.
"""
from __future__ import annotations

import re
from typing import Any, Optional, TypedDict

import ray
import requests
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from training.src.rewards.search_r1 import (
    compute_search_r1_reward,
    em_check,
    extract_solution,
)


VALID_ARMS = ("qwen_native", "paper")


class SearchR1EnvConfig(TypedDict, total=False):
    arm: str
    retriever_url: str
    top_n: int
    max_turns: int
    request_timeout_s: float


# Regexes compiled once. DOTALL so newlines inside content blocks match `.`.
_RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_RE_PAPER_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL)
# Permissive parser for Qwen3.5 tool calls — only the parameter matters.
_RE_QWEN_QUERY = re.compile(
    r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
    re.DOTALL,
)


def _parse_query(arm: str, assistant_text: str) -> Optional[str]:
    """Pull the latest search query out of an assistant message; None if absent."""
    if arm == "qwen_native":
        m = _RE_QWEN_QUERY.search(assistant_text)
    elif arm == "paper":
        m = _RE_PAPER_SEARCH.search(assistant_text)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if not m:
        return None
    q = m.group(1).strip()
    return q or None


def _format_docs_paper(docs: list[str]) -> str:
    """Wrap retrieved docs in the paper's `<information>` envelope."""
    body = "\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    return f"\n<information>\n{body}\n</information>\n"


def _format_docs_qwen_native(docs: list[str]) -> str:
    """Hand-craft the Qwen3.5 chat-template markers for a tool response.

    The previous assistant turn ended at `</tool_call>` (vLLM stop string),
    no `<|im_end|>` was emitted. We close the assistant turn, open a synthetic
    user turn carrying `<tool_response>`, close it, and re-open the assistant
    so the next generation continues in-distribution.
    """
    body = "\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        f"{body}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _retriever_failed_message(arm: str, reason: str) -> str:
    """Wrap a retrieval failure as a tool/info response so the LM can recover."""
    msg = f"Retriever failed: {reason}"
    if arm == "qwen_native":
        return _format_docs_qwen_native([msg])
    return _format_docs_paper([msg])


@ray.remote  # pragma: no cover
class SearchR1Environment(EnvironmentInterface):
    """Multi-turn search-augmented env for Search-R1 GRPO."""

    def __init__(self, cfg: SearchR1EnvConfig) -> None:
        arm = cfg.get("arm", "qwen_native")
        if arm not in VALID_ARMS:
            raise ValueError(f"cfg.arm must be in {VALID_ARMS}, got {arm!r}")
        self.arm = arm
        self.retriever_url = cfg.get("retriever_url", "http://127.0.0.1:3005").rstrip("/")
        self.top_n = int(cfg.get("top_n", 3))
        self.max_turns = int(cfg.get("max_turns", 4))
        self.request_timeout_s = float(cfg.get("request_timeout_s", 30.0))

    # ----- retrieval -----

    def _batch_retrieve(self, queries: list[Optional[str]]) -> list[Optional[list[str]]]:
        """POST {url}/batch_search for the non-None queries; return per-input docs.

        None entries (no query this turn) get None docs, preserving order.
        """
        idx_with_q = [(i, q) for i, q in enumerate(queries) if q]
        if not idx_with_q:
            return [None] * len(queries)

        valid_queries = [q for _, q in idx_with_q]
        try:
            response = requests.post(
                f"{self.retriever_url}/batch_search",
                json={"query": valid_queries, "top_n": self.top_n, "return_score": False},
                timeout=self.request_timeout_s,
            )
            response.raise_for_status()
            docs_batch = response.json()
        except requests.RequestException as e:
            # Graceful: every active sample gets the failure string back.
            err = str(e)
            return [
                [f"Retriever failed: {err}"] if i in {idx for idx, _ in idx_with_q} else None
                for i in range(len(queries))
            ]

        out: list[Optional[list[str]]] = [None] * len(queries)
        for (orig_idx, _), docs in zip(idx_with_q, docs_batch):
            out[orig_idx] = [d["contents"] for d in docs]
        return out

    # ----- step -----

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[dict[str, Any]],
    ) -> EnvironmentReturn:
        """One turn of the rollout for a batch of trajectories."""
        batch_size = len(message_log_batch)

        # Per-sample classification: "answer", "search", or "exhausted".
        kinds: list[str] = []
        queries: list[Optional[str]] = []
        extracted_answers: list[Optional[str]] = []

        for log, meta in zip(message_log_batch, metadata_batch):
            assistant_text = log[-1]["content"]  # latest message — the assistant just spoke

            # 1. Answer takes precedence (terminal).
            if (ans := extract_solution(assistant_text)) is not None:
                kinds.append("answer")
                queries.append(None)
                extracted_answers.append(ans)
                continue

            # 2. Search query — but only if we still have turns left.
            turn = int(meta.get("turn_count", 0))
            if turn + 1 >= self.max_turns:
                kinds.append("exhausted")
                queries.append(None)
                extracted_answers.append(None)
                continue

            q = _parse_query(self.arm, assistant_text)
            if q is None:
                # Neither answer nor search — model output is malformed.
                # Treat as exhausted (no good observation to feed back).
                kinds.append("exhausted")
                queries.append(None)
                extracted_answers.append(None)
                continue

            kinds.append("search")
            queries.append(q)
            extracted_answers.append(None)

        # 3. Single batched HTTP call for all samples that asked to search.
        retrieved = self._batch_retrieve(queries)

        # 4. Assemble per-sample outputs.
        observations: list[dict[str, str]] = []
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        terminateds = torch.zeros(batch_size, dtype=torch.bool)
        next_stop_strings: list[Optional[list[str]]] = []
        new_metadata: list[dict[str, Any]] = []
        answers_out: list[Optional[str]] = []

        for i, (log, meta, kind) in enumerate(zip(message_log_batch, metadata_batch, kinds)):
            updated_meta = {**meta, "turn_count": int(meta.get("turn_count", 0)) + 1}

            if kind == "answer":
                # Reconstruct the full solution string for the reward function.
                # Skip the initial user prompt; include all assistant + env turns.
                solution_str = "".join(
                    m["content"] for m in log if m["role"] != "user"
                )
                gold = meta.get("ground_truth") or []
                reward_info = compute_search_r1_reward(solution_str, gold)
                rewards[i] = float(reward_info["reward"])
                terminateds[i] = True
                observations.append({"role": "tool", "content": ""})
                next_stop_strings.append(None)
                answers_out.append(extracted_answers[i])

            elif kind == "exhausted":
                # No search, no answer (or out of turns). Score whatever we have.
                solution_str = "".join(
                    m["content"] for m in log if m["role"] != "user"
                )
                gold = meta.get("ground_truth") or []
                # Best-effort: maybe an unclosed <answer was emitted; em_check
                # against an empty string yields 0, which is what we want.
                ans = extract_solution(solution_str) or ""
                rewards[i] = float(em_check(ans, gold)) if ans else 0.0
                terminateds[i] = True
                observations.append({"role": "tool", "content": ""})
                next_stop_strings.append(None)
                answers_out.append(ans or None)

            else:  # "search"
                docs = retrieved[i] or [f"Retriever returned no documents for query: {queries[i]}"]
                content = (
                    _format_docs_qwen_native(docs)
                    if self.arm == "qwen_native"
                    else _format_docs_paper(docs)
                )
                observations.append({"role": "tool", "content": content})
                next_stop_strings.append(self._stop_strings_for_arm())
                answers_out.append(None)
                # rewards[i], terminateds[i] stay 0 / False

            new_metadata.append(updated_meta)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
            answers=answers_out,
        )

    def _stop_strings_for_arm(self) -> list[str]:
        if self.arm == "qwen_native":
            return ["</tool_call>", "</answer>"]
        return ["</search>", "</answer>"]

    def shutdown(self) -> None:
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        # Surface a few rollout-shape metrics. The rollout loop already tracks
        # turns/tokens in `sample_turn_counts`, so we keep this minimal.
        return batch, {}
