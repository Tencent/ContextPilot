"""ContextPilot advantage estimator.

Partial rollouts and subtree-mean reward assignment happen in the agent-loop
worker before this estimator runs.  Consequently, each input row already
represents one snapshot whose reward is the mean terminal reward of its
descendant continuations.  This module performs the remaining GRPO
group-relative normalisation across snapshots that share the same query uid.

The dedicated ``contextpilot_grpo`` estimator keeps that trainer contract
explicit while reusing vanilla GRPO's group-relative z-score normalisation.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config.algorithm import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("contextpilot_grpo")
def compute_contextpilot_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ContextPilot GRPO advantage.

    Behaviour: identical to vanilla GRPO group-relative z-score normalisation.
    Each row of ``token_level_rewards`` corresponds to a snapshot whose reward
    has already been assigned by the agent-loop worker using the terminal
    outcomes in that snapshot's subtree.  Grouping by query uid gives the
    paper's group-relative advantage over all retained snapshots for the same
    query.
    """
    norm_adv_by_std_in_grpo = True
    if config is not None:
        try:
            norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
        except Exception:
            norm_adv_by_std_in_grpo = True

    scores = token_level_rewards.sum(dim=-1)
    id2score: dict = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(stacked)
                id2std[idx] = torch.std(stacked)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages
