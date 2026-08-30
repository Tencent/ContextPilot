"""
ContextPilot RL components for verl.

This package contains the trainer-facing additions to verl for the
ContextPilot training scheme described in the paper.

The current pipeline:
    * segments trajectories into trainable snapshots at configured ContextPilot
      boundaries and applies token-level loss masking in ``statelm_agent_loop``
    * ranks context-management actions by sensitivity and performs partial
      rollouts from the selected parent states under a query-level budget
    * assigns each snapshot the mean terminal reward of its descendant
      continuations before query-level sampling and advantage computation
    * applies R_pen (-0.5) to terminal rewards when a tool call fails or the
      trajectory exceeds its context/turn budget
    * computes group-relative GRPO advantages across snapshots that share the
      same query uid via the ``contextpilot_grpo`` estimator

Partial rollout and subtree-reward assignment are implemented by the agent-loop
worker.  The advantage estimator in this package consumes those pre-assigned
snapshot rewards and performs the final query-group normalisation.

The whole stack is gated by the config flag
``actor_rollout_ref.rollout.multi_turn.contextpilot.enable``; when this flag
is False the behaviour is identical to vanilla verl/StateLM.
"""

from . import adv  # noqa: F401
from . import constants  # noqa: F401

__all__ = ["adv", "constants"]
