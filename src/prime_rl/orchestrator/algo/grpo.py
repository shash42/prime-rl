from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty
        self.advantage_estimator = config.advantage_estimator
        self.format_invalid_advantage = config.format_invalid_advantage
        self.format_invalid_reward_penalty = config.format_invalid_reward_penalty
        self.format_invalid_reward_min = config.format_invalid_reward_min
        if self.format_invalid_advantage is not None and self.format_invalid_reward_penalty is not None:
            raise ValueError("format_invalid_advantage and format_invalid_reward_penalty are mutually exclusive")

    async def score_group(self, group: list[Rollout]) -> None:
        if self.format_invalid_reward_penalty is not None:
            valid_rewards = [
                rollout.reward for rollout in group
                if rollout.metrics.get("format_valid") != 0.0
            ]
            invalid_reward = min(valid_rewards) - self.format_invalid_reward_penalty if valid_rewards else 0.0
            if self.format_invalid_reward_min is not None:
                invalid_reward = max(self.format_invalid_reward_min, invalid_reward)
            for rollout in group:
                if rollout.metrics.get("format_valid") == 0.0:
                    rollout.rewards["format_invalid_reward"] = invalid_reward - rollout.reward
                    rollout.info["format_invalid_reward"] = invalid_reward
                    rollout.info["format_invalid_reward_penalty"] = self.format_invalid_reward_penalty

        invalid_format = [
            self.format_invalid_advantage is not None and rollout.metrics.get("format_valid") == 0.0
            for rollout in group
        ]
        scored_group = [rollout for rollout, invalid in zip(group, invalid_format, strict=True) if not invalid]
        rewards = torch.tensor([rollout.reward for rollout in scored_group], dtype=torch.float32)
        length_penalty = self.length_penalty
        if scored_group and length_penalty is not None:
            output = torch.tensor([rollout.num_output_tokens for rollout in scored_group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in scored_group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in scored_group], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            rewards = rewards - penalty
        if scored_group and self.advantage_estimator == "tailrl":
            sorted_rewards, order = rewards.sort()
            gaps = torch.diff(sorted_rewards, prepend=rewards.new_zeros(1))
            n = len(scored_group)
            survivors = torch.arange(n, 0, -1, dtype=rewards.dtype)
            weights = n * (gaps / survivors).cumsum(0)
            rewards = torch.empty_like(rewards).scatter_(0, order, weights)
        advantages = rewards - rewards.mean() if scored_group else rewards
        valid_advantages = iter(advantages.tolist())
        for rollout, invalid in zip(group, invalid_format, strict=True):
            if invalid:
                assert self.format_invalid_advantage is not None
                advantage = self.format_invalid_advantage
            else:
                advantage = next(valid_advantages)
            rollout.assign_advantages(advantage)
