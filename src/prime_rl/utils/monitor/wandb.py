from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from transformers.tokenization_utils import PreTrainedTokenizer
from wandb.errors import CommError
from wandb.sdk.mailbox.mailbox_handle import ServerResponseError
from wandb_gql import gql

from prime_rl.configs.shared import WandbConfig, WandbWithExtrasConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor, sample_items_for_logging

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


def _details_routing_enabled() -> bool:
    return os.environ.get("PRIME_WANDB_DETAILS") == "1"


def route_default_workspace_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep a small mean-only surface and put native PRIME metrics under details/."""
    routed = {(key if key == "step" else f"details/{key}"): value for key, value in metrics.items()}
    aliases = {
        "train/agg/effective/reward/mean": "train/reward",
        "entropy/all/mean": "train/entropy",
        "mismatch_kl/all/mean": "train/sample_policy_kl",
        "optim/lr": "train/lr",
        "loss/mean": "train/loss",
        "optim/grad_norm": "train/grad_norm",
        "train/agg/effective/num_output_tokens/mean": "train/mean_total_output_tokens",
        "train/agg/effective/metrics/answer_tokens/mean": "train/mean_answer_tokens",
        "train/agg/all/metrics/relppl/mean": "train/relppl",
        "train/agg/all/is_truncated/mean": "train/truncation_rate",
        "train/agg/all/has_error/mean": "train/error_rate",
    }
    for source, target in aliases.items():
        if source in metrics:
            routed[target] = metrics[source]

    eval_metrics = {
        "effective/metrics/relppl/mean": "relppl",
        "effective/num_output_tokens/mean": "mean_total_output_tokens",
        "effective/metrics/answer_tokens/mean": "mean_answer_tokens",
        "all/is_truncated/mean": "truncation_rate",
        "all/has_error/mean": "error_rate",
    }
    for key, value in metrics.items():
        match = re.fullmatch(r"eval/(pasttest|futuretest)/(.+)", key)
        if match and match.group(2) in eval_metrics:
            routed[f"eval/{match.group(1)}/{eval_metrics[match.group(2)]}"] = value
    return routed


def _loggable_task(task) -> str:
    """A Table-safe JSON string of the task for sample logging. Image content parts are elided to
    a short placeholder — their base64 data bloats the table and breaks wandb Table's nested-type
    inference on the variable-length content list (a plain dict would otherwise crash on it)."""

    def elide(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "image_url":
                return {"type": "image_url", "image_url": "<image>"}
            return {k: elide(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [elide(v) for v in obj]
        return obj

    return json.dumps(elide(task.model_dump(mode="json")))


class WandbMonitor(Monitor):
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbConfig | WandbWithExtrasConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseConfig | None = None,
        keep_full_history: bool = True,
        train_env_names: list[str] = [],
        eval_env_names: list[str] = [],
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self._keep_full_history = keep_full_history
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0

        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        assert config is not None
        self._maybe_overwrite_wandb_command()

        # WANDB_MODE=disabled/offline takes precedence over shared mode — shared mode
        # requires a server connection and can't work offline.
        _wandb_mode = os.environ.get("WANDB_MODE")
        shared_mode = os.environ.get("WANDB_SHARED_MODE") == "1" and _wandb_mode not in ("disabled", "offline")
        if shared_mode:
            run_id = os.environ.get("WANDB_SHARED_RUN_ID")
            label = os.environ.get("WANDB_SHARED_LABEL")
            primary = label == "orchestrator"
            settings = wandb.Settings(
                mode="shared",
                x_label=label,
                x_primary=primary,
                x_update_finish_state=primary,
            )
            self.logger.info(f"Using shared W&B mode ({label=}, {primary=})")
            is_online = True
        else:
            run_id = None
            primary = False
            mode = os.environ.get("WANDB_MODE", "offline" if config.offline else "online")
            settings = wandb.Settings(mode=mode)
            is_online = mode == "online"

        retryable_errors = (CommError, ServerResponseError) if shared_mode else (CommError,)

        def init_wandb(max_retries: int):
            for attempt in range(max_retries):
                try:
                    return wandb.init(
                        id=run_id,
                        resume="allow" if run_id else None,
                        project=config.project,
                        entity=config.entity,
                        name=config.name,
                        group=config.group,
                        tags=config.tags,
                        dir=output_dir,
                        config=run_config.model_dump() if run_config else None,
                        settings=settings,
                    )
                except retryable_errors as e:
                    if attempt + 1 == max_retries:
                        raise
                    if shared_mode and not primary:
                        msg = (
                            f"Shared W&B run not yet created by primary - retrying in 10s ({attempt + 1}/{max_retries})"
                        )
                    else:
                        msg = f"Transient W&B init error ({e}) - retrying in 10s ({attempt + 1}/{max_retries})"
                    self.logger.info(msg)
                    # A failed wandb.init leaves the run_id registered in the local
                    # wandb-core StreamMux, causing the next attempt to fail with
                    # "run ID ... is in use". Tear down the service so the retry
                    # starts from a clean state.
                    wandb.teardown()
                    time.sleep(10)

        # Non-primary processes in shared mode wait for the primary to create the run.
        # Everyone else still retries to absorb transient W&B server errors (e.g. 404 on upsertBucket).
        max_retries = 30 if shared_mode and not primary else 5
        self.wandb = init_wandb(max_retries)

        wandb.define_metric("*", step_metric="step")

        # Provision the curated "overview" saved view once per project (the run's primary process
        # in shared mode, else the single master). Best-effort: a workspaces/API failure must never
        # take down training.
        if is_online and (primary if shared_mode else True):
            try:
                url = ensure_overview_view(
                    self.wandb.entity,
                    self.wandb.project,
                    train_envs=train_env_names,
                    eval_envs=eval_env_names,
                )
                if url:
                    self.logger.info(f"Updated W&B overview view - {url}")
            except Exception as e:
                self.logger.warning(f"Failed to create W&B overview view - {e}")

        # Optionally, initialize sample logging attributes
        if config is not None and isinstance(config, WandbWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples_cols = ["step", "env_name", "task", "task_idx", "messages", "input_ids", "reward"]
                self.samples_table = wandb.Table(
                    columns=self.samples_cols,
                    log_mode="INCREMENTAL",
                )
                self.tokenizer = tokenizer
                self.eval_samples_cols = ["step", "env", "task", "task_idx", "completion", "reward"]
                self.eval_samples_table = wandb.Table(
                    columns=self.eval_samples_cols,
                    log_mode="INCREMENTAL",
                )

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._keep_full_history:
            self.history.append(metrics)
        else:
            self.history = [metrics]
        if not self.is_master:
            return
        if not self.enabled:
            return
        if _details_routing_enabled():
            metrics = route_default_workspace_metrics(metrics)
        wandb.log({**metrics, "step": step})

    def log_samples(self, rollouts: list[Rollout], step: int) -> None:
        """Logs rollouts to W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log samples if not enabled or not log interval step
            return

        rollouts = sample_items_for_logging(
            rollouts,
            self.config.log_extras.sample_ratio,
        )
        if not rollouts:
            return

        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging {len(rollouts)} samples to W&B table at step {step}")
        start_time = time.perf_counter()

        for rollout in rollouts:
            trace = rollout
            for branch in trace.branches:
                token_ids = branch.token_ids
                if not token_ids:
                    continue
                sample = {
                    "step": step,
                    "env_name": rollout.env_name,
                    "task": _loggable_task(trace.task.data),
                    "task_idx": trace.task.data.idx,
                    "messages": self.tokenizer.decode(token_ids),
                    "input_ids": str(token_ids),
                    "reward": trace.reward,
                }
                assert list(sample.keys()) == self.samples_cols, (
                    "Order of columns in the table must be the same as order of the keys here"
                )
                self.samples_table.add_data(*sample.values())

        key = "details/samples" if _details_routing_enabled() else "samples"
        wandb.log({key: self.samples_table, "step": step})
        self.last_log_samples_step = step
        self.logger.debug(f"Logged samples at step {step} to W&B table in {time.perf_counter() - start_time:.2f}s")

    def log_eval_samples(self, rollouts: list[Rollout], env_name: str, step: int) -> None:
        """Logs eval rollouts to a separate W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
        ):
            return

        for rollout in rollouts:
            trace = rollout
            for branch in trace.branches:
                # Eval runs the openai client (no token ids), so show the assistant message
                # content rather than decoded tokens.
                completion = "".join(m.content or "" for m in branch.messages if m.role == "assistant")
                if not completion:
                    continue
                sample = {
                    "step": step,
                    "env": env_name,
                    "task": _loggable_task(trace.task.data),
                    "task_idx": trace.task.data.idx,
                    "completion": completion,
                    "reward": trace.reward,
                }
                self.eval_samples_table.add_data(*sample.values())

        key = "details/eval/samples" if _details_routing_enabled() else "eval/samples"
        wandb.log({key: self.eval_samples_table, "step": step})

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """Log distributions (no-op for W&B)."""
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to W&B table."""
        if not self.is_master or not self.enabled:
            return

        self.logger.info("Saving final summary to file")
        assert self.output_dir is not None, "Output directory is required for saving final summary"
        dir_path = self.output_dir / f"run-{self.wandb.id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / filename, "w") as f:
            json.dump(wandb.summary._as_dict(), f)


# --- curated "overview" saved view -------------------------------------------------------------
# prime-rl logs many metrics; the default workspace auto-generates a panel per key, which buries the
# few that matter. These build a named saved view grouping the important metrics into sections, so a
# new project gets a usable overview without hand-picking panels. Panels are untitled — each shows
# its raw metric name.

OVERVIEW_NAME = "overview"

# Additional per-rollout metrics shown for evaluation.
COMMON_METRICS = [
    "has_error/mean",
    "is_truncated/mean",
    "num_total_tokens/mean",
    "num_turns/mean",
    "num_branches/mean",
]

STABILITY_METRICS = ["optim/grad_norm", "entropy/all/mean", "mismatch_kl/all/mean", "kl_ent_ratio/mean"]

PERFORMANCE_METRICS = [
    "perf/mfu",
    "time/step",
    "time/wait_for_batch",
    "time/wait_for_policy",
    "time/forward_backward",
    "train/agg/all/timing/generation/mean",
    "train/agg/all/metrics/reader_seconds/mean",
    "inference/agg/throughput",
    "inference/agg/running_requests",
    "inference/agg/waiting_requests",
    "inference/agg/kv_cache_usage_mean",
    "inference/agg/prefix_cache_hit_rate",
]

# Dense grid: more, smaller panels per row and enough rows that sections don't paginate.
COLUMNS = 4
ROWS = 6


def line_panels(metrics: Sequence[str], regexes: Sequence[str]) -> list[wr.LinePlot]:
    # inference/* is logged against wall time (step_metric="_timestamp") → "WallTime" (== W&B's
    # "_timestamp"); everything else on "step" (prime-rl's logged training step, not internal "Step").
    # x is set per-panel because LinePlot defaults it to "Step", which overrides the workspace x_axis.
    return [
        wr.LinePlot(x="WallTime" if m.removeprefix("details/").startswith("inference/") else "step", y=[m])
        for m in metrics
    ] + [
        wr.LinePlot(x="step", metric_regex=r) for r in regexes
    ]


def section(name: str, metrics: Sequence[str] = (), regexes: Sequence[str] = ()) -> ws.Section:
    prefix = "details/" if _details_routing_enabled() else ""
    return ws.Section(
        name=name,
        is_open=True,
        panels=line_panels([prefix + m for m in metrics], [prefix + r for r in regexes]),
        layout_settings=ws.SectionLayoutSettings(columns=COLUMNS, rows=ROWS),
    )


def train_section(name: str, scope: str) -> ws.Section:
    return section(
        name,
        metrics=[f"{scope}/all/reward/{stat}" for stat in ("mean", "p10", "p90")]
        + ["optim/grad_norm"]
        + [f"{scope}/all/metrics/relppl/{stat}" for stat in ("mean", "p10", "p90")]
        + [f"{scope}/all/num_total_tokens/mean"],
    )


def eval_section(name: str, env_pattern: str) -> ws.Section:
    # Eval's score is "avg@k" (dynamic k → regex); these regexes can also cover any env.
    return section(
        name,
        regexes=[f"eval/{env_pattern}/all/avg@.*"] + [f"eval/{env_pattern}/all/{m}" for m in COMMON_METRICS],
    )


def build_sections(train_envs: Sequence[str] = (), eval_envs: Sequence[str] = ()) -> list[ws.Section]:
    # With one env the aggregate == that env, so show only its section. With several, put the
    # cross-env aggregate on top followed by a section per env.
    if len(train_envs) == 1:
        sections = [train_section(f"train/{train_envs[0]}", f"train/{train_envs[0]}")]
    elif len(train_envs) > 1:
        sections = [train_section("train/agg", "train/agg")]
        sections += [train_section(f"train/{env}", f"train/{env}") for env in train_envs]
    else:
        # Env names unknown (e.g. SFT): fall back to the aggregate.
        sections = [train_section("train", "train/agg")]
    if eval_envs:
        sections += [eval_section(f"eval/{env}", re.escape(env)) for env in eval_envs]
    else:
        # Env names unknown (e.g. SFT): one regex section matching any eval env.
        sections.append(eval_section("eval", ".*"))
    sections.append(section("stability", metrics=STABILITY_METRICS))
    sections.append(section("performance", metrics=PERFORMANCE_METRICS))
    return sections


def list_views(entity: str, project: str) -> list[tuple[str, str]]:
    """``(display_name, internal_name)`` for every saved view in the project."""
    query = gql(
        """
        query Views($entity: String!, $project: String!) {
          project(name: $project, entityName: $entity) {
            allViews(viewType: "project-view") { edges { node { name displayName } } }
          }
        }
        """
    )
    res = wandb.Api().client.execute(query, variable_values={"entity": entity, "project": project})
    edges = ((res.get("project") or {}).get("allViews") or {}).get("edges") or []
    return [(e["node"]["displayName"], e["node"]["name"]) for e in edges if e.get("node")]


def merge_overview_sections(target: list[ws.Section], incoming: Sequence[ws.Section]) -> bool:
    """Keep existing charts and add missing sections or metrics in place."""
    changed = False
    by_name = {section.name: section for section in target}
    for section in incoming:
        if section.name not in by_name:
            target.append(section)
            by_name[section.name] = section
            changed = True
            continue
        existing = by_name[section.name]
        for panel in section.panels:
            if isinstance(panel, wr.LinePlot):
                duplicate = any(
                    isinstance(other, wr.LinePlot)
                    and (other.x, other.y, other.metric_regex) == (panel.x, panel.y, panel.metric_regex)
                    for other in existing.panels
                )
            else:
                duplicate = panel in existing.panels
            if not duplicate:
                panel.layout.y = max((p.layout.y + p.layout.h for p in existing.panels), default=0)
                existing.panels.append(panel)
                changed = True
    return changed


def ensure_overview_view(
    entity: str,
    project: str,
    name: str = OVERVIEW_NAME,
    train_envs: Sequence[str] = (),
    eval_envs: Sequence[str] = (),
) -> str | None:
    """Update the single overview in place, retaining charts from previous runs."""
    sections = build_sections(train_envs, eval_envs)
    for display_name, internal_name in list_views(entity, project):
        if display_name == name:
            slug = internal_name.removeprefix("nw-").removesuffix("-v")
            workspace = ws.Workspace.from_url(f"https://wandb.ai/{entity}/{project}?nw={slug}")
            if not merge_overview_sections(workspace.sections, sections):
                return None
            workspace.save()
            return workspace.url
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=sections,
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="step"),
    )
    workspace.save()
    return workspace.url
