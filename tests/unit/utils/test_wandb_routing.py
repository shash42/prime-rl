from prime_rl.utils.monitor.wandb import WandbMonitor, route_default_workspace_metrics


def test_default_workspace_routing_keeps_selected_means_only() -> None:
    routed = route_default_workspace_metrics(
        {
            "step": 20,
            "train/agg/effective/reward/mean": 0.2,
            "train/agg/effective/reward/p90": 0.8,
            "entropy/all/mean": 1.5,
            "loss/mean": 0.3,
            "train/agg/effective/num_output_tokens/mean": 700.0,
            "train/agg/effective/metrics/answer_tokens/mean": 80.0,
            "eval/pasttest/effective/avg@1": 1.0,
            "eval/pasttest/effective/metrics/rel_infogain/mean": 0.1,
            "eval/pasttest/effective/num_output_tokens/mean": 500.0,
            "eval/pasttest/effective/metrics/answer_tokens/mean": 60.0,
            "eval/pasttest/effective/num_output_tokens/p90": 900.0,
            "eval/futuretest/all/has_error/mean": 0.01,
            "time/step": 42.0,
        }
    )

    assert routed["step"] == 20
    assert routed["train/reward"] == 0.2
    assert routed["train/entropy"] == 1.5
    assert routed["train/loss"] == 0.3
    assert routed["train/mean_total_output_tokens"] == 700.0
    assert routed["train/mean_answer_tokens"] == 80.0
    assert routed["eval/pasttest/rel_infogain"] == 0.1
    assert routed["eval/pasttest/mean_total_output_tokens"] == 500.0
    assert routed["eval/pasttest/mean_answer_tokens"] == 60.0
    assert routed["eval/futuretest/error_rate"] == 0.01
    assert "train/agg/effective/reward/p90" not in routed
    assert "eval/pasttest/effective/num_output_tokens/p90" not in routed
    assert routed["details/train/agg/effective/reward/p90"] == 0.8
    assert routed["details/time/step"] == 42.0


def test_default_workspace_routing_does_not_rename_monitor_history(monkeypatch) -> None:
    monkeypatch.setenv("PRIME_WANDB_DETAILS", "1")
    monitor = WandbMonitor.__new__(WandbMonitor)
    monitor._keep_full_history = True
    monitor.history = []
    monitor.is_master = False
    monitor.enabled = False
    metrics = {"perf/peak_memory": 47.3}

    monitor.log(metrics, step=10)

    assert monitor.history == [metrics]
