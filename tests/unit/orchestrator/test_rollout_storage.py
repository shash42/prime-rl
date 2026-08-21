import gzip

from prime_rl.orchestrator.utils import compress_rollouts, save_rollouts


def test_completed_trace_is_atomically_compressed(tmp_path):
    path = tmp_path / "traces.jsonl"
    save_rollouts([{"id": "one"}, {"id": "two"}], path)

    compress_rollouts(path)

    compressed = tmp_path / "traces.jsonl.gz"
    assert not path.exists()
    with gzip.open(compressed, "rt") as handle:
        assert [line.strip() for line in handle] == [
            '{"id":"one"}',
            '{"id":"two"}',
        ]

    save_rollouts([{"id": "three"}], path)
    compress_rollouts(path)
    with gzip.open(compressed, "rt") as handle:
        assert [line.strip() for line in handle][-1] == '{"id":"three"}'
