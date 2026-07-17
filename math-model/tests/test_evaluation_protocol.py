"""Tests for the shared hash-locked evaluation protocol primitives."""

import hashlib
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.protocol import (
    canonical_sha256,
    prepare_locked_protocol,
    sha256_file,
)


def test_sha256_file_matches_standard_library(tmp_path):
    payload = b"AUV protocol\x00payload"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=3) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_rejects_non_positive_chunk_size(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sha256_file(path, chunk_size=0)


def test_canonical_sha256_ignores_dictionary_insertion_order():
    first = {"b": [2, 3], "a": {"x": 1}}
    second = {"a": {"x": 1}, "b": [2, 3]}

    assert canonical_sha256(first) == canonical_sha256(second)


def test_prepare_locked_protocol_returns_common_inputs(tmp_path):
    runner = tmp_path / "runner.py"
    protocol_path = tmp_path / "protocol.json"
    runner.write_text("print('ok')\n", encoding="utf-8")
    protocol_path.write_text("{}\n", encoding="utf-8")
    protocol = {
        "protocol_id": "benchmark_v1",
        "locked_before_execution": True,
        "evaluation_type": "paired",
        "configuration": {"missions": 3},
        "output_directory": "results/new_run",
        "code_sha256": {"runner.py": sha256_file(runner)},
    }

    configuration, output_dir, protocol_hash = prepare_locked_protocol(
        protocol,
        protocol_path,
        tmp_path,
        "benchmark_v1",
        evaluation_type="paired",
    )

    assert configuration == {"missions": 3}
    assert output_dir == tmp_path / "results" / "new_run"
    assert protocol_hash == sha256_file(protocol_path)


def test_prepare_locked_protocol_rejects_hash_mismatch(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    runner = tmp_path / "runner.py"
    protocol_path.write_text("{}\n", encoding="utf-8")
    runner.write_text("changed\n", encoding="utf-8")
    protocol = {
        "protocol_id": "benchmark_v1",
        "locked_before_execution": True,
        "configuration": {},
        "output_directory": "results/new_run",
        "code_sha256": {"runner.py": "0" * 64},
    }

    with pytest.raises(RuntimeError, match="changed after freeze: runner.py"):
        prepare_locked_protocol(
            protocol,
            protocol_path,
            tmp_path,
            "benchmark_v1",
            code_message="changed after freeze: {relative}",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"protocol_id": "wrong"}, "unexpected protocol_id"),
        ({"locked_before_execution": False}, "protocol is not locked"),
        ({"evaluation_type": "blind"}, "unexpected evaluation_type"),
    ],
)
def test_prepare_locked_protocol_rejects_invalid_metadata(
    tmp_path, override, message
):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    protocol = {
        "protocol_id": "benchmark_v1",
        "locked_before_execution": True,
        "evaluation_type": "paired",
        "configuration": {},
        "output_directory": "results/new_run",
        **override,
    }

    with pytest.raises(ValueError, match=message):
        prepare_locked_protocol(
            protocol,
            protocol_path,
            tmp_path,
            "benchmark_v1",
            evaluation_type="paired",
        )


def test_prepare_locked_protocol_rejects_output_overwrite(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "results" / "existing"
    output_dir.mkdir(parents=True)
    protocol = {
        "protocol_id": "benchmark_v1",
        "locked_before_execution": True,
        "configuration": {},
        "output_directory": "results/existing",
    }

    with pytest.raises(FileExistsError, match="already frozen"):
        prepare_locked_protocol(
            protocol,
            protocol_path,
            tmp_path,
            "benchmark_v1",
            output_message="already frozen",
        )
