"""Small, shared primitives for hash-locked evaluation protocols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest for one file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data independently of dictionary insertion order."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().lower()


def prepare_locked_protocol(
    protocol: Mapping[str, Any],
    protocol_path: str | Path,
    repository_root: str | Path,
    protocol_ids: str | tuple[str, ...],
    *,
    evaluation_type: str | None = None,
    code_message: str = "code hash mismatch: {relative}",
    artifact_message: str = "artifact hash mismatch: {relative}",
    output_message: str = "locked evaluation output already exists",
) -> tuple[Mapping[str, Any], Path, str]:
    """Run the common immutable-protocol preflight and return its inputs."""

    expected = (protocol_ids,) if isinstance(protocol_ids, str) else protocol_ids
    if protocol.get("protocol_id") not in expected:
        raise ValueError("unexpected protocol_id")
    if not protocol.get("locked_before_execution", False):
        raise ValueError("protocol is not locked")
    if (
        evaluation_type is not None
        and protocol.get("evaluation_type") != evaluation_type
    ):
        raise ValueError(
            f"unexpected evaluation_type: {protocol.get('evaluation_type')}"
        )

    root = Path(repository_root)
    for field, message in (
        ("code_sha256", code_message),
        ("artifact_sha256", artifact_message),
    ):
        manifest = protocol.get(field, {})
        if not isinstance(manifest, Mapping):
            raise TypeError(f"{field} must be a mapping")
        for relative, digest in manifest.items():
            if sha256_file(root / relative) != str(digest).lower():
                raise RuntimeError(message.format(relative=relative))

    configuration = protocol.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    relative_output = protocol.get("output_directory")
    if not isinstance(relative_output, str) or not relative_output.strip():
        raise ValueError("output_directory must be a non-empty string")
    output_dir = root / relative_output
    if output_dir.exists():
        raise FileExistsError(output_message)
    return configuration, output_dir, sha256_file(protocol_path)
