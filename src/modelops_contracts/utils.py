"""Utility functions for consistent ID generation and hashing."""

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import SimTask


def canonical_task_id(task: "SimTask") -> str:
    """Generate canonical task ID for simulation tasks.

    This ID is stable across workers and indexers.

    Args:
        task: SimTask to generate ID for

    Returns:
        Hex string task ID (32 chars)
    """
    s = f"sim:v1|{task.bundle_ref}|{task.entrypoint}|{task.params.param_id}|{task.seed}"
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()


def bundle12(bundle_ref: str) -> str:
    """Extract first 12 chars of bundle digest from reference.

    Args:
        bundle_ref: Full bundle reference like "oci://reg/model@sha256:abc123..."

    Returns:
        First 12 chars of digest
    """
    # Handle both @ and : separators
    digest = bundle_ref.split("@")[-1] if "@" in bundle_ref else bundle_ref
    if ":" in digest:
        return digest.split(":", 1)[1][:12]
    return digest[:12]


def aggregation_id(target_entrypoint: str, task_ids: list[str]) -> str:
    """Generate stable aggregation ID from target and task IDs.

    Args:
        target_entrypoint: Target module:object string
        task_ids: List of task IDs to aggregate

    Returns:
        Hex string aggregation ID (16 chars)
    """
    key = f"{target_entrypoint}:{','.join(sorted(task_ids))}"
    return hashlib.blake2b(key.encode(), digest_size=16).hexdigest()[:16]