from __future__ import annotations

import secrets
from datetime import UTC, datetime


def new_id(prefix: str, when: datetime | None = None) -> str:
    """Sortable, human-readable id like `T-20260901-140322-a3f9`."""
    when = when or datetime.now(UTC)
    return f"{prefix}-{when.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
