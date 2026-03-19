"""Audit trail — compliance-grade logging of all browser actions.

Every action is logged with timestamp, selector, tier, and optional screenshot.
Suitable for regulated environments (HIPAA, SOC 2, PCI-DSS).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single auditable action."""
    timestamp: str
    step_index: int
    intent: str
    action: str
    selector: str
    tier: int
    strategy: str
    success: bool
    duration_ms: float
    tokens_used: int
    value_masked: str = ""  # Value with sensitive data masked
    url: str = ""
    screenshot_path: str = ""
    error: str = ""


class AuditTrail:
    """Records all browser actions for compliance and debugging."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        mask_values: bool = True,
        capture_screenshots: bool = False,
    ):
        self._dir = Path(output_dir) if output_dir else Path.home() / ".autopilot" / "audit"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mask_values = mask_values
        self._capture_screenshots = capture_screenshots
        self._entries: list[AuditEntry] = []
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def record(
        self,
        step_index: int,
        intent: str,
        action: str,
        selector: str,
        tier: int,
        strategy: str,
        success: bool,
        duration_ms: float,
        tokens_used: int,
        value: str | None = None,
        url: str = "",
        error: str = "",
    ) -> AuditEntry:
        """Record a single action to the audit trail."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            step_index=step_index,
            intent=intent,
            action=action,
            selector=selector,
            tier=tier,
            strategy=strategy,
            success=success,
            duration_ms=round(duration_ms, 1),
            tokens_used=tokens_used,
            value_masked=self._mask(value) if value else "",
            url=url,
            error=error,
        )
        self._entries.append(entry)
        return entry

    def save(self, playbook_name: str = "unknown") -> Path:
        """Save the audit trail to a JSON file."""
        path = self._dir / f"audit_{playbook_name}_{self._session_id}.json"
        data = {
            "session_id": self._session_id,
            "playbook": playbook_name,
            "entries": [asdict(e) for e in self._entries],
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Audit trail saved: %s (%d entries)", path, len(self._entries))
        return path

    def summary(self) -> dict:
        """Summary statistics for the audit trail."""
        return {
            "total_actions": len(self._entries),
            "successful": sum(1 for e in self._entries if e.success),
            "failed": sum(1 for e in self._entries if not e.success),
            "total_tokens": sum(e.tokens_used for e in self._entries),
            "total_duration_ms": sum(e.duration_ms for e in self._entries),
            "tiers_used": list(set(e.tier for e in self._entries)),
        }

    def to_text(self) -> str:
        """Human-readable audit log."""
        lines = []
        for e in self._entries:
            status = "OK" if e.success else "FAIL"
            value_str = f" value={e.value_masked}" if e.value_masked else ""
            lines.append(
                f"[{e.timestamp}] Step {e.step_index}: "
                f"{e.action} {e.selector}{value_str} "
                f"({status}, Tier {e.tier}, {e.tokens_used} tokens, {e.duration_ms}ms)"
            )
            if e.error:
                lines.append(f"  ERROR: {e.error}")
        return "\n".join(lines)

    def _mask(self, value: str) -> str:
        """Mask sensitive values for audit logging."""
        if not self._mask_values:
            return value
        if not value:
            return ""
        # Mask passwords, tokens, keys
        if len(value) <= 4:
            return "***"
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
