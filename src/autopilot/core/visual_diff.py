"""Visual regression detection — screenshot diffing between runs.

Compares screenshots taken at each step against a baseline.
Detects UI changes that pass functionally but look different.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class VisualDiff:
    """Result of comparing two screenshots."""
    step_index: int
    intent: str
    changed: bool
    change_pct: float  # 0.0-100.0
    baseline_path: Path | None
    current_path: Path | None


class VisualDiffTracker:
    """Track and compare screenshots across playbook runs."""

    def __init__(self, baseline_dir: str | Path | None = None):
        self._dir = Path(baseline_dir) if baseline_dir else Path.home() / ".autopilot" / "baselines"
        self._dir.mkdir(parents=True, exist_ok=True)
        self.diffs: list[VisualDiff] = []

    async def capture_and_compare(
        self,
        page: Page,
        playbook_name: str,
        step_index: int,
        intent: str,
    ) -> VisualDiff:
        """Capture screenshot and compare against baseline."""
        current_path = self._dir / f"{playbook_name}_step{step_index}_current.png"
        baseline_path = self._dir / f"{playbook_name}_step{step_index}_baseline.png"

        # Capture current
        await page.screenshot(path=str(current_path))

        if not baseline_path.exists():
            # No baseline — save as baseline
            current_path.rename(baseline_path)
            diff = VisualDiff(
                step_index=step_index, intent=intent,
                changed=False, change_pct=0.0,
                baseline_path=baseline_path, current_path=None,
            )
        else:
            # Compare by file hash (pixel-perfect comparison)
            baseline_hash = _file_hash(baseline_path)
            current_hash = _file_hash(current_path)

            if baseline_hash == current_hash:
                current_path.unlink()  # No change, remove current
                diff = VisualDiff(
                    step_index=step_index, intent=intent,
                    changed=False, change_pct=0.0,
                    baseline_path=baseline_path, current_path=None,
                )
            else:
                # Files differ — compute approximate change percentage
                change_pct = await _estimate_change_pct(baseline_path, current_path)
                diff = VisualDiff(
                    step_index=step_index, intent=intent,
                    changed=True, change_pct=change_pct,
                    baseline_path=baseline_path, current_path=current_path,
                )
                logger.warning(
                    "Visual change at step %d '%s': %.1f%% different",
                    step_index, intent, change_pct,
                )

        self.diffs.append(diff)
        return diff

    def update_baselines(self) -> int:
        """Promote all current screenshots to baselines. Returns count updated."""
        updated = 0
        for diff in self.diffs:
            if diff.changed and diff.current_path and diff.current_path.exists():
                if diff.baseline_path:
                    diff.current_path.rename(diff.baseline_path)
                    updated += 1
        return updated

    def summary(self) -> dict:
        return {
            "total_steps": len(self.diffs),
            "changed": sum(1 for d in self.diffs if d.changed),
            "unchanged": sum(1 for d in self.diffs if not d.changed),
            "max_change_pct": max((d.change_pct for d in self.diffs), default=0.0),
        }


def _file_hash(path: Path) -> str:
    """SHA-256 hash of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _estimate_change_pct(baseline: Path, current: Path) -> float:
    """Estimate visual change percentage by comparing raw bytes.

    This is a fast approximation — not pixel-perfect diffing, but good
    enough to detect meaningful changes. For exact diffing, use pillow.
    """
    b_bytes = baseline.read_bytes()
    c_bytes = current.read_bytes()

    # Compare file sizes as rough proxy
    size_diff = abs(len(b_bytes) - len(c_bytes))
    max_size = max(len(b_bytes), len(c_bytes))
    if max_size == 0:
        return 0.0

    # Compare byte content overlap
    min_len = min(len(b_bytes), len(c_bytes))
    if min_len == 0:
        return 100.0

    # Sample every 100th byte for speed
    sample_size = max(1, min_len // 100)
    step = min_len // sample_size
    diff_count = sum(
        1 for i in range(0, min_len, step)
        if b_bytes[i] != c_bytes[i]
    )

    return min(100.0, (diff_count / sample_size) * 100 + (size_diff / max_size) * 50)
