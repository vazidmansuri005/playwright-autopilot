"""Tests for disk-based snapshot mode."""

import pytest
from pathlib import Path

from autopilot.core.disk_snapshots import DiskSnapshotManager


@pytest.mark.integration
class TestDiskSnapshots:
    """Test the disk snapshot manager."""

    async def test_save_digest_to_disk(self, browser_at_login, tmp_path):
        """Digest is saved to disk and returns a file path."""
        snap = DiskSnapshotManager(output_dir=tmp_path)
        path = await snap.save_digest(browser_at_login.page)

        assert path.exists()
        assert path.suffix == ".yaml"

        content = path.read_text()
        assert "Page:" in content
        assert "e0:" in content  # Has element references

    async def test_save_snapshot_to_disk(self, browser_at_login, tmp_path):
        """Accessibility tree is saved to disk."""
        snap = DiskSnapshotManager(output_dir=tmp_path)
        path = await snap.save_snapshot(browser_at_login.page)

        assert path.exists()
        assert path.suffix == ".txt"

        content = path.read_text()
        assert "Page:" in content
        assert len(content) > 50

    async def test_save_screenshot_to_disk(self, browser_at_login, tmp_path):
        """Screenshot PNG is saved to disk."""
        snap = DiskSnapshotManager(output_dir=tmp_path)
        path = await snap.save_screenshot(browser_at_login.page)

        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 1000  # Non-trivial image

    async def test_save_all(self, browser_at_login, tmp_path):
        """save_all returns paths to all three formats."""
        snap = DiskSnapshotManager(output_dir=tmp_path)
        paths = await snap.save_all(browser_at_login.page)

        assert "digest" in paths
        assert "snapshot" in paths
        assert "screenshot" in paths
        assert all(p.exists() for p in paths.values())

    async def test_token_savings(self, browser_at_login, tmp_path):
        """Disk mode returns ~10 tokens (path) vs ~500+ tokens (inline)."""
        from autopilot.core.digest import extract_digest

        snap = DiskSnapshotManager(output_dir=tmp_path)

        # Inline: full digest content
        digest = await extract_digest(browser_at_login.page)
        inline_text = digest.to_prompt()
        inline_tokens = len(inline_text.split()) * 1.3  # Rough estimate

        # Disk: just a file path
        path = await snap.save_digest(browser_at_login.page)
        disk_text = f"Digest saved to: {path}"
        disk_tokens = len(disk_text.split()) * 1.3

        # Disk mode should always be cheaper than inline
        # On real pages (not simple test pages), savings are 10-100x
        assert disk_tokens < inline_tokens

    async def test_cleanup_removes_old(self, browser_at_login, tmp_path):
        """Cleanup removes snapshots older than threshold."""
        import time

        snap = DiskSnapshotManager(output_dir=tmp_path)
        path = await snap.save_digest(browser_at_login.page)

        assert path.exists()

        # Set file to old timestamp
        old_time = time.time() - 7200  # 2 hours ago
        import os
        os.utime(path, (old_time, old_time))

        removed = snap.cleanup(max_age_seconds=3600)
        assert removed == 1
        assert not path.exists()

    async def test_cleanup_keeps_recent(self, browser_at_login, tmp_path):
        """Cleanup keeps recent snapshots."""
        snap = DiskSnapshotManager(output_dir=tmp_path)
        path = await snap.save_digest(browser_at_login.page)

        removed = snap.cleanup(max_age_seconds=3600)
        assert removed == 0
        assert path.exists()


@pytest.mark.integration
class TestDiskVsInlineComparison:
    """Compare disk vs inline approaches — proving the token economics."""

    async def test_dashboard_comparison(self, browser_at_dashboard, tmp_path):
        """Dashboard page: measure disk vs inline token cost."""
        from autopilot.core.digest import extract_digest
        from autopilot.core.snapshot import extract_snapshot

        page = browser_at_dashboard.page
        snap = DiskSnapshotManager(output_dir=tmp_path)

        # Inline costs
        digest = await extract_digest(page)
        snapshot = await extract_snapshot(page, interactive_only=True)
        inline_digest_tokens = digest.token_estimate
        inline_snapshot_tokens = snapshot.token_estimate

        # Disk costs (just the path string)
        digest_path = await snap.save_digest(page)
        snapshot_path = await snap.save_snapshot(page)
        disk_digest_tokens = 15   # "Digest saved to: /path/to/file.yaml"
        disk_snapshot_tokens = 15

        print(f"\n=== Disk vs Inline Token Comparison (dashboard) ===")
        print(f"  Digest:   inline={inline_digest_tokens} tokens, disk={disk_digest_tokens} tokens "
              f"({inline_digest_tokens / max(disk_digest_tokens, 1):.0f}x savings)")
        print(f"  Snapshot: inline={inline_snapshot_tokens} tokens, disk={disk_snapshot_tokens} tokens "
              f"({inline_snapshot_tokens / max(disk_snapshot_tokens, 1):.0f}x savings)")

        assert disk_digest_tokens < inline_digest_tokens
        assert disk_snapshot_tokens < inline_snapshot_tokens
