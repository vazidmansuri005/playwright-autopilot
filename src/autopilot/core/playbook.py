"""Playbook: portable, human-readable browser automation recordings."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PlaybookStep:
    """A single recorded browser action."""

    intent: str                           # Human-readable: "fill email field"
    selector: str                         # Primary selector: "input[name='email']"
    selector_alternatives: list[str] = field(default_factory=list)
    action: str = "click"                 # click, fill, select, press, navigate, wait
    value: str | None = None              # For fill/select/press actions
    value_template: str | None = None     # "${email}" — variable substitution
    dom_fingerprint: str = ""             # Hash of surrounding DOM structure
    confidence: float = 1.0              # 0.0-1.0 confidence in selector
    tier_resolved: int = 0               # Which tier last resolved this step
    last_healed: str | None = None       # ISO timestamp of last heal
    wait_after_ms: int = 0               # Optional wait after action
    assertions: list[dict] = field(default_factory=list)  # Post-step assertions

    # Conditional execution
    condition: str = ""                   # "if_visible", "if_not_visible", "if_url_contains"
    condition_value: str = ""             # Value for condition (e.g., URL substring)
    skip_on_fail: bool = False            # Skip this step instead of failing the playbook

    # Natural language assertion (evaluated by LLM)
    assert_after: str = ""                # "cart badge shows 1 item", "login succeeded"

    # Network mocking
    network_mocks: list[dict] = field(default_factory=list)  # [{"url": "*/api/*", "response": {...}}]

    # Tracking
    heal_count: int = 0                   # Times this step has been healed
    run_count: int = 0                    # Times this step has been executed
    fail_count: int = 0                   # Times this step has failed

    def resolve_value(self, variables: dict[str, str]) -> str | None:
        """Substitute ${var} templates with actual values."""
        if self.value_template and variables:
            tpl = string.Template(self.value_template)
            return tpl.safe_substitute(variables)
        return self.value

    def all_selectors(self) -> list[str]:
        """Return primary + alternatives in order."""
        return [self.selector] + self.selector_alternatives


@dataclass
class Playbook:
    """A recorded browser automation flow."""

    name: str
    url: str                             # Starting URL
    steps: list[PlaybookStep] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)  # Required variable names
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    run_count: int = 0
    success_count: int = 0

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def success_rate(self) -> float:
        if self.run_count == 0:
            return 0.0
        return self.success_count / self.run_count

    def add_step(self, step: PlaybookStep) -> None:
        self.steps.append(step)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def update_step(self, index: int, **kwargs: Any) -> None:
        """Update a step's fields (e.g., after healing)."""
        step = self.steps[index]
        for key, val in kwargs.items():
            if hasattr(step, key):
                setattr(step, key, val)
        step.last_healed = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def record_run(self, success: bool) -> None:
        self.run_count += 1
        if success:
            self.success_count += 1

    def extract_variables(self) -> list[str]:
        """Scan all steps for ${var} templates and return variable names."""
        pattern = re.compile(r"\$\{(\w+)\}")
        found: set[str] = set()
        for step in self.steps:
            if step.value_template:
                found.update(pattern.findall(step.value_template))
        self.variables = sorted(found)
        return self.variables

    def save(self, path: str | Path) -> None:
        """Save playbook to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Playbook saved: %s (%d steps)", path, len(self.steps))

    @classmethod
    def load(cls, path: str | Path) -> "Playbook":
        """Load playbook from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        steps_data = data.pop("steps", [])
        steps = [PlaybookStep(**s) for s in steps_data]
        return cls(steps=steps, **data)

    def fingerprint(self) -> str:
        """Generate a unique hash for this playbook's structure."""
        content = json.dumps(
            [{"intent": s.intent, "action": s.action, "selector": s.selector} for s in self.steps],
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
