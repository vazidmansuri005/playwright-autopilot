"""playwright-autopilot: Browser automation that gets cheaper every time you run it."""

from autopilot.core.browser import Browser
from autopilot.core.runner import Runner
from autopilot.core.playbook import Playbook, PlaybookStep
from autopilot.core.explorer import Explorer
from autopilot.agent import Agent

__version__ = "0.1.0"
__all__ = ["Agent", "Browser", "Explorer", "Runner", "Playbook", "PlaybookStep"]
