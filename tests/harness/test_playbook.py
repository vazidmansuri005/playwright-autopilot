"""Playbook persistence and manipulation tests."""

import json
import tempfile
from pathlib import Path

import pytest

from autopilot.core.playbook import Playbook, PlaybookStep


class TestPlaybook:
    """Test playbook serialization and features."""

    def test_create_playbook(self):
        pb = Playbook(name="test", url="https://example.com")
        assert pb.name == "test"
        assert pb.run_count == 0
        assert pb.version == 1
        assert pb.created_at

    def test_add_step(self):
        pb = Playbook(name="test", url="https://example.com")
        step = PlaybookStep(intent="click button", selector="#btn", action="click")
        pb.add_step(step)
        assert len(pb.steps) == 1

    def test_save_and_load(self, tmp_path):
        pb = Playbook(
            name="save-test",
            url="https://example.com",
            steps=[
                PlaybookStep(
                    intent="fill email",
                    selector="#email",
                    selector_alternatives=["input[type='email']"],
                    action="fill",
                    value_template="${email}",
                ),
                PlaybookStep(
                    intent="click submit",
                    selector="#submit",
                    action="click",
                ),
            ],
        )

        path = tmp_path / "test_playbook.json"
        pb.save(path)

        loaded = Playbook.load(path)
        assert loaded.name == "save-test"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].intent == "fill email"
        assert loaded.steps[0].value_template == "${email}"
        assert loaded.steps[1].action == "click"

    def test_variable_extraction(self):
        pb = Playbook(
            name="var-test",
            url="https://example.com",
            steps=[
                PlaybookStep(intent="fill", selector="#a", action="fill", value_template="${email}"),
                PlaybookStep(intent="fill", selector="#b", action="fill", value_template="${password}"),
                PlaybookStep(intent="fill", selector="#c", action="fill", value_template="${email}"),
            ],
        )

        vars = pb.extract_variables()
        assert vars == ["email", "password"]

    def test_variable_substitution(self):
        step = PlaybookStep(
            intent="fill", selector="#a", action="fill",
            value_template="${email}",
        )
        result = step.resolve_value({"email": "user@test.com"})
        assert result == "user@test.com"

    def test_variable_substitution_no_template(self):
        step = PlaybookStep(
            intent="fill", selector="#a", action="fill",
            value="literal_value",
        )
        result = step.resolve_value({"email": "user@test.com"})
        assert result == "literal_value"

    def test_run_tracking(self):
        pb = Playbook(name="track", url="https://example.com")
        pb.record_run(True)
        pb.record_run(True)
        pb.record_run(False)

        assert pb.run_count == 3
        assert pb.success_count == 2
        assert abs(pb.success_rate - 0.666) < 0.01

    def test_fingerprint_stable(self):
        pb = Playbook(
            name="fp", url="https://example.com",
            steps=[PlaybookStep(intent="click", selector="#a", action="click")],
        )
        fp1 = pb.fingerprint()
        fp2 = pb.fingerprint()
        assert fp1 == fp2

    def test_fingerprint_changes_on_edit(self):
        pb = Playbook(
            name="fp", url="https://example.com",
            steps=[PlaybookStep(intent="click", selector="#a", action="click")],
        )
        fp1 = pb.fingerprint()
        pb.steps[0].selector = "#b"
        fp2 = pb.fingerprint()
        assert fp1 != fp2

    def test_all_selectors(self):
        step = PlaybookStep(
            intent="click",
            selector="#primary",
            selector_alternatives=["#alt1", "#alt2"],
            action="click",
        )
        assert step.all_selectors() == ["#primary", "#alt1", "#alt2"]

    def test_update_step(self):
        pb = Playbook(
            name="update", url="https://example.com",
            steps=[PlaybookStep(intent="click", selector="#old", action="click")],
        )
        pb.update_step(0, selector="#new")
        assert pb.steps[0].selector == "#new"
        assert pb.steps[0].last_healed is not None
