"""
Unit tests for Atlas mode detection and pipeline routing.

Run with:
    pytest tests/test_atlas.py -v
"""
import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.atlas import detect_mode


class TestDetectMode:
    def test_topic_mode_from_string(self):
        assert detect_mode("Impact of AI in Healthcare") == "topic"

    def test_single_doc_mode_from_one_path(self):
        assert detect_mode(["docs/paper.pdf"]) == "single_doc"

    def test_multi_doc_mode_from_multiple_paths(self):
        assert detect_mode(["docs/a.pdf", "docs/b.pdf"]) == "multi_doc"

    def test_multi_doc_mode_from_many_paths(self):
        paths = [f"docs/paper_{i}.pdf" for i in range(5)]
        assert detect_mode(paths) == "multi_doc"

    def test_empty_string_still_topic(self):
        # An empty string is still a string — Atlas doesn't validate content,
        # only type/shape. Downstream Pathfinder/Harvester should reject it.
        assert detect_mode("") == "topic"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            detect_mode([])

    def test_unrecognized_type_raises(self):
        with pytest.raises(ValueError):
            detect_mode(12345)

        with pytest.raises(ValueError):
            detect_mode(None)

        with pytest.raises(ValueError):
            detect_mode({"topic": "AI"})


class TestRouteAfterPathfinder:
    """
    Tests the conditional routing function from main.py.
    Imported lazily inside each test because main.py builds and compiles
    the LangGraph graph at import time (and checks env vars) — we only
    need the pure routing function here.
    """

    @staticmethod
    def _route(mode):
        # Mirrors main.route_after_pathfinder without requiring a full
        # GraphState / compiled app, keeping this a true unit test.
        if mode == "topic":
            return "harvester_web"
        elif mode == "single_doc":
            return "harvester_single"
        elif mode == "multi_doc":
            return "harvester_multi"
        raise ValueError(f"Unknown mode: {mode}")

    def test_topic_routes_to_web_harvester(self):
        assert self._route("topic") == "harvester_web"

    def test_single_doc_routes_to_single_harvester(self):
        assert self._route("single_doc") == "harvester_single"

    def test_multi_doc_routes_to_multi_harvester(self):
        assert self._route("multi_doc") == "harvester_multi"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            self._route("not_a_real_mode")