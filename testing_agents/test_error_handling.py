"""
Unit tests for the exception-handling guards added to each agent.
These use lightweight fakes/mocks so no real API keys or network
calls are needed to run the suite.

Run with:
    pytest tests/test_error_handling.py -v
"""
import sys
import os
import types
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import Vault


# ---------------------------------------------------------------------
# Pathfinder
# ---------------------------------------------------------------------
class TestPathfinderErrors:
    def test_empty_llm_response_raises(self):
        from agents.pathfinder import pathfinder_plan, PathfinderError

        vault = Vault()
        vault.mode = "topic"
        vault.input_data = "AI in Healthcare"

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = types.SimpleNamespace(content="   ")  # whitespace only
        with patch("agents.pathfinder.llm", fake_llm):
            with pytest.raises(PathfinderError):
                pathfinder_plan(vault)

    def test_llm_exception_is_wrapped(self):
        from agents.pathfinder import pathfinder_plan, PathfinderError

        vault = Vault()
        vault.mode = "topic"
        vault.input_data = "AI in Healthcare"

        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = RuntimeError("rate limited")
        with patch("agents.pathfinder.llm", fake_llm):
            with pytest.raises(PathfinderError):
                pathfinder_plan(vault)

    def test_successful_plan_populates_subtasks(self):
        from agents.pathfinder import pathfinder_plan

        vault = Vault()
        vault.mode = "topic"
        vault.input_data = "AI in Healthcare"

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = types.SimpleNamespace(content="1. Q1\n2. Q2\n3. Q3")
        with patch("agents.pathfinder.llm", fake_llm):
            result = pathfinder_plan(vault)

        assert result == ["1. Q1", "2. Q2", "3. Q3"]
        assert vault.subtasks == result


# ---------------------------------------------------------------------
# Harvester
# ---------------------------------------------------------------------
class TestHarvesterErrors:
    def test_missing_file_raises_file_not_found(self):
        from agents.harvester import harvester_parse_single

        vault = Vault()
        vault.input_data = ["docs/does_not_exist.pdf"]

        with pytest.raises(FileNotFoundError):
            harvester_parse_single(vault)

    def test_no_subtasks_raises_before_calling_tavily(self):
        from agents.harvester import harvester_web_search, HarvesterError

        vault = Vault()
        vault.subtasks = []

        with pytest.raises(HarvesterError):
            harvester_web_search(vault)

    def test_all_tavily_queries_failing_raises(self):
        from agents.harvester import harvester_web_search, HarvesterError

        vault = Vault()
        vault.subtasks = ["question one", "question two"]

        with patch("agents.harvester.tavily.search", side_effect=RuntimeError("network error")):
            with pytest.raises(HarvesterError):
                harvester_web_search(vault)

    def test_one_failed_query_does_not_kill_the_run(self):
        from agents.harvester import harvester_web_search

        vault = Vault()
        vault.subtasks = ["good question", "bad question"]

        good_result = {"results": [{"content": "some content", "url": "https://example.com"}]}

        def fake_search(query, max_results):
            if query == "bad question":
                raise RuntimeError("temporary failure")
            return good_result

        with patch("agents.harvester.tavily.search", side_effect=fake_search):
            with patch("agents.harvester.save_to_vector_db", return_value=MagicMock()) as mock_save:
                harvester_web_search(vault)

        # The good query's result should still have been stored despite the bad one failing.
        mock_save.assert_called_once()
        stored_texts = mock_save.call_args[0][0]
        assert "some content" in stored_texts


# ---------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------
class TestSynthesizerErrors:
    def test_no_facts_raises_for_topic_mode(self):
        from agents.synthesizer import synthesizer_run, SynthesizerError

        vault = Vault()
        vault.mode = "topic"
        vault.facts = []

        with pytest.raises(SynthesizerError):
            synthesizer_run(vault)

    def test_missing_vector_store_raises_for_multi_doc(self):
        from agents.synthesizer import synthesizer_run, SynthesizerError

        vault = Vault()
        vault.mode = "multi_doc"
        vault.vector_store = None
        vault.subtasks = ["dim1", "dim2"]

        with pytest.raises(SynthesizerError):
            synthesizer_run(vault)


# ---------------------------------------------------------------------
# Scribe
# ---------------------------------------------------------------------
class TestScribeErrors:
    def test_missing_synthesis_raises(self):
        from agents.scribe import scribe_write, ScribeError

        vault = Vault()
        vault.mode = "topic"
        vault.synthesis = ""

        with pytest.raises(ScribeError):
            scribe_write(vault)

    def test_unknown_mode_raises_value_error(self):
        from agents.scribe import scribe_write

        vault = Vault()
        vault.mode = "not_a_real_mode"
        vault.synthesis = "some synthesis text"

        with pytest.raises(ValueError):
            scribe_write(vault)