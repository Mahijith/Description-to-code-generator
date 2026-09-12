from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.agents import ArchitectAgent, DeveloperAgent, ProjectManagerAgent, QAReviewerAgent, RequirementsAnalystAgent
from pipeline.llm import MockLLMProvider
from pipeline.orchestrator import Orchestrator
from pipeline.secrets import Secrets, mask_key
from pipeline.transcribe import GroqWhisperTranscriber, PassthroughTranscriber, TranscriptionError, make_transcriber_for

SAMPLE_TRANSCRIPT = Path("examples/sample_transcript.txt").read_text()


def test_secrets_never_exposes_raw_key_via_repr_or_str():
    secrets = Secrets("sk-super-secret-value-123")
    assert "sk-super-secret-value-123" not in repr(secrets)
    assert "sk-super-secret-value-123" not in str(secrets)
    assert secrets.reveal_openrouter_key() == "sk-super-secret-value-123"


def test_secrets_masks_short_keys_safely():
    secrets = Secrets("short")
    assert "short" not in repr(secrets)


def test_mask_key_never_exposes_the_raw_value_but_shows_length_and_ends():
    masked = mask_key("gsk_abcdefghijklmnopqrstuvwxyz")
    assert "abcdefghijklmnopqrstuvwxyz" not in masked
    assert masked.startswith("gsk_")
    assert masked.endswith("wxyz (len 30)")


def test_mask_key_handles_empty_and_short_keys():
    assert mask_key(None) == "<no key>"
    assert mask_key("") == "<no key>"
    assert "short" not in mask_key("short")


def test_secrets_reports_no_key():
    secrets = Secrets(None)
    assert not secrets.has_openrouter_key
    with pytest.raises(ValueError):
        secrets.reveal_openrouter_key()


def test_test_connection_succeeds_on_a_working_provider():
    class OkProvider(MockLLMProvider):
        def complete(self, prompt):
            return "OK"

    assert OkProvider().test_connection() == "OK"


def test_test_connection_propagates_llm_error():
    from pipeline.llm import LLMError, LLMProvider

    class BrokenProvider(LLMProvider):
        def complete(self, prompt):
            raise LLMError("OpenRouter rejected the API key (401 Unauthorized)")

    with pytest.raises(LLMError, match="401"):
        BrokenProvider().test_connection()


def test_passthrough_transcriber_reads_text_file(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    assert PassthroughTranscriber().transcribe(f) == "hello world"


def test_make_transcriber_for_picks_passthrough_for_text():
    assert isinstance(make_transcriber_for("foo.txt"), PassthroughTranscriber)
    assert isinstance(make_transcriber_for("foo.mp3"), GroqWhisperTranscriber)


def test_pm_kickoff_produces_a_brief():
    log = []
    pm = ProjectManagerAgent(MockLLMProvider(), log)
    brief = pm.kickoff(SAMPLE_TRANSCRIPT)
    assert brief.goal
    assert log and log[0]["stage"] == "pm_kickoff"


def test_requirements_agent_extracts_primary_entity():
    log = []
    pm = ProjectManagerAgent(MockLLMProvider(), log)
    brief = pm.kickoff(SAMPLE_TRANSCRIPT)
    req_agent = RequirementsAnalystAgent(MockLLMProvider(), log)
    requirements = req_agent.extract(SAMPLE_TRANSCRIPT, brief)
    assert requirements.primary_entity is not None
    field_names = {f.name for f in requirements.primary_entity.fields}
    assert {"title", "due_date", "priority", "completed"} <= field_names
    assert "add" in requirements.actions and "filter" in requirements.actions


def test_developer_agent_produces_well_formed_html():
    log = []
    llm = MockLLMProvider()
    req_agent = RequirementsAnalystAgent(llm, log)
    pm = ProjectManagerAgent(llm, log)
    brief = pm.kickoff(SAMPLE_TRANSCRIPT)
    requirements = req_agent.extract(SAMPLE_TRANSCRIPT, brief)
    architecture = ArchitectAgent(llm, log).design(requirements)
    html = DeveloperAgent(llm, log).build(requirements, architecture)

    assert html.strip().lower().startswith("<!doctype html>")
    assert "innerHTML" not in html
    assert "localStorage" in html
    for field_name in ("title", "due_date", "priority"):
        assert field_name in html


def test_qa_reviewer_flags_missing_empty_state_then_passes():
    log = []
    llm = MockLLMProvider()
    qa = QAReviewerAgent(llm, log)
    report1 = qa.review(_dummy_requirements(), _dummy_architecture(), "<html>no empty state here</html>")
    assert report1.passed is False
    assert any("empty" in issue.lower() for issue in report1.issues)

    report2 = qa.review(_dummy_requirements(), _dummy_architecture(), "<html>fixed</html>")
    assert report2.passed is True


def test_orchestrator_runs_full_loop_and_ends_passed():
    orchestrator = Orchestrator(MockLLMProvider(), max_qa_iterations=2)
    stages = []
    result = orchestrator.run(SAMPLE_TRANSCRIPT, on_stage=lambda s, status: stages.append((s, status)))

    assert result.iterations == 2
    assert result.qa_reports[0].passed is False
    assert result.qa_reports[-1].passed is True
    assert result.html.strip().lower().startswith("<!doctype html>")
    assert result.summary
    assert any(s.startswith("developer (pass 2)") for s, _ in stages)
    assert len(result.prompt_log) == len(orchestrator.prompt_log)
    # every prompt actually sent is recorded, in call order
    assert [p["stage"] for p in result.prompt_log][:3] == ["pm_kickoff", "requirements", "architect"]


def test_groq_transcriber_returns_text_on_success(tmp_path):
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")  # never actually decoded; the HTTP call is mocked

    transcriber = GroqWhisperTranscriber(api_key="test-key")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {"text": "  Hello world.  "}
    with patch("requests.post", return_value=fake_response) as mock_post:
        assert transcriber.transcribe(audio_path) == "Hello world."
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_groq_transcriber_requires_an_api_key(tmp_path):
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    transcriber = GroqWhisperTranscriber(api_key="")
    with pytest.raises(TranscriptionError, match="No Groq API key configured"):
        transcriber.transcribe(audio_path)


def test_groq_transcriber_wraps_http_errors_as_transcription_error(tmp_path):
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    transcriber = GroqWhisperTranscriber(api_key="test-key")
    fake_response = MagicMock(status_code=401, text="Invalid API Key")
    with patch("requests.post", return_value=fake_response):
        with pytest.raises(TranscriptionError, match="401"):
            transcriber.transcribe(audio_path)


def test_orchestrator_respects_max_iterations_even_if_never_passes():
    class AlwaysFailLLM(MockLLMProvider):
        def complete_json(self, prompt):
            if "STAGE: QA" in prompt:
                return {"passed": False, "issues": ["still broken"]}
            return super().complete_json(prompt)

    orchestrator = Orchestrator(AlwaysFailLLM(), max_qa_iterations=2)
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.iterations == 2
    assert result.qa_reports[-1].passed is False


def _dummy_requirements():
    from pipeline.schema import Entity, Field, Requirements

    return Requirements(
        app_name="Test",
        description="d",
        entities=[Entity(name="Task", fields=[Field(name="title", type="text")])],
        actions=["add"],
        filters=[],
        screens=[],
    )


def _dummy_architecture():
    from pipeline.schema import ArchitectureDoc

    return ArchitectureDoc(tech_approach="x", data_model_notes="y", screen_breakdown=[], style_notes="")
