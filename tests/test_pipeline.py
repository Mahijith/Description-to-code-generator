import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.agents import ArchitectAgent, DeveloperAgent, ProjectManagerAgent, QAReviewerAgent, RequirementsAnalystAgent
from pipeline.agents import TesterAgent as PipelineTesterAgent
from pipeline.auth_contract import (
    AUTH_STATUS_ID,
    LOGIN_PASSWORD_ID,
    LOGIN_SUBMIT_ID,
    LOGIN_USERNAME_ID,
    REGISTER_PASSWORD_ID,
    REGISTER_SUBMIT_ID,
    REGISTER_USERNAME_ID,
)
from pipeline.browser_tester import run_browser_auth_test
from pipeline.llm import AIHubMixProvider, LLMError, MockLLMProvider, OpenRouterProvider
from pipeline.orchestrator import Orchestrator
from pipeline.secrets import Secrets, mask_key
from pipeline.transcribe import GroqWhisperTranscriber, PassthroughTranscriber, TranscriptionError, make_transcriber_for

CORRECT_AUTH_HTML = f"""<!doctype html>
<html><body>
<input id="{REGISTER_USERNAME_ID}"><input id="{REGISTER_PASSWORD_ID}" type="password">
<button id="{REGISTER_SUBMIT_ID}">Register</button>
<input id="{LOGIN_USERNAME_ID}"><input id="{LOGIN_PASSWORD_ID}" type="password">
<button id="{LOGIN_SUBMIT_ID}">Login</button>
<div id="{AUTH_STATUS_ID}"></div>
<script>
function getUsers() {{ return JSON.parse(localStorage.getItem('users') || '{{}}'); }}
function saveUsers(u) {{ localStorage.setItem('users', JSON.stringify(u)); }}
document.getElementById('{REGISTER_SUBMIT_ID}').onclick = function() {{
  var u = document.getElementById('{REGISTER_USERNAME_ID}').value;
  var p = document.getElementById('{REGISTER_PASSWORD_ID}').value;
  var users = getUsers();
  if (users[u]) {{ return; }}
  users[u] = p;
  saveUsers(users);
}};
document.getElementById('{LOGIN_SUBMIT_ID}').onclick = function() {{
  var u = document.getElementById('{LOGIN_USERNAME_ID}').value;
  var p = document.getElementById('{LOGIN_PASSWORD_ID}').value;
  var users = getUsers();
  document.getElementById('{AUTH_STATUS_ID}').textContent = (users[u] === p) ? ('Logged in as ' + u) : '';
}};
</script>
</body></html>"""

BUGGY_AUTH_HTML = f"""<!doctype html>
<html><body>
<input id="{REGISTER_USERNAME_ID}"><input id="{REGISTER_PASSWORD_ID}" type="password">
<button id="{REGISTER_SUBMIT_ID}">Register</button>
<input id="{LOGIN_USERNAME_ID}"><input id="{LOGIN_PASSWORD_ID}" type="password">
<button id="{LOGIN_SUBMIT_ID}">Login</button>
<div id="{AUTH_STATUS_ID}"></div>
<script>
function getUsers() {{ return JSON.parse(localStorage.getItem('users') || '{{}}'); }}
function saveUsers(u) {{ localStorage.setItem('users', JSON.stringify(u)); }}
document.getElementById('{REGISTER_SUBMIT_ID}').onclick = function() {{
  var u = document.getElementById('{REGISTER_USERNAME_ID}').value;
  var p = document.getElementById('{REGISTER_PASSWORD_ID}').value;
  var users = getUsers();
  users[u] = p;
  saveUsers(users);
}};
document.getElementById('{LOGIN_SUBMIT_ID}').onclick = function() {{
  var u = document.getElementById('{LOGIN_USERNAME_ID}').value;
  document.getElementById('{AUTH_STATUS_ID}').textContent = 'Logged in as ' + u;
}};
</script>
</body></html>"""

SAMPLE_TRANSCRIPT = Path("examples/sample_transcript.txt").read_text()


def _real_browser_available() -> bool:
    """Whether a real Chromium binary can actually be launched here — not
    just whether the `playwright` package is importable. A pip install
    alone doesn't include the (large, separately-fetched) browser binary,
    so tests that need real execution skip cleanly rather than fail when
    only the package, not a usable browser, is present.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            kwargs = {"headless": True, "args": ["--no-sandbox"]}
            executable_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
            if executable_path:
                kwargs["executable_path"] = executable_path
            browser = p.chromium.launch(**kwargs)
            browser.close()
        return True
    except Exception:
        return False


REAL_BROWSER_AVAILABLE = _real_browser_available()
needs_real_browser = pytest.mark.skipif(not REAL_BROWSER_AVAILABLE, reason="no real Chromium binary available in this environment")


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
    from pipeline.llm import LLMProvider

    class BrokenProvider(LLMProvider):
        def complete(self, prompt):
            raise LLMError("OpenRouter rejected the API key (401 Unauthorized)")

    with pytest.raises(LLMError, match="401"):
        BrokenProvider().test_connection()


def test_openrouter_provider_surfaces_embedded_error_object_even_on_http_200():
    """OpenRouter can return HTTP 200 with an `error` key in the JSON body
    instead of a real completion — e.g. proxying an upstream provider
    outage (this exact case: Nvidia's Nemotron backend briefly overloaded).
    Without checking for this, it fell through to a generic "unexpected
    response shape" message with no indication anything upstream failed.
    """
    provider = OpenRouterProvider(Secrets("sk-test"), model="nvidia/nemotron-3-ultra-550b-a55b:free")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "id": "gen-1789245235-4RT7cCsJAHPvIUIOWjNV",
        "error": {
            "message": "Upstream error from Nvidia: Service temporarily overloaded",
            "code": 502,
            "metadata": {"error_type": "provider_unavailable"},
        },
    }
    with patch("requests.post", return_value=fake_response):
        with pytest.raises(LLMError, match="Service temporarily overloaded"):
            provider.complete("hello")


def test_openrouter_provider_sends_no_token_limit():
    """This app must never impose its own max_tokens/output cap — confirmed
    with the deployer that any such cap, however generous, could itself be
    the reason a reply gets cut short. The request body must not contain a
    max_tokens key at all.
    """
    provider = OpenRouterProvider(Secrets("sk-test"))
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
    }
    with patch("requests.post", return_value=fake_response) as mock_post:
        provider.complete("hello")
    assert "max_tokens" not in mock_post.call_args.kwargs["json"]


def test_openrouter_provider_raises_on_truncated_reply():
    """A `finish_reason` of "length" means the model or provider cut the
    reply off at its own limit — this app sends no max_tokens of its own,
    so there's no cap on this end to blame. Accepting the truncated content
    silently is exactly the "pipeline stops mid process" bug: a cut-off
    HTML file with no error to explain why.
    """
    provider = OpenRouterProvider(Secrets("sk-test"), model="some/small-model:free")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "<!doctype html><html>...cut off"}, "finish_reason": "length"}]
    }
    with patch("requests.post", return_value=fake_response):
        with pytest.raises(LLMError, match="cut the reply short"):
            provider.complete("hello")


def test_openrouter_provider_shows_actual_completion_tokens_on_truncation():
    """When OpenRouter's response includes a `usage` object, the error
    should show the model's actual completion_tokens count — real evidence
    of how far the model got, since this app no longer sends a cap to
    compare it against.
    """
    provider = OpenRouterProvider(Secrets("sk-test"), model="some/small-model:free")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "<!doctype html>...cut off"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 412, "total_tokens": 1312},
    }
    with patch("requests.post", return_value=fake_response):
        with pytest.raises(LLMError, match="412"):
            provider.complete("hello")


def test_aihubmix_provider_requires_an_api_key():
    with pytest.raises(LLMError, match="No AIHubMix API key configured"):
        AIHubMixProvider(api_key="")


def test_aihubmix_provider_returns_text_on_success():
    provider = AIHubMixProvider(api_key="test-key", model="ling-3.0-flash-free")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}]
    }
    with patch("requests.post", return_value=fake_response) as mock_post:
        assert provider.complete("hi") == "hello there"
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert mock_post.call_args.args[0] == "https://aihubmix.com/v1/chat/completions"
    assert mock_post.call_args.kwargs["json"]["model"] == "ling-3.0-flash-free"
    assert "max_tokens" not in mock_post.call_args.kwargs["json"]


def test_aihubmix_provider_reuses_the_same_error_handling_as_openrouter():
    """Both providers share one request/error-handling implementation, so
    a 401 from AIHubMix should read the same way OpenRouter's does, just
    naming AIHubMix instead.
    """
    provider = AIHubMixProvider(api_key="bad-key")
    fake_response = MagicMock(status_code=401, text="Invalid API Key")
    with patch("requests.post", return_value=fake_response):
        with pytest.raises(LLMError, match="AIHubMix rejected the API key \\(401"):
            provider.complete("hi")


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
    assert result.code.strip().lower().startswith("<!doctype html>")
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


def test_developer_includes_auth_contract_when_has_auth():
    log = []
    architecture = _dummy_architecture()
    architecture.has_auth = True
    DeveloperAgent(MockLLMProvider(), log).build(_dummy_requirements(), architecture)
    prompt = log[-1]["prompt"]
    assert REGISTER_USERNAME_ID in prompt
    assert LOGIN_SUBMIT_ID in prompt
    assert AUTH_STATUS_ID in prompt


def test_developer_omits_auth_contract_when_no_auth():
    log = []
    DeveloperAgent(MockLLMProvider(), log).build(_dummy_requirements(), _dummy_architecture())
    prompt = log[-1]["prompt"]
    assert REGISTER_USERNAME_ID not in prompt


def test_qa_reviewer_includes_auth_addendum_when_has_auth():
    log = []
    architecture = _dummy_architecture()
    architecture.has_auth = True
    QAReviewerAgent(MockLLMProvider(), log).review(_dummy_requirements(), architecture, "<html></html>")
    prompt = log[-1]["prompt"]
    assert "duplicate username" in prompt.lower()


def test_tester_agent_returns_test_report():
    log = []

    class ScriptedLLM(MockLLMProvider):
        def complete_json(self, prompt):
            if "STAGE: TESTING" in prompt:
                return {"passed": False, "notes": ["duplicate registration was silently accepted"]}
            return super().complete_json(prompt)

    report = PipelineTesterAgent(ScriptedLLM(), log).test(_dummy_requirements(), _dummy_architecture(), "print('hi')")
    assert report.passed is False
    assert "duplicate registration" in report.notes[0]
    assert log[-1]["stage"] == "testing"


@needs_real_browser
def test_run_browser_auth_test_passes_for_correct_contract_html():
    report = run_browser_auth_test(CORRECT_AUTH_HTML)
    assert report.executed is True
    assert report.passed is True


@needs_real_browser
def test_run_browser_auth_test_fails_when_wrong_password_is_accepted():
    report = run_browser_auth_test(BUGGY_AUTH_HTML)
    assert report.executed is True
    assert report.passed is False
    assert any("wrong password" in note.lower() for note in report.notes)


@needs_real_browser
def test_run_browser_auth_test_reports_missing_contract_ids():
    report = run_browser_auth_test("<!doctype html><html><body>no auth here</body></html>")
    assert report.executed is True
    assert report.passed is False


def test_run_browser_auth_test_degrades_gracefully_without_a_real_browser():
    if REAL_BROWSER_AVAILABLE:
        pytest.skip("a real browser is available in this environment; the degrade path isn't exercised")
    report = run_browser_auth_test(CORRECT_AUTH_HTML)
    assert report.executed is False
    assert report.passed is True


def test_orchestrator_defaults_to_two_iterations_now():
    class AlwaysFailLLM(MockLLMProvider):
        def complete_json(self, prompt):
            if "STAGE: QA" in prompt:
                return {"passed": False, "issues": ["still broken"]}
            return super().complete_json(prompt)

    orchestrator = Orchestrator(AlwaysFailLLM())  # no max_qa_iterations kwarg — pins the new default
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.iterations == 2


def test_orchestrator_skips_testing_when_no_auth():
    orchestrator = Orchestrator(MockLLMProvider(), max_qa_iterations=2)
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.test_reports
    for test_report in result.test_reports:
        assert test_report.executed is False
        assert test_report.passed is True
        assert "skipped" in test_report.notes[0].lower()


@needs_real_browser
def test_orchestrator_runs_real_browser_test_when_html_app_has_auth():
    class AuthAppLLM(MockLLMProvider):
        def complete_json(self, prompt):
            data = super().complete_json(prompt)
            if "STAGE: ARCHITECT" in prompt:
                data["has_auth"] = True
            return data

        def complete(self, prompt):
            if "STAGE: DEVELOPER" in prompt and REGISTER_USERNAME_ID in prompt:
                return CORRECT_AUTH_HTML
            return super().complete(prompt)

    orchestrator = Orchestrator(AuthAppLLM())
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.architecture.has_auth is True
    last_test = result.test_reports[-1]
    assert last_test.executed is True
    assert last_test.passed is True


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
