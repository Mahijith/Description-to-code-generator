import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.agents import ArchitectAgent, DeveloperAgent, ProjectManagerAgent, QAReviewerAgent, RequirementsAnalystAgent
from pipeline.auth_contract import (
    AUTH_STATUS_ID,
    LOGIN_PASSWORD_ID,
    LOGIN_SUBMIT_ID,
    LOGIN_USERNAME_ID,
    REGISTER_PASSWORD_ID,
    REGISTER_SUBMIT_ID,
    REGISTER_USERNAME_ID,
)
from pipeline.browser_tester import run_browser_functional_test
from pipeline.llm import AIHubMixProvider, LLMError, MockLLMProvider, OpenRouterProvider
from pipeline.orchestrator import Orchestrator
from pipeline.preview import isolate_local_storage
from pipeline.schema import Entity, Field, Requirements
from pipeline.secrets import Secrets, mask_key
from pipeline.transcribe import GroqWhisperTranscriber, PassthroughTranscriber, TranscriptionError, make_transcriber_for

SAMPLE_TRANSCRIPT = Path("examples/sample_transcript.txt").read_text()

TASK_TRACKER_REQUIREMENTS = Requirements(
    app_name="Task Tracker",
    description="Track personal tasks with due dates and priority.",
    entities=[
        Entity(
            name="Task",
            fields=[
                Field(name="title", type="text"),
                Field(name="description", type="text"),
                Field(name="due_date", type="date"),
                Field(name="priority", type="select", options=["Low", "Medium", "High"]),
                Field(name="completed", type="boolean"),
            ],
        )
    ],
    actions=["add", "edit", "delete", "complete", "filter"],
    filters=["priority", "completed"],
    screens=[],
)

# MockLLMProvider's own canned Developer output already follows the CRUD
# contract for exactly this Requirements shape (see pipeline/llm.py) — reuse
# it directly instead of hand-duplicating a near-identical fixture.
GOLDEN_CRUD_HTML = MockLLMProvider().complete("STAGE: DEVELOPER")

BUGGY_CRUD_DELETE_HTML = GOLDEN_CRUD_HTML.replace(
    "save(load().filter(x => x.id !== t.id)); render();",
    "/* delete intentionally broken for a test */",
)

_AUTH_MARKUP_AND_SCRIPT = f"""
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
"""

# One combined "everything correct" fixture (accounts + CRUD), built by
# splicing the auth markup/script into the already-correct CRUD fixture —
# this is what a real generated has_auth app should look like. Bug-specific
# variants below start from this and break exactly one thing each, so a
# test failure is attributable to that one thing, not fixture drift.
GOLDEN_COMBINED_HTML = GOLDEN_CRUD_HTML.replace(
    "<h1>Task Tracker</h1>", "<h1>Task Tracker</h1>" + _AUTH_MARKUP_AND_SCRIPT
)

BUGGY_LOGIN_HTML = GOLDEN_COMBINED_HTML.replace(
    "document.getElementById('auth-status').textContent = (users[u] === p) ? ('Logged in as ' + u) : '';",
    "document.getElementById('auth-status').textContent = 'Logged in as ' + u;",
)

# An app with neither an entity nor accounts — exercises the generic
# smoke-test tier in browser_tester.py, which is all that's left when
# neither of the fixed-id contracts applies.
NO_ENTITY_REQUIREMENTS = Requirements(
    app_name="Quick Calculator",
    description="Add, subtract, multiply, or divide two numbers.",
    entities=[],
    actions=[],
    filters=[],
    features=[
        "Accept two numbers and an operator (+, -, *, /) and show the result",
        "Show a clear message for divide-by-zero instead of crashing",
    ],
    screens=[],
)

CALCULATOR_HTML = """<!doctype html><html><body>
<h1>Calculator</h1>
<input id="a" type="number"><select id="op"><option>+</option><option>-</option><option>*</option><option>/</option></select><input id="b" type="number">
<button id="compute">=</button>
<div id="result"></div>
<script>
document.getElementById("compute").onclick = function() {
  var a = parseFloat(document.getElementById("a").value);
  var b = parseFloat(document.getElementById("b").value);
  var op = document.getElementById("op").value;
  var out = document.getElementById("result");
  if (op === "/" && b === 0) { out.textContent = "Cannot divide by zero"; return; }
  var r = op === "+" ? a + b : op === "-" ? a - b : op === "*" ? a * b : a / b;
  out.textContent = String(r);
};
</script>
</body></html>"""

BROKEN_ON_LOAD_HTML = """<!doctype html><html><body>
<h1>Broken</h1>
<script>thisFunctionDoesNotExist();</script>
</body></html>"""


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


def test_developer_includes_crud_contract_with_real_field_ids():
    log = []
    DeveloperAgent(MockLLMProvider(), log).build(TASK_TRACKER_REQUIREMENTS, _dummy_architecture())
    prompt = log[-1]["prompt"]
    assert "field-title" in prompt
    assert "field-due-date" in prompt
    assert "filter-priority" in prompt
    assert "field-completed" not in prompt  # boolean fields are excluded from the add-form
    normalized = " ".join(prompt.lower().split())
    assert "visible without any navigation" in normalized


def test_qa_reviewer_includes_crud_addendum():
    log = []
    QAReviewerAgent(MockLLMProvider(), log).review(TASK_TRACKER_REQUIREMENTS, _dummy_architecture(), "<html></html>")
    prompt = log[-1]["prompt"]
    assert "add-form" in prompt
    assert "item-list" in prompt


def test_developer_omits_crud_contract_when_no_entity():
    log = []
    DeveloperAgent(MockLLMProvider(), log).build(NO_ENTITY_REQUIREMENTS, _dummy_architecture())
    prompt = log[-1]["prompt"].lower()
    assert "add-form" not in prompt
    assert "do not bolt on a generic" in prompt


def test_qa_reviewer_omits_entity_questions_when_no_entity():
    log = []
    QAReviewerAgent(MockLLMProvider(), log).review(NO_ENTITY_REQUIREMENTS, _dummy_architecture(), "<html></html>")
    prompt = log[-1]["prompt"].lower()
    assert "primary entity" not in prompt
    assert "add-form" not in prompt  # crud_contract.qa_prompt_addendum shouldn't fire either
    assert "capability listed in 'features'" in prompt


def test_crud_contract_slugify_and_ids():
    from pipeline import crud_contract

    assert crud_contract.slugify("Due Date") == "due-date"
    assert crud_contract.field_id("due_date") == "field-due-date"
    assert crud_contract.filter_id("priority") == "filter-priority"


def test_crud_contract_creatable_fields_excludes_booleans():
    from pipeline import crud_contract

    names = {f.name for f in crud_contract.creatable_fields(TASK_TRACKER_REQUIREMENTS)}
    assert "completed" not in names
    assert "title" in names


def test_crud_contract_first_testable_filter_only_matches_select_with_options():
    from pipeline import crud_contract

    result = crud_contract.first_testable_filter(TASK_TRACKER_REQUIREMENTS)
    assert result is not None
    name, field = result
    assert name == "priority"
    assert field.type == "select"


@needs_real_browser
def test_run_browser_functional_test_passes_when_everything_is_correct():
    report = run_browser_functional_test(TASK_TRACKER_REQUIREMENTS, True, GOLDEN_COMBINED_HTML)
    assert report.executed is True
    assert report.passed is True


@needs_real_browser
def test_run_browser_functional_test_fails_when_wrong_password_is_accepted():
    report = run_browser_functional_test(TASK_TRACKER_REQUIREMENTS, True, BUGGY_LOGIN_HTML)
    assert report.executed is True
    assert report.passed is False
    assert any("must be rejected" in note.lower() for note in report.notes)


@needs_real_browser
def test_run_browser_functional_test_fails_when_delete_is_broken():
    report = run_browser_functional_test(TASK_TRACKER_REQUIREMENTS, False, BUGGY_CRUD_DELETE_HTML)
    assert report.executed is True
    assert report.passed is False
    assert any("still present" in note.lower() for note in report.notes)


@needs_real_browser
def test_run_browser_functional_test_reports_missing_auth_ids():
    report = run_browser_functional_test(TASK_TRACKER_REQUIREMENTS, True, "<!doctype html><html><body>empty</body></html>")
    assert report.executed is True
    assert report.passed is False


@needs_real_browser
def test_run_browser_functional_test_reports_missing_crud_ids():
    report = run_browser_functional_test(TASK_TRACKER_REQUIREMENTS, False, "<!doctype html><html><body>empty</body></html>")
    assert report.executed is True
    assert report.passed is False


@needs_real_browser
def test_run_browser_functional_test_reports_a_clear_reason_when_add_form_is_hidden():
    # A real generated app hid its primary entity's form behind a
    # multi-screen nav (the default "screen" wasn't the one with the
    # form) — the test used to just time out waiting to fill a hidden
    # field, surfacing as an opaque "Browser test crashed" instead of a
    # clear, specific reason. This pins the fix: a present-but-hidden
    # #add-form is now reported immediately and specifically.
    html = """<!doctype html><html><body>
<div id="screen-a" style="display:block"><p>Screen A</p></div>
<div id="screen-b" style="display:none">
  <form id="add-form"><input id="field-title"><button id="add-submit">Add</button></form>
  <ul id="item-list"></ul>
</div>
</body></html>"""
    report = run_browser_functional_test(TASK_TRACKER_REQUIREMENTS, False, html)
    assert report.executed is True
    assert report.passed is False
    assert any("exists but isn't visible" in note.lower() for note in report.notes)


@needs_real_browser
def test_run_browser_functional_test_skips_crud_gracefully_with_no_creatable_fields():
    boolean_only = Requirements(
        app_name="Checklist",
        description="d",
        entities=[Entity(name="Item", fields=[Field(name="done", type="boolean")])],
        actions=["complete"],
        filters=[],
        screens=[],
    )
    # A real add-form always has visible input fields; give this one a
    # visible label so it has an actual layout box (an empty <form> has
    # zero size, which Playwright's is_visible() correctly treats as not
    # visible — a real, if incidental, gap in this test fixture, not in
    # the visibility check itself).
    html = '<!doctype html><html><body><form id="add-form">not empty</form></body></html>'
    report = run_browser_functional_test(boolean_only, False, html)
    assert report.executed is True
    assert report.passed is True
    assert any("wasn't executed" in note.lower() for note in report.notes)


@needs_real_browser
def test_run_browser_functional_test_uses_smoke_test_when_no_entity_and_no_auth():
    report = run_browser_functional_test(NO_ENTITY_REQUIREMENTS, False, CALCULATOR_HTML)
    assert report.executed is True
    assert report.passed is True
    assert any("visible content" in note.lower() for note in report.notes)


@needs_real_browser
def test_run_browser_functional_test_smoke_test_catches_uncaught_errors():
    report = run_browser_functional_test(NO_ENTITY_REQUIREMENTS, False, BROKEN_ON_LOAD_HTML)
    assert report.executed is True
    assert report.passed is False
    assert any("uncaught error" in note.lower() for note in report.notes)


def test_run_browser_functional_test_degrades_gracefully_without_a_real_browser():
    if REAL_BROWSER_AVAILABLE:
        pytest.skip("a real browser is available in this environment; the degrade path isn't exercised")
    report = run_browser_functional_test(TASK_TRACKER_REQUIREMENTS, True, GOLDEN_COMBINED_HTML)
    assert report.executed is False
    assert report.passed is True


def test_orchestrator_defaults_to_three_iterations_now():
    class AlwaysFailLLM(MockLLMProvider):
        def complete_json(self, prompt):
            if "STAGE: QA" in prompt:
                return {"passed": False, "issues": ["still broken"]}
            return super().complete_json(prompt)

    orchestrator = Orchestrator(AlwaysFailLLM())  # no max_qa_iterations kwarg — pins the new default
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.iterations == 3


def test_requirements_has_buildable_scope():
    empty = Requirements(app_name="X", description="d")
    assert empty.has_buildable_scope is False

    with_entity = Requirements(app_name="X", description="d", entities=[Entity(name="Task")])
    assert with_entity.has_buildable_scope is True

    with_action = Requirements(app_name="X", description="d", actions=["add"])
    assert with_action.has_buildable_scope is True

    with_feature = Requirements(app_name="X", description="d", features=["step through code"])
    assert with_feature.has_buildable_scope is True


def test_requirements_analyst_prompt_distinguishes_no_scope_from_vague():
    log: list[dict] = []
    RequirementsAnalystAgent(MockLLMProvider(), log).extract(SAMPLE_TRANSCRIPT, _dummy_brief())
    prompt = log[-1]["prompt"]
    normalized = " ".join(prompt.lower().split())
    assert "leave \"entities\", \"actions\", and \"features\" all empty" in normalized
    assert "vague-but-real request" in normalized
    # Coherent, detailed, unrelated content (a story/anecdote/review) must
    # be called out as its own no-scope case, distinct from noise like a
    # greeting — a model can otherwise reverse-engineer an entity out of
    # whatever nouns happen to be present (e.g. treating an offhand
    # restaurant recommendation as a request for a restaurant/review app).
    assert "panda express" in normalized
    assert "reverse-engineer an entity" in normalized


def test_orchestrator_skips_pipeline_when_no_buildable_scope():
    class NoScopeLLM(MockLLMProvider):
        def complete_json(self, prompt):
            if "STAGE: REQUIREMENTS" in prompt:
                return {
                    "app_name": "Untitled",
                    "description": "Just a greeting, nothing to build.",
                    "entities": [],
                    "actions": [],
                    "filters": [],
                    "features": [],
                    "screens": [],
                }
            return super().complete_json(prompt)

    orchestrator = Orchestrator(NoScopeLLM())
    stages: list[str] = []
    result = orchestrator.run("hello", on_stage=lambda s, status: stages.append(s))

    assert result.requirements.has_buildable_scope is False
    assert result.architecture is None
    assert result.code == ""
    assert result.summary
    assert not any(s.startswith(("architect", "developer", "qa", "testing", "pm_summary")) for s in stages)


def test_orchestrator_skips_pipeline_for_coherent_but_unrelated_content():
    """Distinct from the "hello" case: this transcript is a real, detailed
    sentence — it just isn't a request to build anything. Proves the
    short-circuit works the same way once the Requirements stage reports
    empty scope, regardless of why (noise vs. off-topic content)."""

    restaurant_review = (
        "I know a restaurant down my lane. It tastes very good. "
        "It's a Chinese restaurant named Panda Express."
    )

    class NoScopeLLM(MockLLMProvider):
        def complete_json(self, prompt):
            if "STAGE: REQUIREMENTS" in prompt:
                return {
                    "app_name": "Untitled",
                    "description": "An anecdote about a restaurant, not a request to build anything.",
                    "entities": [],
                    "actions": [],
                    "filters": [],
                    "features": [],
                    "screens": [],
                }
            return super().complete_json(prompt)

    orchestrator = Orchestrator(NoScopeLLM())
    stages: list[str] = []
    result = orchestrator.run(restaurant_review, on_stage=lambda s, status: stages.append(s))

    assert result.requirements.has_buildable_scope is False
    assert result.architecture is None
    assert result.code == ""
    assert not any(s.startswith(("architect", "developer", "qa", "testing", "pm_summary")) for s in stages)


@needs_real_browser
def test_orchestrator_runs_crud_test_even_without_auth():
    orchestrator = Orchestrator(MockLLMProvider())
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.architecture.has_auth is False
    last_test = result.test_reports[-1]
    assert last_test.executed is True
    assert last_test.passed is True


@needs_real_browser
def test_orchestrator_builds_non_entity_app_without_forcing_crud():
    class CalculatorLLM(MockLLMProvider):
        def complete_json(self, prompt):
            if "STAGE: REQUIREMENTS" in prompt:
                return {
                    "app_name": NO_ENTITY_REQUIREMENTS.app_name,
                    "description": NO_ENTITY_REQUIREMENTS.description,
                    "entities": [],
                    "actions": [],
                    "filters": [],
                    "features": NO_ENTITY_REQUIREMENTS.features,
                    "screens": [],
                }
            return super().complete_json(prompt)

        def complete(self, prompt):
            if "STAGE: DEVELOPER" in prompt:
                return CALCULATOR_HTML
            return super().complete(prompt)

    orchestrator = Orchestrator(CalculatorLLM())
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.requirements.primary_entity is None
    last_test = result.test_reports[-1]
    assert last_test.executed is True
    assert last_test.passed is True


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
                return GOLDEN_COMBINED_HTML
            return super().complete(prompt)

    orchestrator = Orchestrator(AuthAppLLM())
    result = orchestrator.run(SAMPLE_TRANSCRIPT)
    assert result.architecture.has_auth is True
    last_test = result.test_reports[-1]
    assert last_test.executed is True
    assert last_test.passed is True


_STORAGE_PROBE_HTML = """<!doctype html><html><body>
<div id="out"></div>
<script>
document.getElementById("out").textContent = "sees: " + localStorage.getItem("shared_key");
localStorage.setItem("shared_key", "hello");
</script>
</body></html>"""


@needs_real_browser
def test_isolate_local_storage_keeps_different_run_ids_apart(tmp_path):
    from playwright.sync_api import sync_playwright

    html_a = isolate_local_storage(_STORAGE_PROBE_HTML, "run-A")
    html_b = isolate_local_storage(_STORAGE_PROBE_HTML, "run-B")
    path_a = tmp_path / "a.html"
    path_b = tmp_path / "b.html"
    path_a.write_text(html_a)
    path_b.write_text(html_b)

    kwargs = {"headless": True, "args": ["--no-sandbox"]}
    executable_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if executable_path:
        kwargs["executable_path"] = executable_path

    with sync_playwright() as p:
        browser = p.chromium.launch(**kwargs)
        page = browser.new_page()
        page.goto(path_a.as_uri())
        assert "sees: null" in page.locator("#out").inner_text()
        page.goto(path_b.as_uri())
        assert "sees: null" in page.locator("#out").inner_text()  # different run_id — doesn't see A's write
        page.goto(path_a.as_uri())
        assert "sees: hello" in page.locator("#out").inner_text()  # same run_id as before — persists normally
        browser.close()


def test_isolate_local_storage_prepends_before_everything():
    html = '<!doctype html><html><body class="x" data-y="1"><p>hi</p></body></html>'
    result = isolate_local_storage(html, "abc")
    assert result.startswith("<script>")
    assert result.index("<script>") < result.index("<!doctype html>")


def test_isolate_local_storage_works_with_no_body_tag():
    html = "<p>no body tag here</p>"
    result = isolate_local_storage(html, "abc")
    assert result.startswith("<script>")
    assert "no body tag here" in result


def test_isolate_local_storage_is_not_confused_by_a_fake_body_tag_in_a_script_string():
    # A generated app building HTML via a JS template string can easily
    # contain the literal substring "<body" somewhere that isn't the real
    # tag — searching the text for it (the old implementation) landed the
    # shim inside that string instead, corrupting the script. Confirmed
    # directly before fixing; this pins the fix by construction: the new
    # implementation never searches for "<body" at all, so this can't
    # recur regardless of where such a substring appears.
    html = '<html><head><script>var tpl = "<body class=nope>fake</body>";</script></head><body><p>real</p></body></html>'
    result = isolate_local_storage(html, "abc")
    assert result.startswith("<script>")
    assert 'var tpl = "<body class=nope>fake</body>";' in result


def _dummy_brief():
    from pipeline.schema import ProjectBrief

    return ProjectBrief(goal="g", scope="s", out_of_scope="o", success_criteria=[])


def _dummy_requirements():
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
