"""Streamlit UI: record/upload/paste a description, watch the SDLC pipeline
run stage by stage, preview and download the generated prototype.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from pipeline.llm import DEFAULT_MODEL, LLMError, MockLLMProvider, OpenRouterProvider
from pipeline.orchestrator import Orchestrator
from pipeline.secrets import Secrets
from pipeline.transcribe import PassthroughTranscriber, TranscriptionError, make_transcriber_for

st.set_page_config(page_title="Description → Code Generator", page_icon="assets/logo.png", layout="wide")


def _friendly_llm_error(exc: LLMError) -> str:
    msg = str(exc)
    if "401" in msg:
        return "That API key was rejected. Double-check it at openrouter.ai/keys."
    if "429" in msg:
        return "Rate-limited by OpenRouter. Wait a bit and try again, or switch models."
    if "not valid JSON" in msg:
        return "The model didn't reply in the expected format. Try again, or use a different model id."
    if "Request to OpenRouter failed" in msg:
        return "Couldn't reach OpenRouter — check your network connection."
    return msg

# ---------------------------------------------------------------------------
# Almost all styling (colors, fonts, borders, radii) lives in
# .streamlit/config.toml, which is the native, theme-switcher-preserving way
# to brand a Streamlit app — see that file's comment. The only custom CSS
# below is for two bespoke widgets Streamlit has no built-in equivalent for
# (pill badges, a step tracker), and even those pick their colors from
# st.context.theme.type so they adapt when a visitor changes theme.
# ---------------------------------------------------------------------------
_THEME = (st.context.theme.type if st.context.theme else None) or "dark"
if _THEME == "dark":
    C = dict(card="#1a1f2b", border="#2a3040", muted="#94a3b8", accent="#8b5cf6", accent_soft="#a78bfa", danger="#f59e0b")
else:
    C = dict(card="#f6f5fb", border="#e2e5eb", muted="#6b7280", accent="#7c3aed", accent_soft="#7c3aed", danger="#b45309")

EXAMPLES = {
    "Task tracker": Path("examples/sample_transcript.txt").read_text(),
    "Recipe box": Path("examples/sample_recipe.txt").read_text(),
    "Contact list": Path("examples/sample_contacts.txt").read_text(),
}

STAGE_NODES = [
    ("pm_kickoff", "Kickoff"),
    ("requirements", "Requirements"),
    ("architect", "Architect"),
    ("developer", "Developer"),
    ("qa", "QA Review"),
    ("pm_summary", "Sign-off"),
]

st.markdown(
    f"""
<style>
.badge-row {{ display:flex; gap:6px; flex-wrap:wrap; margin: 0.2rem 0 0.6rem 0; }}
.badge {{ background:{C['card']}; border:1px solid {C['border']}; color:{C['accent_soft']};
          border-radius:999px; padding:1px 10px; font-size:0.7rem; font-family:monospace; }}
.step-track {{ display:flex; justify-content:space-between; position:relative; margin: 0.6rem 0 0.7rem 0; }}
.step-track::before {{ content:''; position:absolute; top:14px; left:6%; right:6%; height:2px;
          background:{C['border']}; z-index:0; }}
.step {{ position:relative; z-index:1; display:flex; flex-direction:column; align-items:center;
          gap:3px; flex:1; }}
.step-icon {{ width:28px; height:28px; border-radius:50%; display:flex; align-items:center;
          justify-content:center; background:{C['card']}; border:2px solid {C['border']};
          color:{C['muted']}; font-size:0.85rem; }}
.step-label {{ font-size:0.68rem; color:{C['muted']}; text-align:center; }}
.step-done .step-icon {{ background:{C['accent']}; border-color:{C['accent']}; color:white; }}
.step-done .step-label {{ color:inherit; }}
.step-running .step-icon {{ border-color:{C['accent']}; color:{C['accent']};
          box-shadow: 0 0 0 3px {C['accent']}22; }}
.step-failed .step-icon {{ border-color:{C['danger']}; color:{C['danger']}; }}
.step-badge {{ font-size:0.68rem; color:{C['danger']}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("assets/logo.png", width=36)
    st.subheader("Model")
    demo_mode = st.toggle(
        "Demo mode (no API key)",
        value=True,
        help="Runs the same 5-agent pipeline against deterministic canned answers — no network call, no key.",
    )
    api_key = ""
    model_id = DEFAULT_MODEL
    if not demo_mode:
        api_key = st.text_input(
            "OpenRouter API key",
            type="password",
            help="Free at openrouter.ai/keys. Kept only in this browser session — never logged or saved.",
        )
        model_id = st.text_input(
            "Model id",
            value=DEFAULT_MODEL,
            help="Any OpenRouter model id. Check openrouter.ai/models for what's currently free.",
        )
        # Changing the key or model invalidates any earlier "known good" test.
        if st.session_state.get("tested_key") != (api_key, model_id):
            st.session_state.pop("connection_ok", None)
            st.session_state.pop("connection_error", None)

        test_col, status_col = st.columns([1, 2], vertical_alignment="center")
        with test_col:
            test_clicked = st.button("🔌 Test", disabled=not api_key, use_container_width=True)
        with status_col:
            if "connection_ok" in st.session_state:
                st.caption("✅ Connected" if st.session_state["connection_ok"] else "❌ Failed — see below")
        if test_clicked:
            with st.spinner("Testing…"):
                try:
                    OpenRouterProvider(Secrets(api_key), model=model_id).test_connection()
                    st.session_state["connection_ok"] = True
                    st.session_state.pop("connection_error", None)
                except LLMError as exc:
                    st.session_state["connection_ok"] = False
                    st.session_state["connection_error"] = _friendly_llm_error(exc)
                st.session_state["tested_key"] = (api_key, model_id)
            st.rerun()

        if st.session_state.get("connection_ok") is False and "connection_error" in st.session_state:
            st.error(st.session_state["connection_error"])

        st.caption("One OpenRouter key powers every agent below. Transcription runs locally and needs no key.")

    with st.expander("How it works"):
        st.markdown(
            "Five agents mirror a small software team:\n\n"
            "1. **Project Manager** — writes a brief, and later decides whether to ship or loop back\n"
            "2. **Requirements Analyst** — extracts entities, fields, and actions\n"
            "3. **Architect** — decides the technical approach\n"
            "4. **Developer** — writes the prototype\n"
            "5. **QA Reviewer** — checks it against requirements\n\n"
            "If QA finds issues, it goes back to the Developer (up to 2 tries) before the "
            "PM signs off."
        )

    st.caption("🌓 Use the **⋮ menu (top right) → Settings** to switch light/dark theme.")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 12], vertical_alignment="center")
with col_logo:
    st.image("assets/logo.png", width=40)
with col_title:
    st.title("Description → Code Generator")
    st.caption("A 5-agent SDLC pipeline turns a spoken description into a working prototype.")
st.markdown(
    "<div class='badge-row'>"
    + "".join(f"<span class='badge'>{label}</span>" for _, label in STAGE_NODES)
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
tab_upload, tab_record, tab_text = st.tabs(["Upload audio/video", "Record", "Paste text"])
transcript_source: tuple[str, object] | None = None

with tab_upload:
    uploaded = st.file_uploader("Audio or video file describing the app", type=["mp3", "wav", "m4a", "mp4", "mov", "webm"])
    if uploaded is not None:
        transcript_source = ("file", uploaded)

with tab_record:
    recorded = st.audio_input("Record a description")
    if recorded is not None:
        transcript_source = ("file", recorded)

with tab_text:
    # Streamlit widgets keep their own frontend value once mounted — clearing
    # session_state alone won't reset an existing textarea. Bumping the key's
    # generation suffix forces a fresh widget instance instead.
    st.session_state.setdefault("input_generation", 0)
    text_key = f"description_text_{st.session_state['input_generation']}"

    col_pick, col_load = st.columns([3, 1])
    with col_pick:
        example_choice = st.selectbox("Example descriptions", list(EXAMPLES.keys()), label_visibility="collapsed")
    with col_load:
        if st.button("Load example", use_container_width=True):
            st.session_state[text_key] = EXAMPLES[example_choice]
    pasted = st.text_area(
        "Type or paste a description",
        height=150,
        placeholder="So the app I want is basically a...",
        key=text_key,
    )
    if pasted:
        transcript_source = ("text", pasted)

generate = st.button("Generate", type="primary", disabled=transcript_source is None)

# ---------------------------------------------------------------------------
# Run the pipeline (persisted in session_state so it survives reruns caused
# by other widgets, e.g. clicking Download).
# ---------------------------------------------------------------------------


def _run_pipeline(transcript: str) -> None:
    llm = MockLLMProvider() if demo_mode else OpenRouterProvider(Secrets(api_key), model=model_id)
    orchestrator = Orchestrator(llm)

    tracker_box = st.empty()
    progress: dict[str, str] = {}
    iteration_count = {"n": 1}

    def node_key_for(stage: str) -> str:
        if stage.startswith("developer"):
            iteration_count["n"] = max(iteration_count["n"], int(stage.split("pass ")[1].rstrip(")")))
            return "developer"
        if stage.startswith("qa"):
            iteration_count["n"] = max(iteration_count["n"], int(stage.split("pass ")[1].rstrip(")")))
            return "qa"
        return stage

    def render_tracker() -> None:
        icons = {"pending": "○", "running": "◐", "done": "✓", "failed": "↺"}
        parts = []
        for key, label in STAGE_NODES:
            status = progress.get(key, "pending")
            badge = (
                f"<span class='step-badge'> ×{iteration_count['n']}</span>"
                if key in ("developer", "qa") and iteration_count["n"] > 1
                else ""
            )
            parts.append(
                f"<div class='step step-{status}'><div class='step-icon'>{icons[status]}</div>"
                f"<div class='step-label'>{label}{badge}</div></div>"
            )
        tracker_box.markdown(f"<div class='step-track'>{''.join(parts)}</div>", unsafe_allow_html=True)

    def on_stage(stage: str, stage_status: str) -> None:
        key = node_key_for(stage)
        progress[key] = "done" if stage_status == "done" else ("failed" if stage_status == "failed" else "running")
        render_tracker()

    render_tracker()
    try:
        result = orchestrator.run(transcript, on_stage=on_stage)
    except LLMError as exc:
        st.error(_friendly_llm_error(exc))
        return
    st.session_state["result"] = result
    st.session_state["last_transcript"] = transcript


if generate:
    if transcript_source is None:
        st.error("Provide a description first: upload a file, record one, or paste text.")
        st.stop()
    if not demo_mode and not api_key:
        st.error("Enter an OpenRouter API key in the sidebar, or turn on Demo mode.")
        st.stop()

    kind, payload = transcript_source
    status_label = "Transcribing..." if kind == "text" else "Transcribing (first use downloads the speech model — can take a minute)..."
    with st.status(status_label, expanded=False) as status:
        try:
            if kind == "text":
                transcript = PassthroughTranscriber.from_text(payload)
            else:
                audio_bytes = payload.getvalue()
                if len(audio_bytes) < 1000:
                    raise TranscriptionError(
                        "No audio was captured (the recording is empty). Check that your "
                        "browser has microphone access for this site and the right input "
                        "device selected, then try recording again — or paste the "
                        "description as text instead."
                    )
                suffix = Path(getattr(payload, "name", "recording.wav")).suffix or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                transcriber = make_transcriber_for(tmp_path)
                transcript = transcriber.transcribe(tmp_path)
                if not transcript.strip():
                    raise TranscriptionError(
                        "Transcription produced no text. The recording may be silent — "
                        "check your microphone and try again, or paste the description as text."
                    )
        except TranscriptionError as exc:
            status.update(label="Transcription failed", state="error", expanded=True)
            st.error(str(exc))
            st.stop()
        status.update(label="Transcribed", state="complete")

    with st.expander("Transcript", expanded=False):
        st.text(transcript)

    _run_pipeline(transcript)

if st.session_state.pop("regenerate_requested", False) and "last_transcript" in st.session_state:
    _run_pipeline(st.session_state["last_transcript"])

# ---------------------------------------------------------------------------
# Result (rendered from session_state so it survives unrelated reruns)
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]

    st.success(result.summary, icon="✅")

    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("Requirements", expanded=False):
            st.json(result.requirements.to_dict())
        with st.expander("Architecture", expanded=False):
            st.json(result.architecture.to_dict())
        with st.container(border=True):
            st.caption(f"QA history — {result.iterations} iteration(s)")
            for i, qa in enumerate(result.qa_reports, start=1):
                st.write(f"Pass {i}: {'✅ passed' if qa.passed else '⚠️ issues found'}")
                for issue in qa.issues:
                    st.caption(f"– {issue}")
    with col_right:
        st.caption("Live preview")
        st.components.v1.html(result.html, height=340, scrolling=True)
        dl_col, regen_col, reset_col = st.columns(3)
        with dl_col:
            st.download_button("⬇ Download", data=result.html, file_name="index.html", mime="text/html", use_container_width=True)
        with regen_col:
            if st.button("🔁 Regenerate", use_container_width=True):
                st.session_state["regenerate_requested"] = True
                st.rerun()
        with reset_col:
            if st.button("↺ Start over", use_container_width=True):
                st.session_state.pop("result", None)
                st.session_state.pop("last_transcript", None)
                st.session_state["input_generation"] = st.session_state.get("input_generation", 0) + 1
                st.rerun()
        with st.expander("View source"):
            st.code(result.html, language="html")

    with st.expander("Prompt log (every prompt sent to the model)"):
        for entry in result.prompt_log:
            st.write(f"**{entry['stage']}**")
            st.code(entry["prompt"])

st.divider()
st.caption(
    "Built as a multi-agent SDLC pipeline — no Claude, one free OpenRouter key, local Whisper "
    "transcription. [Source on GitHub](https://github.com/Mahijith/Description-to-code-generator)"
)
