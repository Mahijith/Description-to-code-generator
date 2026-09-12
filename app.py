"""Streamlit UI: record or upload an audio/video description, watch the
SDLC pipeline run stage by stage, preview and download the generated
prototype.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from pipeline.llm import DEFAULT_MODEL, LLMError, OpenRouterProvider
from pipeline.orchestrator import Orchestrator
from pipeline.secrets import Secrets
from pipeline.transcribe import TranscriptionError, make_transcriber_for

MAX_UPLOAD_BYTES = 19_500_000  # Groq's API stops accepting files above ~19.5MB in practice
MIME_TYPES = {"html": "text/html", "py": "text/x-python", "js": "text/javascript"}

st.set_page_config(page_title="Description → Code Generator", page_icon="assets/logo.png", layout="wide")


def _bridge_secret_to_env(name: str) -> None:
    """Copies a Streamlit Cloud secret into the plain env-var convention
    pipeline/*.py and cli.py use, so those modules stay framework-agnostic
    and behave the same from the CLI. st.secrets raises (rather than
    behaving like an empty dict) when no secrets.toml exists at all — e.g.
    a fresh deployment with nothing configured yet — which isn't an error
    here, it just means this particular secret isn't set.
    """
    if name in os.environ:
        return
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value:
        os.environ[name] = value


_bridge_secret_to_env("GROQ_API_KEY")
_bridge_secret_to_env("OPENROUTER_API_KEY")


def _friendly_llm_error(exc: LLMError) -> str:
    msg = str(exc)
    if "No OpenRouter API key configured" in msg:
        return (
            "This app isn't configured with an OpenRouter API key yet. If you're the "
            "deployer: add OPENROUTER_API_KEY under the app's Settings → Secrets."
        )
    if "401" in msg:
        return f"This app's configured API key was rejected by OpenRouter ({msg}). If you're the deployer: check it at openrouter.ai/keys."
    if "429" in msg:
        return "OpenRouter rate-limited this app (every visitor shares one key). Wait a bit and try again."
    if "OpenRouter upstream error" in msg:
        return f"The model's provider is temporarily unavailable ({msg}). This isn't a key/config problem — wait a moment and try again."
    if "not valid JSON" in msg:
        return "The model didn't reply in the expected format. Try again."
    if "Request to OpenRouter failed" in msg:
        return "Couldn't reach OpenRouter right now. Try again shortly."
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
# Hero
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 12], vertical_alignment="center")
with col_logo:
    st.image("assets/logo.png", width=40)
with col_title:
    st.title("Description → Code Generator")
    st.caption("A 5-agent SDLC pipeline turns an audio or video description into a working prototype.")
st.markdown(
    "<div class='badge-row'>"
    + "".join(f"<span class='badge'>{label}</span>" for _, label in STAGE_NODES)
    + "</div>",
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.markdown(
        "Five agents mirror a small software team:\n\n"
        "1. **Project Manager** — writes a brief, and later decides whether to ship or loop back\n"
        "2. **Requirements Analyst** — extracts entities, fields, and actions\n"
        "3. **Architect** — picks the best language for this app and the technical approach\n"
        "4. **Developer** — writes the prototype\n"
        "5. **QA Reviewer** — checks it against requirements\n\n"
        "If QA finds issues, it goes back to the Developer (up to 2 tries) before the "
        "PM signs off. Powered by one shared OpenRouter key and one shared Groq key, both "
        "configured by whoever deployed this app — nothing to enter here."
    )

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
tab_upload, tab_record = st.tabs(["Upload audio/video", "Record"])
audio_payload = None

with tab_upload:
    uploaded = st.file_uploader(
        "Audio or video file describing the app",
        type=["mp3", "wav", "m4a", "mp4", "mov", "webm"],
        help=f"Max {MAX_UPLOAD_BYTES / 1_000_000:.1f}MB — Groq's transcription API rejects larger files.",
    )
    if uploaded is not None:
        audio_payload = uploaded

with tab_record:
    recorded = st.audio_input("Record a description")
    if recorded is not None:
        audio_payload = recorded

generate = st.button("Generate", type="primary", disabled=audio_payload is None)

# ---------------------------------------------------------------------------
# Run the pipeline (persisted in session_state so it survives reruns caused
# by other widgets, e.g. clicking Download).
# ---------------------------------------------------------------------------


def _run_pipeline(transcript: str) -> None:
    try:
        llm = OpenRouterProvider(Secrets(os.environ.get("OPENROUTER_API_KEY")), model=DEFAULT_MODEL)
        orchestrator = Orchestrator(llm)
    except LLMError as exc:
        st.error(_friendly_llm_error(exc))
        return

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
    if audio_payload is None:
        st.error("Provide a description first: upload a file or record one.")
        st.stop()

    with st.status("Transcribing via Groq...", expanded=False) as status:
        try:
            audio_bytes = audio_payload.getvalue()
            if len(audio_bytes) < 1000:
                raise TranscriptionError(
                    "No audio was captured (the recording is empty). Check that your "
                    "browser has microphone access for this site and the right input "
                    "device selected, then try recording again."
                )
            if len(audio_bytes) > MAX_UPLOAD_BYTES:
                raise TranscriptionError(
                    f"That file is {len(audio_bytes) / 1_000_000:.1f}MB, over the "
                    f"{MAX_UPLOAD_BYTES / 1_000_000:.1f}MB limit Groq's API accepts. Try a "
                    "shorter recording, or compress the file first."
                )
            suffix = Path(getattr(audio_payload, "name", "recording.wav")).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                transcriber = make_transcriber_for(tmp_path)
                transcript = transcriber.transcribe(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if not transcript.strip():
                raise TranscriptionError(
                    "Transcription produced no text. The recording may be silent — "
                    "check your microphone and try again."
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
        language = result.architecture.language
        extension = result.architecture.file_extension
        if language == "html":
            st.caption("Live preview")
            st.components.v1.html(result.code, height=340, scrolling=True)
        else:
            st.caption(f"Generated {language} source")
            st.code(result.code, language=language, height=340)
        dl_col, regen_col, reset_col = st.columns(3)
        with dl_col:
            st.download_button(
                "⬇ Download",
                data=result.code,
                file_name=f"app.{extension}",
                mime=MIME_TYPES.get(extension, "text/plain"),
                use_container_width=True,
            )
        with regen_col:
            if st.button("🔁 Regenerate", use_container_width=True):
                st.session_state["regenerate_requested"] = True
                st.rerun()
        with reset_col:
            if st.button("↺ Start over", use_container_width=True):
                st.session_state.pop("result", None)
                st.session_state.pop("last_transcript", None)
                st.rerun()
        if language == "html":
            with st.expander("View source"):
                st.code(result.code, language="html")

    with st.expander("Prompt log (every prompt sent to the model)"):
        for entry in result.prompt_log:
            st.write(f"**{entry['stage']}**")
            st.code(entry["prompt"])

st.divider()
st.caption(
    "Built as a multi-agent SDLC pipeline — no Claude, powered by one shared OpenRouter key and "
    "Groq-hosted Whisper transcription. [Source on GitHub](https://github.com/Mahijith/Description-to-code-generator)"
)
