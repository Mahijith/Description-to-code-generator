"""Streamlit UI: record or upload an audio/video description, watch the
SDLC pipeline run stage by stage, preview and download the generated
prototype.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from pipeline.llm import DEFAULT_MODEL, LLMError, OpenRouterProvider
from pipeline.orchestrator import Orchestrator
from pipeline.preview import isolate_local_storage
from pipeline.secrets import Secrets
from pipeline.transcribe import TranscriptionError, make_transcriber_for

MAX_UPLOAD_BYTES = 19_500_000  # Groq's API stops accepting files above ~19.5MB in practice
MIME_TYPES = {"html": "text/html", "py": "text/x-python", "js": "text/javascript"}

st.set_page_config(page_title="AV2A", page_icon="assets/logo.png", layout="wide")


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
    # The browser only ever sees a generic message (deliberately, so it never
    # names a platform or exposes a key) — this is the one place the real
    # cause survives, for whoever owns the app to check in its server logs.
    print(f"[app] LLM error: {msg}", file=sys.stderr)
    if "API key configured" in msg:
        return (
            "This app isn't configured yet. If you're the deployer, please add the "
            "required credentials under the app's Settings → Secrets."
        )
    if "401" in msg:
        return "This app's configured credentials were rejected. If you're the deployer, please verify them."
    if "429" in msg:
        return "This app is being rate-limited right now (shared usage). Wait a bit and try again."
    if "upstream error" in msg:
        return "The generation service is temporarily unavailable right now. This isn't a configuration problem — wait a moment and try again."
    if "not valid JSON" in msg:
        return "The model didn't reply in the expected format. Try again."
    if "Request to" in msg and "failed" in msg:
        return "Couldn't reach the generation service right now. Try again shortly."
    return "Something went wrong generating your app. Please try again, or contact the app owner if this keeps happening."


def _friendly_transcription_error(exc: TranscriptionError) -> str:
    msg = str(exc)
    lowered = msg.lower()
    if "groq" not in lowered and "api key" not in lowered:
        return msg  # already a generic, platform-agnostic message we wrote ourselves
    print(f"[app] Transcription error: {msg}", file=sys.stderr)
    if "429" in msg or "rate-limited" in lowered:
        return "This app is being rate-limited right now (shared usage). Wait a bit and try again."
    if "too large" in lowered:
        return "That recording is too large for this app to transcribe. Try a shorter recording, or compress the file first."
    if "request to" in lowered and "failed" in lowered:
        return "Couldn't reach the transcription service right now. Try again shortly."
    return "This app isn't configured correctly right now. If you're the deployer, please check its configured credentials."

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
    ("qa", "Code Review"),
    ("testing", "Testing"),
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
    st.title("AV2A(Audio/Video2Application)")
    st.caption("A 5-agent SDLC pipeline plus a real browser-testing stage turns an audio or video description into a working prototype.")
st.markdown(
    "<div class='badge-row'>"
    + "".join(f"<span class='badge'>{label}</span>" for _, label in STAGE_NODES)
    + "</div>",
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.markdown(
        "Five agents mirror a small software team, plus a real automated Testing stage:\n\n"
        "1. **Project Manager** — writes a brief, and signs off at the end\n"
        "2. **Requirements Analyst** — extracts whatever the app actually needs: entities "
        "and actions for a records-list app, freeform features for anything else (a tool, "
        "a game, a converter), or a mix\n"
        "3. **Architect** — plans the technical approach and whether it needs user accounts\n"
        "4. **Developer** — builds it, in whatever shape actually fits — no fixed template\n"
        "5. **Code Reviewer** — checks it against requirements and reports what it finds\n"
        "6. **Testing** — actually runs the app in a real browser: registration/login when it "
        "has accounts, add/edit/delete/filter when it manages records, or a load/render check "
        "for anything else\n\n"
        "Up to three build → review → test passes — if Code Review or Testing finds some issues, "
        "the Developer gets up to two more chances to fix it, then the result is produced."
        "Click Regenerate for a fresh attempt. Access is already "
        "preconfigured."
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
        help=f"Max {MAX_UPLOAD_BYTES / 1_000_000:.1f}MB — larger files are rejected.",
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
        if stage.startswith("testing"):
            iteration_count["n"] = max(iteration_count["n"], int(stage.split("pass ")[1].rstrip(")")))
            return "testing"
        return stage

    def render_tracker() -> None:
        icons = {"pending": "○", "running": "◐", "done": "✓", "failed": "↺"}
        parts = []
        for key, label in STAGE_NODES:
            status = progress.get(key, "pending")
            badge = (
                f"<span class='step-badge'> ×{iteration_count['n']}</span>"
                if key in ("developer", "qa", "testing") and iteration_count["n"] > 1
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
    # A fresh id per successful generation (including Regenerate) so the
    # live preview's localStorage never carries over data from a previous,
    # unrelated generation shown earlier in this same browser tab — see
    # pipeline/preview.py.
    st.session_state["preview_run_id"] = uuid.uuid4().hex


if generate:
    if audio_payload is None:
        st.error("Provide a description first: upload a file or record one.")
        st.stop()

    with st.status("Transcribing...", expanded=False) as status:
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
                    f"{MAX_UPLOAD_BYTES / 1_000_000:.1f}MB limit this app accepts. Try a "
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
            st.error(_friendly_transcription_error(exc))
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

    # The preview renders full-width, not squeezed into a half-width column:
    # a generated app can include its own narrow-viewport CSS (e.g. a
    # @media(max-width:768px) sidebar breakpoint), and a half-width column at
    # normal browser sizes was frequently narrower than that — triggering a
    # mobile layout the app's own author never actually verified, making a
    # perfectly fine app look broken in "the preview" specifically. Confirmed
    # empirically: the same iframe at full page width reliably renders above
    # that threshold at any normal window size.
    language = result.architecture.language
    extension = result.architecture.file_extension
    if language == "html":
        st.caption("Live preview")
        preview_run_id = st.session_state.get("preview_run_id", "0")
        st.iframe(isolate_local_storage(result.code, preview_run_id), height=480)
    else:
        st.caption(f"Generated {language} source")
        st.code(result.code, language=language, height=480)

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

    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("Requirements", expanded=False):
            st.json(result.requirements.to_dict())
        with st.expander("Architecture", expanded=False):
            st.json(result.architecture.to_dict())
    with col_right:
        with st.container(border=True):
            st.caption(f"Code review history — {result.iterations} iteration(s)")
            for i, qa in enumerate(result.qa_reports, start=1):
                st.write(f"Pass {i}: {'✅ passed' if qa.passed else '⚠️ issues found'}")
                for issue in qa.issues:
                    st.caption(f"– {issue}")
        with st.container(border=True):
            st.caption(f"Testing history — {result.iterations} iteration(s)")
            for i, test in enumerate(result.test_reports, start=1):
                mode = "browser-tested" if test.executed else "not executed"
                st.write(f"Pass {i}: {'✅ passed' if test.passed else '⚠️ issues found'} ({mode})")
                for note in test.notes:
                    st.caption(f"– {note}")

    with st.expander("Prompt log (every prompt sent to the model)"):
        for entry in result.prompt_log:
            st.write(f"**{entry['stage']}**")
            st.code(entry["prompt"])

st.divider()
st.caption(
    "Built as a multi-agent SDLC pipeline. "
    "[Source on GitHub](https://github.com/Mahijith/Description-to-code-generator)"
)
