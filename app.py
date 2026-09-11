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

st.set_page_config(page_title="Description → Code Generator", page_icon="🛠️", layout="wide")

EXAMPLE_TRANSCRIPT = Path("examples/sample_transcript.txt").read_text()

with st.sidebar:
    st.header("Model")
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
        st.caption(
            "One OpenRouter key powers every agent below. Transcription runs "
            "locally and needs no key at all."
        )

st.title("Description → Code Generator")
st.caption(
    "A 5-agent SDLC pipeline (PM → Requirements → Architect → Developer → QA) "
    "turns a spoken description into a working prototype."
)

tab_upload, tab_record, tab_text = st.tabs(["Upload audio/video", "Record", "Paste text"])
transcript_source: str | None = None

with tab_upload:
    uploaded = st.file_uploader("Audio or video file describing the app", type=["mp3", "wav", "m4a", "mp4", "mov", "webm"])
    if uploaded is not None:
        transcript_source = ("file", uploaded)

with tab_record:
    recorded = st.audio_input("Record a description")
    if recorded is not None:
        transcript_source = ("file", recorded)

with tab_text:
    if st.button("Load example"):
        st.session_state["description_text"] = EXAMPLE_TRANSCRIPT
    pasted = st.text_area(
        "Type or paste a description",
        height=150,
        placeholder=EXAMPLE_TRANSCRIPT[:80] + "...",
        key="description_text",
    )
    if pasted:
        transcript_source = ("text", pasted)

generate = st.button("Generate", type="primary", disabled=transcript_source is None)

if generate:
    if transcript_source is None:
        st.error("Provide a description first: upload a file, record one, or paste text.")
        st.stop()
    if not demo_mode and not api_key:
        st.error("Enter an OpenRouter API key in the sidebar, or turn on Demo mode.")
        st.stop()

    with st.status("Transcribing...", expanded=False) as status:
        kind, payload = transcript_source
        try:
            if kind == "text":
                transcript = PassthroughTranscriber.from_text(payload)
            else:
                suffix = Path(getattr(payload, "name", "recording.wav")).suffix or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(payload.getvalue())
                    tmp_path = tmp.name
                transcriber = make_transcriber_for(tmp_path)
                transcript = transcriber.transcribe(tmp_path)
        except TranscriptionError as exc:
            status.update(label="Transcription failed", state="error")
            st.error(str(exc))
            st.stop()
        status.update(label="Transcribed", state="complete")

    with st.expander("Transcript", expanded=False):
        st.text(transcript)

    llm = MockLLMProvider() if demo_mode else OpenRouterProvider(Secrets(api_key), model=model_id)
    orchestrator = Orchestrator(llm)

    stage_boxes: dict[str, st.delta_generator.DeltaGenerator] = {}
    stage_labels = {
        "pm_kickoff": "Project Manager — kickoff brief",
        "requirements": "Requirements Analyst",
        "architect": "Architect",
        "pm_summary": "Project Manager — final summary",
    }

    def on_stage(stage: str, stage_status: str) -> None:
        label = stage_labels.get(stage, stage.replace("_", " ").title())
        if stage not in stage_boxes:
            stage_boxes[stage] = st.status(label, expanded=False)
        icon = {"running": "⏳", "done": "✅", "failed": "⚠️"}[stage_status]
        stage_boxes[stage].update(label=f"{icon} {label}")
        if stage_status != "running":
            stage_boxes[stage].update(state="complete" if stage_status == "done" else "error")

    try:
        result = orchestrator.run(transcript, on_stage=on_stage)
    except LLMError as exc:
        st.error(f"The model backend failed: {exc}")
        st.stop()

    st.subheader("Result")
    st.success(result.summary)

    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("Requirements", expanded=False):
            st.json(result.requirements.to_dict())
        with st.expander("Architecture", expanded=False):
            st.json(result.architecture.to_dict())
        with st.expander(f"QA history ({result.iterations} iteration(s))", expanded=True):
            for i, qa in enumerate(result.qa_reports, start=1):
                st.write(f"**Pass {i}:** {'✅ passed' if qa.passed else '⚠️ issues found'}")
                for issue in qa.issues:
                    st.write(f"- {issue}")
    with col_right:
        st.write("**Live preview**")
        st.components.v1.html(result.html, height=500, scrolling=True)
        st.download_button("Download index.html", data=result.html, file_name="index.html", mime="text/html")
        with st.expander("View source"):
            st.code(result.html, language="html")

    with st.expander("Prompt log (every prompt sent to the model)"):
        for entry in result.prompt_log:
            st.write(f"**{entry['stage']}**")
            st.code(entry["prompt"])
