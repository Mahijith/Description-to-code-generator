# Description → Code Generator

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.38+-ff4b4b) ![License](https://img.shields.io/badge/model%20cost-free%20tier-8b5cf6) ![Tests](https://img.shields.io/badge/tests-passing-2ea043)

Turn a spoken (or typed) description of an app into a working prototype,
using a small team of AI agents that mirror a real software development
lifecycle — a Project Manager, a Requirements Analyst, an Architect, a
Developer, and a QA Reviewer, with a genuine feedback loop between QA and
the Developer.

<p>
  <img src="docs/screenshots/landing-dark.png" width="49%" alt="App landing screen, dark theme">
  <img src="docs/screenshots/landing-light.png" width="49%" alt="App landing screen, light theme">
</p>
<p>
  <img src="docs/screenshots/result-dark.png" width="100%" alt="Pipeline result: step tracker, QA history, live preview">
</p>

Light and dark are Streamlit's own native theme switcher (top-right "⋮"
menu → Settings) — every visitor gets both, no code needed on their end.

## How it works

```
recording/transcript
        │
        ▼
 Transcriber
   ├─ LocalWhisperTranscriber   (faster-whisper, on-device — real audio/
   │                              video file upload, no API key, ffmpeg
   │                              extracts audio from video)
   └─ PassthroughTranscriber    (typed/pasted text, or the bundled example)
        │
        ▼
 Project Manager  ──kickoff brief──▶
 Requirements Analyst ──requirements.json──▶
 Architect ──architecture.json──▶
 Developer ──index.html──▶
 QA Reviewer ──pass/fail + issues──▶
        │
        └── if QA fails (max 2 tries): loop back to the Developer with
            QA's findings, then re-review
        ▼
 Project Manager writes a final summary
        ▼
 a single self-contained, working HTML/CSS/JS prototype
 (localStorage persistence, no build step, no external requests)
```

Every one of the five agents is a thin wrapper around one `LLMProvider`
interface (`pipeline/llm.py`) — they never know which model is actually
answering them. The default implementation is **`OpenRouterProvider`**,
which talks to [OpenRouter](https://openrouter.ai)'s free-tier models
(default: `inclusionai/ling-3.0-flash-vl:free`). **No Claude is used
anywhere in this app** — that was a deliberate choice so the whole thing
runs on free models a visitor can get a key for in under a minute. Because
OpenRouter is one gateway, **a single API key powers every agent**; there
is no per-agent or per-stage key. Swapping in a different backend (Claude,
Gemini, a local Ollama model) later means writing one more class, not
rewriting the app.

Transcription is a separate, independent concern (`pipeline/transcribe.py`):
audio/video files are transcribed **entirely on-device** with
`faster-whisper`, so it needs no API key and never sends your recording
anywhere. Typed/pasted text always works as a fallback (and is the fastest
way to try the app).

## Running it locally

```bash
pip install -r requirements.txt
# ffmpeg is needed only for transcribing video files:
#   apt install ffmpeg   (or) brew install ffmpeg
streamlit run app.py
```

Open the sidebar and either:
- turn on **Demo mode** — runs the full 5-agent pipeline (including the
  QA→Developer loop) against deterministic canned responses, no API key,
  no network call, works immediately; or
- turn it off, paste in a **free OpenRouter API key** (see below), and hit
  **🔌 Test** to confirm the key/model combo actually works before running
  the full pipeline on it.

Pick one of three bundled example descriptions (task tracker, recipe box,
contact list) from the dropdown and click **Load example** to try it
instantly. After a run, **Regenerate** re-runs the pipeline on the same
description (useful to see a different draft against a live model), and
**Start over** clears everything back to the input screen.

### Command line (no Streamlit)

```bash
python cli.py run examples/sample_transcript.txt --out examples/output --mock
# or, with a real key:
OPENROUTER_API_KEY=sk-... python cli.py run my_recording.mp3 --out output
```

### Getting a free OpenRouter API key

1. Sign up at [openrouter.ai](https://openrouter.ai) (no card required).
2. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys).
3. Check [openrouter.ai/models](https://openrouter.ai/models) (filter by
   "Free") for what's currently free — free-tier availability and rate
   limits change over time, so the model id in the sidebar is just a text
   field you can point at whatever's free when you use it.

## Deploying it for free (Streamlit Community Cloud)

This repo is deploy-ready for [share.streamlit.io](https://share.streamlit.io),
which runs a Streamlit app straight from a public GitHub repo at no cost:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick this repository and branch, and set the main
   file path to `app.py`.
4. Click **Deploy**. `packages.txt` (`ffmpeg`) and `requirements.txt` are
   picked up automatically.

Each visitor pastes in their **own** free OpenRouter key in the sidebar —
it's kept only in their browser session, never logged or written to disk —
so whoever deploys this doesn't get stuck paying for everyone else's usage.
Visitors who don't want to get a key at all can just use **Demo mode**.

### Making the deployed app look polished (a few manual, one-time steps)

Everything below is either already done in this repo or a dashboard click
only the repo/deploy owner can make — nothing here needs code:

1. **Custom app URL** — in Streamlit Cloud, *App settings → General*, set a
   clean slug (e.g. `description-to-code.streamlit.app`) instead of the
   random default.
2. **GitHub social preview** — repo *Settings → General → Social preview*,
   upload [`assets/social_preview.png`](assets/social_preview.png) (already
   generated, 1280×640) so shared links look good on Slack/Twitter/etc.
3. **Repo description & topics** — same *Settings* page: add a one-line
   description and topics like `ai`, `streamlit`, `multi-agent`, `llm` for
   discoverability.
4. **Rebrand later, if you want**: colors live entirely in
   [`.streamlit/config.toml`](.streamlit/config.toml) (edit the hex values —
   both `[theme.light]` and `[theme.dark]` need to stay defined together, or
   Streamlit removes the light/dark switcher entirely) and the mark itself
   is [`assets/logo.png`](assets/logo.png) (swap the file, same filename).
5. **Dark/light mode** — nothing to configure; it's Streamlit's native "⋮"
   menu → Settings, already on by construction (see previous point).

## Security notes

- The OpenRouter key is wrapped in a small `Secrets` class
  (`pipeline/secrets.py`) whose `__repr__`/`__str__` never expose the raw
  value. The raw key is read in exactly one place in the entire codebase —
  inside `OpenRouterProvider`, to build the request header — and is never
  written into the prompt log, the generated prototype, or any error
  message.
- Every agent only ever holds a reference to the `LLMProvider` interface,
  never to `Secrets` itself, so a bug in a prompt can't leak a key.
- The generated prototype renders list items with `textContent`, never by
  concatenating user input into `innerHTML`, to avoid XSS-style bugs in the
  app it produces.

## Limitations

- Each generated prototype manages one primary entity (the thing the
  recording is mostly about — tasks, contacts, recipes, etc.) with
  create/edit/delete/complete/filter-style actions, as a single HTML file
  with `localStorage` persistence. This is a rapid-prototyping tool, not a
  full application compiler — multi-entity apps, real backends, and
  authentication are out of scope by design.
- Free-tier model quality and rate limits vary and change over time; if a
  run fails or produces poor output, try Demo mode to confirm the pipeline
  itself is working, or try a different free model id.
- `faster-whisper` downloads its model on first use and needs `ffmpeg` on
  PATH for video files; if either is unavailable, paste the transcript as
  text instead — it always works.

## Project layout

```
pipeline/
  secrets.py       Secrets — encapsulates the API key
  llm.py           LLMProvider (ABC), OpenRouterProvider, MockLLMProvider
  transcribe.py    Transcriber (ABC), LocalWhisperTranscriber, PassthroughTranscriber
  schema.py        ProjectBrief, Requirements, ArchitectureDoc, QAReport, PipelineResult
  agents.py        Agent (ABC) + the 5 SDLC personas
  orchestrator.py  Orchestrator — runs the pipeline incl. the QA/Dev loop
app.py             Streamlit UI (hero, step tracker, results, buttons)
cli.py             headless runner
.streamlit/config.toml   theme (light + dark; edit here to rebrand)
assets/            logo.png (page icon + in-app mark), social_preview.png
examples/          sample transcripts (3 example descriptions) + committed mock-mode output
tests/             pytest suite (runs entirely against MockLLMProvider)
docs/PROCESS.md    the brainstorming / prompt-iteration write-up
docs/screenshots/  README screenshots
```

See [`docs/PROCESS.md`](docs/PROCESS.md) for the design process, the AI
tools used, and the prompt iterations behind each agent.
