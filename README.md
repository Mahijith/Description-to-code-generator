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
   ├─ GroqWhisperTranscriber    (Groq's hosted Whisper API — real audio/
   │                              video file upload, needs one GROQ_API_KEY
   │                              set by whoever deploys the app)
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
(default: `nvidia/nemotron-3-super-120b-a12b:free`, picked for coding
ability — the Developer agent is writing actual HTML/CSS/JS). **No Claude
is used anywhere in this app** — that was a deliberate choice to run on
free models. Because OpenRouter is one gateway, **a single API key powers
every agent**; there is no per-agent or per-stage key. The app owner sets
this key once (`OPENROUTER_API_KEY`, as a Streamlit Cloud secret or
environment variable) — visitors don't need their own, and there's no
sidebar key/model input for them to fill in. Swapping in a different
backend (Claude, Gemini, a local Ollama model) or model id later means
writing one more class or changing one constant, not rewriting the app.

Transcription is a separate, independent concern (`pipeline/transcribe.py`):
audio/video files are transcribed via **Groq's hosted Whisper API**. An
earlier version ran `faster-whisper` entirely on-device, but that broke on
Streamlit Community Cloud (a missing system OpenMP library, plus an
unreliable first-use model download on the free tier — see
`docs/PROCESS.md`), so it was replaced with one hosted API call instead.
The app owner sets one free `GROQ_API_KEY` (console.groq.com/keys) as a
Streamlit Cloud secret or environment variable; visitors don't need their
own. Typed/pasted text always works as a fallback (and is the fastest way
to try the app) and needs no key at all.

## Running it locally

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-...  # free key from openrouter.ai/keys, powers the 5-agent pipeline
export GROQ_API_KEY=gsk_...       # free key from console.groq.com/keys, needed only for audio/video transcription
streamlit run app.py
```

Both keys are read once at startup (from these env vars, or from
`.streamlit/secrets.toml` when running under Streamlit) — there's nothing
to enter in the app itself. Without `OPENROUTER_API_KEY` set, Generate
shows a clear "not configured" error instead of running; without
`GROQ_API_KEY`, the same is true for Record/Upload, but Paste text still
works with no key at all.

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
   "Free") if you want to point `DEFAULT_MODEL` (`pipeline/llm.py`) at a
   different model — free-tier availability changes over time.

## Deploying it for free (Streamlit Community Cloud)

This repo is deploy-ready for [share.streamlit.io](https://share.streamlit.io),
which runs a Streamlit app straight from a public GitHub repo at no cost:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick this repository and branch, and set the main
   file path to `app.py`.
4. Click **Deploy**. `requirements.txt` is picked up automatically.
5. Under the app's **Settings → Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "sk-..."
   GROQ_API_KEY = "gsk_..."
   ```
   (free at [openrouter.ai/keys](https://openrouter.ai/keys) and
   [console.groq.com/keys](https://console.groq.com/keys)). Both are the
   *deployer's* keys, set once, shared by every visitor — nobody pastes in
   their own key or picks a model; Generate and Record/Upload each show a
   clear "not configured" error if their key is missing, and Paste text
   always works regardless.

Since both keys are shared across every visitor rather than one each,
watch OpenRouter's/Groq's free-tier rate limits under real traffic — a
busy app can hit them faster than a per-visitor-key design would.

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
  message. Both `OPENROUTER_API_KEY` and `GROQ_API_KEY` live only in
  Streamlit secrets/environment variables, set by the deployer — never in
  a widget a visitor's browser can read back.
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
- Free-tier model quality and rate limits vary and change over time, and
  since one `OPENROUTER_API_KEY` is now shared across every visitor, a busy
  deployment can hit rate limits faster than a per-visitor-key design
  would; if a run fails, try again shortly or point `DEFAULT_MODEL` at a
  different model.
- Audio/video transcription needs a `GROQ_API_KEY` configured by the
  deployer and outbound internet access to Groq; if either is unavailable,
  paste the transcript as text instead — it always works and needs no key.
- Every OpenRouter request asks for `MAX_OUTPUT_TOKENS` (`pipeline/llm.py`,
  currently 8000) tokens back, and a reply that still gets cut off (a
  model's own free-tier cap can be smaller) raises a clear error rather
  than silently handing a truncated HTML file to QA — but a model with a
  genuinely small output window will keep hitting that ceiling on the
  Developer stage specifically, since it writes the largest output of any
  agent. If that happens repeatedly, try a model with a larger context
  window rather than assuming it's a bug.

## Project layout

```
pipeline/
  secrets.py       Secrets — encapsulates the API key
  llm.py           LLMProvider (ABC), OpenRouterProvider, MockLLMProvider
  transcribe.py    Transcriber (ABC), GroqWhisperTranscriber, PassthroughTranscriber
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
