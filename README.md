# ATA System

**ATA — Audio to Application System.**

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.38+-ff4b4b) ![License](https://img.shields.io/badge/model%20cost-free%20tier-8b5cf6) ![Tests](https://img.shields.io/badge/tests-passing-2ea043)

Turn a spoken audio/video description into a working prototype of
*whatever you actually described* — a small team of AI agents runs the
whole SDLC (a Project Manager, a Requirements Analyst, an Architect, a
Developer, and a Code Reviewer) without assuming every app manages a list
of records. A task tracker gets one thing built; a calculator, a game, or
a debugging tool gets something else entirely — see "No forced shape"
below. A real automated Testing stage then actually runs the result in a
headless browser, at whichever tier applies: registration/login when the
app has accounts, add/edit/delete/filter when it manages records, or a
load/render check for anything else. See "Accounts and testing" below.

<p>
  <img src="docs/screenshots/landing-dark.png" width="49%" alt="App landing screen, dark theme">
  <img src="docs/screenshots/landing-light.png" width="49%" alt="App landing screen, light theme">
</p>
<p>
  <img src="docs/screenshots/result-dark.png" width="100%" alt="Pipeline result: step tracker, code review history, live preview">
</p>

Light and dark are Streamlit's own native theme switcher (top-right "⋮"
menu → Settings) — every visitor gets both, no code needed on their end.

## How it works

```
recording (upload or microphone)
        │
        ▼
 GroqWhisperTranscriber   (Groq's hosted Whisper API — needs one
        │                  GROQ_API_KEY set by whoever deploys the app;
        │                  files over ~19.5MB are rejected before upload)
        ▼
 Project Manager  ──kickoff brief──▶
 Requirements Analyst ──requirements.json (entity+actions, freeform features, or a mix)──▶
        │
        ▼ (stops here with a plain notice if nothing to build was found)
 Architect ──architecture.json (has_auth)──▶
 Developer ──source code, whatever shape actually fits──▶
 Code Reviewer ──pass/fail + issues──▶
 Testing ──pass/fail + notes (a real headless browser drives the app itself)──▶
        ▼
 (loop back to Developer if Code Review or Testing found something — max 3 passes)
        ▼
 Project Manager writes a final summary
        ▼
 a single self-contained HTML file
 (no build step, no external requests, no third-party dependencies)
```

`Orchestrator` runs up to `max_qa_iterations` (default **3**) build →
review → test passes: the Developer gets up to two more chances to fix
whatever Code Review or Testing found, then whatever's produced ships
either way — not an unbounded retry loop. (An earlier version of this app
defaulted to a single pass with no retry at all, after several rounds of
rate-limit/timeout/truncation pain on a longer description or a slower
free model — see `docs/PROCESS.md`. The default has grown since — first
to 2, now to 3 — as Testing got real teeth: real browser execution costs
no LLM requests at all, so the retry budget is sized purely around the LLM
calls, not the testing itself.) If you don't like the result,
**Regenerate** gives you a fresh attempt any time.

Every app is one self-contained HTML file — inline CSS/JS, `localStorage`
persistence, opens directly in a browser, no server. The Architect no
longer picks a language per app (an earlier design let it choose Python
for automation-style tools); committing to one output shape is what makes
real, non-optional browser testing possible for every app, not just the
ones that happen to land on HTML.

### No forced shape

The Requirements Analyst doesn't assume every app manages a list of
records. If it does (a task tracker, a contacts list, an inventory), it's
described as a primary entity + fields + actions, same as always. If it
doesn't (a calculator, a game, a converter, a debugging aid, a chat
interface), the entity is left empty and the app's real capabilities go
into a freeform `features` list instead — and the Developer is told
explicitly not to bolt on a generic add/edit/delete/filter form onto
something that isn't about managing records. Many apps are a genuine mix
of both; neither is required. This isn't a cosmetic change — a real bug
report ("asked for a code debugging application, got a rent-management
app") turned out to be exactly this: the old prompt forced a "primary
entity" onto every description, so the model invented a generic one when
the actual concept didn't have one. See `docs/PROCESS.md` for the full
diagnosis, including a second bug this fix uncovered: Testing's browser
driver used to call the CRUD check unconditionally too, so an app with
correctly *no* entity was failing Testing outright for lacking a form it
was correctly never told to build.

### No scope at all is a different case, and stops the pipeline early

"No forced shape" is about *not* assuming a shape a description doesn't
call for — a calculator correctly has no entity but still has real
`features`. It's a separate question whether a description has anything
to build at all. Rather than growing a list of specific rejected phrases
every time a new gibberish form turns up, the Requirements Analyst is
given one general test: *after reading the transcript, could you say
what app/tool/process should be built and roughly what it's for?* If
not, `entities`, `actions`, and `features` all stay empty — a hard gate
— regardless of how long, confident, or coherent the transcript sounds,
and regardless of whether it mentions technology at all. A short,
vague-but-real request like "make me something for my tasks" still
clears the bar fine (it names a domain and a goal); the bar is "is there
an actual target," not "is it fully detailed."

Below that bar, none of the following count as scope — this list is
explicitly non-exhaustive; the same test applies to any other content
that isn't a software specification, including forms not listed here:

- Greetings, sign-offs, filler, or mic-check phrases ("hello," "can you
  hear me," "testing one two three").
- Silence, background noise, or fragmented/incoherent speech.
- A personal opinion or preference stated on its own ("I like France")
  — a preference is not a feature request.
- A question or remark directed at a listener as if in conversation
  ("what country do you like?") — talking *to* someone, not specifying
  software, even when it reads like natural dialogue.
- A story, anecdote, review, or description of something in the real
  world that isn't a request to build software — even a detailed one
  with concrete nouns that superficially look like fields (e.g. "I know
  a restaurant down my lane... it's a Chinese restaurant named Panda
  Express" describes a restaurant, not a request for a restaurant app).
- A bare instruction to build *something*, with no goal, domain, or
  feature actually named ("build me an app," "make something cool") —
  an instruction alone isn't a specification, even though it's literally
  about building an app.
- Meta-commentary about the recording itself, or off-topic small talk.

`Orchestrator.run` checks `Requirements.has_buildable_scope` right after
that stage and returns immediately if it's false, before Architect/
Developer/Code Review/Testing ever run. The app shows a plain notice
plus the transcript instead of a generated app. This costs no extra LLM
call: it reuses the Requirements stage that already runs on every
request.

### Accounts and testing

The Architect decides whether an app's concept genuinely implies user
accounts (`has_auth`) — most rapid prototypes don't (a calculator, a
single shared list), some do (a personal tracker, a multi-user tool). When
it does, accounts live in `localStorage` under their own key (separate
from the app's main data) — a small embedded "database" that survives a
reload, not an in-memory value that resets. The Developer is given a
fixed contract of element ids (`pipeline/auth_contract.py`) for the
registration/login form and a status element, so the flow is both usable
and — critically — testable by something other than an LLM's opinion of
its own code.

The Testing stage is real, not an LLM's guess, and it's not limited to
accounts — every app's core add/edit/delete/filter flow gets driven for
real:

- `pipeline/crud_contract.py` derives a fixed-but-per-app set of element
  ids from the app's *actual* fields (e.g. a "due date" field becomes
  `#field-due-date`) and declared filters, and hands the same ids to both
  the Developer's prompt and the browser driver — they can't drift apart.
  Boolean fields (e.g. a "completed" flag) are deliberately left out of
  this contract — a real app almost never lets you create something
  pre-completed — and aren't real-execution-tested; Code Review's text
  judgment still covers them, along with any filter that doesn't target a
  `select`-typed field (there's no unambiguous way to browser-test a
  free-text filter's UI convention), and any entity beyond the first one
  the Requirements Analyst extracted (only `entities[0]` gets a fixed-id
  contract). The contract also requires `#add-form`/`#item-list` to be
  visible without any navigation first — a real generated multi-screen
  app once hid its primary entity's form behind a sidebar click, which
  surfaced as an opaque test timeout rather than a real failure until
  `browser_tester.py` started checking visibility explicitly.
- `pipeline/browser_tester.py` loads the generated file in a real headless
  Chromium and, when the app has accounts, logs in *first* (a plausible
  app design gates the entity UI behind login) — including a duplicate
  username registration attempt woven into the existing wrong-password
  check, so a broken "reject duplicates" rule gets caught for free — then,
  if the app manages an entity, adds an item, edits it in place, exercises
  the first filterable field, and deletes it, checking real DOM state at
  every step.
- When an app has **neither** accounts nor an entity (a calculator, a
  debugging tool, anything genuinely open-ended), there's no fixed-id
  contract to derive without knowing the app's shape in advance — so
  Testing falls back to a generic smoke test: load the page for real,
  confirm it rendered visible content, and fail on any uncaught JS error
  (`page.on("pageerror", ...)`, checked for every app regardless of tier).
  Honest scope: this proves the page didn't silently fail to load, not
  that a bespoke feature like "step through code line by line" is
  correct — Code Review's text judgment is what actually grades that.
- You get one `TestReport` per iteration in the UI ("Testing history"),
  flagged `executed=True`/`False` so you can tell a real result from a
  skipped one.
- Playwright is a **dev/test-only** dependency (`requirements-dev.txt`),
  not part of the deployed app's `requirements.txt`. A bare Streamlit
  Community Cloud deployment has no browser binary available, so its
  Testing stage will show "skipped" rather than a real pass/fail,
  until/unless a browser is set up there separately (not done by this
  project — see `docs/PROCESS.md`).

### The live preview is isolated per generation

Streamlit's HTML-embedding components (`st.iframe`, and its predecessor
`st.components.v1.html`) render the app in a `srcdoc` iframe with
same-origin access to the Streamlit app itself — confirmed straight from
their own docstrings. Same origin means one shared `localStorage`: without
`pipeline/preview.py`'s `isolate_local_storage`, generating a task tracker
and then regenerating a completely different app that also happens to use
`localStorage.setItem("tasks", ...)` (a very common key) would let the
second preview see, or silently corrupt, data left over from the first —
confirmed empirically with a throwaway Streamlit + Playwright probe, not
just reasoned about. `app.py` generates a fresh id per successful
generation and wraps *only the preview copy* of the code with a small shim
that namespaces every `localStorage` key under that id (via
`Object.defineProperty` — a plain `window.localStorage = ...` assignment
silently no-ops in Chromium, also confirmed directly). The download button
still serves the real, unmodified file — a deployed app should have real,
permanent storage, not a preview-only isolation hack.

`app.py` uses `st.iframe` rather than the older `st.components.v1.html`:
the latter's own deprecation notice says it "will be removed after
2026-06-01," which has already passed, and `requirements.txt` has no
Streamlit upper bound — a future dependency resolution could silently
break every preview with no code change here to explain why.

### The preview renders full-width

A generated app can include its own narrow-viewport CSS (e.g. a
`@media(max-width:768px)` rule that shrinks a sidebar). Squeezed into a
half-width column, the preview's real rendered width was frequently
narrower than that breakpoint at ordinary browser sizes — triggering a
mobile layout the app's own author never actually verified, which can
overflow/overlap (confirmed with a real bug report and reproduced
directly: a generated app's sidebar nav text overflowed a shrunk 60px
column straight into the main heading). The preview now renders full
page width, which reliably stays above that threshold at any normal
window size; `DeveloperAgent`'s prompt also now asks explicitly for any
responsive breakpoint to actually be checked, not just added.

Every one of the five agents is a thin wrapper around one `LLMProvider`
interface (`pipeline/llm.py`) — they never know which model is actually
answering them. Both `app.py` and `cli.py` default to **`OpenRouterProvider`**
(default model: `inclusionai/ling-3.0-flash-vl:free` — see `docs/PROCESS.md`
for the full back-and-forth on picking a model/provider, including a round
trip through `AIHubMixProvider` and back). `AIHubMixProvider` is still
fully implemented and tested — both talk to an OpenAI-chat-completions-shaped
API, so they share one request/error-handling implementation and differ
only in URL, key source, and default model; switching `app.py` to it is a
one-line change if OpenRouter's shared-key rate limit becomes a problem
again. **No Claude is used anywhere in this app** — that was a
deliberate choice to run on free models. Because each provider is one
gateway, **a single API key powers every agent** for whichever one is
active; there is no per-agent or per-stage key. The app owner sets this key
once (an environment variable or Streamlit Cloud secret) — visitors don't
need their own, and there's no sidebar key/model input for them to fill in.
Swapping in a different backend (Claude, Gemini, a local Ollama model) or
model id later means writing one more class or changing one constant, not
rewriting the app.

Transcription is a separate, independent concern (`pipeline/transcribe.py`):
audio/video files are transcribed via **Groq's hosted Whisper API**. An
earlier version ran `faster-whisper` entirely on-device, but that broke on
Streamlit Community Cloud (a missing system OpenMP library, plus an
unreliable first-use model download on the free tier — see
`docs/PROCESS.md`), so it was replaced with one hosted API call instead.
The app owner sets one free `GROQ_API_KEY` (console.groq.com/keys) as a
Streamlit Cloud secret or environment variable; visitors don't need their
own. Uploads/recordings over ~19.5MB are rejected before ever reaching
Groq's API, since that's the point it stops accepting files in practice.

## Running it locally

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-...  # free key from openrouter.ai/keys, powers the 5-agent pipeline
export GROQ_API_KEY=gsk_...       # free key from console.groq.com/keys, needed only for audio/video transcription
streamlit run app.py
```

Both keys are read once at startup (from these env vars, or from
`.streamlit/secrets.toml` when running under Streamlit) — there's nothing
to enter in the app itself, and no sidebar. Without either key set,
Generate shows a clear "not configured" error instead of running or
failing silently.

Record or upload a description, click **Generate**, and watch the step
tracker. After a run, **Regenerate** re-runs the pipeline on the same
transcript (useful to see a different draft against a live model), and
**Start over** clears everything back to the input screen.

### Command line (no Streamlit)

```bash
python cli.py run examples/sample_transcript.txt --out examples/output --mock
# or, with a real key:
OPENROUTER_API_KEY=sk-... python cli.py run my_recording.mp3 --out output
```

### Getting API keys

- **OpenRouter** (powers both `app.py` and `cli.py`): sign up at
  [openrouter.ai](https://openrouter.ai) (no card required), create a key
  at [openrouter.ai/keys](https://openrouter.ai/keys), and check
  [openrouter.ai/models](https://openrouter.ai/models) (filter by "Free")
  for current free-tier options — `DEFAULT_MODEL` in `pipeline/llm.py` is
  one constant to change.
- **AIHubMix** (an alternate provider, implemented and tested as
  `AIHubMixProvider` but not the app's default right now): sign up at
  [aihubmix.com](https://aihubmix.com) and create a key from their
  dashboard if you want to switch to it — see `docs/PROCESS.md` for why
  this project tried it and reverted.

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
   their own key or picks a model; Generate shows a clear "not configured"
   error if either key is missing.

Since both keys are shared across every visitor rather than one each,
watch OpenRouter's/Groq's rate limits under real traffic — a busy app can
hit them faster than a per-visitor-key design would. `AIHubMixProvider` is
implemented and tested as an alternate provider if OpenRouter's limit
becomes a real problem — see `docs/PROCESS.md` for the round trip this
project already made through it and back.

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

- No provider ever logs or serializes a raw key. `OpenRouterProvider` wraps
  its key in a small `Secrets` class (`pipeline/secrets.py`) whose
  `__repr__`/`__str__` never expose the raw value; `AIHubMixProvider` and
  `GroqWhisperTranscriber` take the key as a plain string but only ever
  print a masked form (`pipeline/secrets.py`'s `mask_key` — first/last 4
  chars and length) to the server log, and only on an auth failure, to help
  diagnose a wrong/stale key without exposing it. No key is ever written
  into the prompt log, the generated prototype, or an error message shown
  in the browser. `OPENROUTER_API_KEY`, `GROQ_API_KEY`, and (if you switch
  to it) `AIHUBMIX_API_KEY` all live only in Streamlit secrets/environment
  variables, set by the deployer — never in a widget a visitor's browser
  can read back.
- Every agent only ever holds a reference to the `LLMProvider` interface,
  never to a raw key or `Secrets` itself, so a bug in a prompt can't leak
  one.
- The Developer/QA prompts require safe handling of untrusted input
  regardless of which language the Architect picks: `textContent` instead
  of string-built `innerHTML` for HTML output, and no string-concatenated
  shell commands or SQL queries for any other language.

## Limitations

- Each generated prototype manages one primary entity (the thing the
  recording is mostly about — tasks, contacts, recipes, etc.) with
  create/edit/delete/complete/filter-style actions, as a single
  self-contained source file. This is a rapid-prototyping tool, not a full
  application compiler — multi-entity apps, real backends, multi-file
  projects, and authentication are out of scope by design.
- The Architect chooses the output language per app; there's no way to
  force a specific one from the UI (the deployer can steer this by editing
  the Architect's prompt in `pipeline/agents.py`). Only the generated
  HTML gets a live in-app preview — other languages show as syntax-
  highlighted source, not an executed result, since there's no safe way to
  run arbitrary generated code inside the app itself.
- Free-tier model quality and rate limits vary and change over time, and
  since one `OPENROUTER_API_KEY` is shared across every visitor, a busy
  deployment can hit rate limits faster than a per-visitor-key design
  would; if a run fails, try again shortly, point `DEFAULT_MODEL` at a
  different model, or switch `app.py` to `AIHubMixProvider`.
- Audio/video input needs a `GROQ_API_KEY` configured by the deployer,
  outbound internet access to Groq, and a file no larger than ~19.5MB
  (Groq's real cutoff in practice, tighter than its documented 25MB) — the
  app rejects anything bigger before ever calling the API.
- Neither `OpenRouterProvider` nor `AIHubMixProvider` sends a `max_tokens`/
  output cap of its own — deliberate, per the deployer. If a model's reply
  still gets cut off (`finish_reason: "length"`, meaning the model or
  provider hit *its own* limit), that raises a clear error naming how many
  tokens the model actually produced, rather than silently handing a
  truncated file to QA. The Developer stage is most exposed to this since
  it writes the
  largest output of any agent; a model that doesn't converge on a single,
  complete file will keep hitting this regardless of any cap, ours or its
  own.

## Project layout

```
pipeline/
  secrets.py       Secrets — encapsulates the API key
  llm.py           LLMProvider (ABC), OpenRouterProvider (default), AIHubMixProvider, MockLLMProvider
  transcribe.py    Transcriber (ABC), GroqWhisperTranscriber, PassthroughTranscriber
  schema.py        ProjectBrief, Requirements (entities/actions or features), ArchitectureDoc, QAReport, TestReport, PipelineResult
  agents.py        Agent (ABC) + the 5 SDLC personas
  auth_contract.py the register/login element-id contract shared by DeveloperAgent's prompt and browser_tester.py
  crud_contract.py the per-app add/edit/delete/filter element-id contract, derived from real Requirements fields
  browser_tester.py real headless-browser test: auth, CRUD, or a generic smoke test — dev/test dep, degrades gracefully
  preview.py       isolate_local_storage — namespaces the live preview's storage per generation (app.py only)
  orchestrator.py  Orchestrator — runs the pipeline incl. the QA+Testing/Dev loop
app.py             Streamlit UI (hero, step tracker, results, buttons) — no sidebar
cli.py             headless runner
.streamlit/config.toml   theme (light + dark) + maxUploadSize
assets/            logo.png (page icon + in-app mark), social_preview.png
examples/          sample transcripts (used by cli.py/tests) + committed mock-mode output
tests/             pytest suite (runs entirely against MockLLMProvider; browser-executed
                   auth tests skip cleanly where no real Chromium binary is available)
requirements-dev.txt  dev/test-only deps (pytest, playwright) — not part of the deployed app
docs/PROCESS.md    the brainstorming / prompt-iteration write-up
docs/screenshots/  README screenshots
```

See [`docs/PROCESS.md`](docs/PROCESS.md) for the design process, the AI
tools used, and the prompt iterations behind each agent.
