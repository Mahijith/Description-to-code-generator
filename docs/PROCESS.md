# Brainstorming & Process

## The problem, as I framed it

The brief was: given an audio/video recording describing a process or
application, design an AI-driven workflow that prototypes it. The
interesting part isn't "call an LLM once with the transcript and get code
back" — a single prompt can produce something, but it collapses several
genuinely different jobs (understanding intent, deciding scope, choosing an
implementation, writing code, checking the code against what was asked)
into one pass with no way to catch its own mistakes. Real software teams
split those jobs up for a reason. So I framed this as: **can a small team
of AI agents, each doing one job well, mirror how a real team turns a
verbal description into a working draft — including catching its own
mistakes before shipping?**

That reframing is what led to the two decisions that shaped everything
else: a multi-agent pipeline instead of one prompt, and a real feedback
loop instead of a straight line.

## How the design evolved (this was a real back-and-forth)

I want to be honest that this design went through several iterations
during planning, each driven by a concrete constraint I hit rather than
taste:

1. **First idea: a 3-stage pipeline** (transcribe → extract a spec → generate
   code), hosted as a local Python app. Straightforward, but it collapsed
   "understand requirements," "decide the technical approach," and "check
   the work" into one generation step — it couldn't catch its own mistakes.
2. **Reframed as a 5-agent SDLC pipeline** with a Project Manager
   orchestrating a real feedback loop: PM → Requirements Analyst → Architect
   → Developer → QA Reviewer → back to PM, which sends QA's findings back to
   the Developer (bounded at 2 iterations) or signs off. This is the
   structural idea that survived every later pivot.
3. **Hosting exploration.** I first considered publishing this as a Claude
   Artifact, using a runtime capability that lets a hosted page ask Claude
   on the *viewer's own account* — free hosting, no API key, ever. I
   actually read the platform's type definitions before committing to this,
   which turned up a real constraint: that capability accepts text and
   images only, no audio. Since the brief is specifically about *audio/video*
   recordings, and a hosted static page can't call an external transcription
   API either (CSP blocks it), this path would have meant mic-only input
   with no file upload — a real capability, not a rewrite, but not what "an
   audio or video recording" implies to me. I said so plainly instead of
   quietly building around it.
4. **Model backend.** I was asked directly whether the pipeline should use
   Claude at all. It doesn't. Free-tier flexibility mattered more here, so
   every agent talks to an `LLMProvider` interface backed by
   **OpenRouter** (originally `inclusionai/ling-3.0-flash-vl:free`), not
   Claude. I verified that model exists at the URL I was given, but I
   could not inspect its actual input modalities or current rate limits —
   `openrouter.ai` is blocked by this sandbox's own network policy for both
   `curl` and a web-fetch tool — so I said that plainly rather than
   asserting specifics I couldn't check, and made the model id a one-line
   config value instead of a hardcoded assumption. **Update:** the original
   model was Vision-Language, not chosen for coding ability, and the
   sidebar originally let each visitor paste in their own OpenRouter key
   and model id. The deployer later chose to switch to one shared,
   deployer-supplied key so visitors need nothing, and picked
   `nvidia/nemotron-3-ultra-550b-a55b:free` specifically for the Developer
   agent's actual job (writing HTML/CSS/JS) — again a model id I couldn't
   independently verify from this sandbox, so I took it as given rather
   than guess a substitute. **Update:** Nemotron's free-tier access on
   OpenRouter turned out to route through Nvidia's own backend, which hit
   a transient "service temporarily overloaded" 502 in practice (exposed a
   real bug in error handling along the way — see the round-five note
   below). The deployer then switched `DEFAULT_MODEL` twice more, to
   `thinkingmachines/inkling:free` and then `google/gemma-4-31b-it:free` —
   neither of which I could verify from this sandbox (`openrouter.ai`
   stayed blocked throughout this project), so both were taken as given
   rather than checked.
5. **Transcription backend.** "VL" in a model name means Vision-Language
   (text + images), not audio — so the free chat model above can't
   transcribe a recording. I first moved transcription to `faster-whisper`
   running locally: free, offline, needs no key, and — as a side effect of
   being a real backend instead of a static page — it can handle genuine
   audio/video file uploads, which the Artifact path couldn't. **Update:**
   this broke on Streamlit Community Cloud (its base image is missing the
   OpenMP system library `ctranslate2` needs, and the free tier's
   resources/network don't reliably support the first-use Hugging Face
   model download either), and by the time I could verify that against
   real deployment logs, the deployed branch had also drifted from the one
   I'd patched — so I replaced local Whisper with one call to Groq's
   hosted Whisper API instead. It costs the "entirely on-device" property
   and needs one `GROQ_API_KEY` (the deployer's, not each visitor's), but
   sidesteps the system-dependency and resource-limit problems entirely,
   which matters more for a Cloud deployment meant to just work.
6. **Two keys, not one per visitor.** Every agent goes through the same
   OpenRouter gateway (one `OPENROUTER_API_KEY`) and transcription goes
   through Groq (one `GROQ_API_KEY`) — two capabilities, two keys, but both
   set once by the deployer rather than collected from visitors. This is a
   real trade-off, not a free lunch: it costs the deployer's own free-tier
   rate limits under real traffic, in exchange for a zero-setup visitor
   experience.
7. **OOP structure.** Once there were 5 agents doing structurally identical
   work (build a prompt, call a model, parse the reply), the repetition was
   worth naming: one `Agent` base class holds the shared "ask for text" /
   "ask for JSON" plumbing and every prompt-log entry, and one `LLMProvider`
   interface means the agents never know or care which model answers them.
   The one piece of genuinely sensitive data — the API key — is wrapped in
   its own `Secrets` class specifically so a bug in prompt-building code
   structurally cannot leak it (agents hold an `LLMProvider`, never a
   `Secrets`).

## Prompt iterations worth calling out

**Getting JSON back reliably, without a JSON-mode guarantee.** Since
free-tier models on an arbitrary gateway don't all support a strict JSON
mode, an early version of each structured prompt just said "reply in JSON."
That's not enough — models routinely wrap replies in explanations or
markdown fences. The prompts now say explicitly *"Reply with ONLY a JSON
object of this exact shape: {...}"* with the shape spelled out inline, and
`LLMProvider.complete_json` parses tolerantly (whole reply → first fenced
code block → the substring from the first `{`/`[` to the last `}`/`]`)
before giving up and asking the model once more with a sharper instruction.
This is the same discipline OpenRouter's raw chat-completions endpoint and
Claude's own `sample.json()` capability both push you toward, for the same
reason: there is no server-side guarantee, only a well-shaped ask plus
tolerant parsing on your end.

**Making the QA→Developer loop real, not decorative.** It would have been
easy to have QA always pass, or to fake the "second pass" cosmetically. I
didn't want that — I wanted the loop to actually change something. The QA
prompt explicitly checks for an empty-state message ("is there a message
shown when the list is empty?"), and the Developer prompt, when it
receives QA feedback, is told to fix exactly what QA flagged. `MockLLMProvider`
mirrors this honestly (it scripts QA to fail once, citing the missing
empty-state message, then pass once the Developer's second draft includes
one) so that the test suite (and `cli.py run --mock`) exercises a *real*
two-pass loop rather than asserting it happened.

**Keeping agents from stepping on each other's job.** The Architect's
prompt originally also asked for field-level detail, which duplicated the
Requirements Analyst's job and produced inconsistent field lists between
the two stages. I narrowed the Architect to technical decisions only (file
structure, storage approach, screen breakdown) and left the data model's
actual shape to Requirements — each agent now has one clearly-owned output.

**Safety in the generated code, stated as an instruction, not assumed.**
The Developer prompt explicitly says to render list items with
`textContent`, "never by concatenating user input into `innerHTML`" — and
QA's checklist explicitly re-checks that. Without stating it, a model
asked to "make a nice list" will often reach for the shortest
`innerHTML.= ...` line it can, which is an XSS-style trap the moment a
field value contains `<`.

## Trade-offs

- **Free models over paid ones**: means variable quality and rate limits.
  I accepted this because the assignment explicitly asks how I use
  accessible AI tools, and the user I built this with prioritized
  zero-cost, swappable models over guaranteed quality. The `LLMProvider`
  seam means a paid backend is a one-class addition later, not a rewrite.
- **A hosted transcription API (Groq) over local Whisper**: costs one API
  key (the deployer's) and sends the recording off-machine, in exchange for
  working reliably on a constrained, shared hosting environment. Local
  Whisper's "genuinely free, keyless, on-device" properties were real
  advantages for a laptop or a fully-controlled server; on Streamlit
  Community Cloud's free tier they weren't worth the missing system
  libraries and unreliable first-use download that came with them.
- **One primary entity per prototype**: a real app usually has several
  related entities. I scoped the Developer/QA agents to one primary entity
  deliberately — this is a rapid-prototyping tool for validating an idea
  fast, not a schema compiler; multi-entity support is a natural next step,
  not a missing feature.
- **A capped 2-iteration loop**: unbounded QA↔Developer looping risks
  cost/latency blowups on a flaky free model with no guarantee of
  convergence. Two passes catches the common case (one real gap, one fix)
  without that risk.

## Round two: UI polish and free-tier hosting

After the pipeline worked end to end, the ask shifted to making the app
"look professional" within the free-tier budget, plus real dark/light mode
and a few more functional buttons. Two things worth recording:

**Dark/light mode was a config bug, not a missing feature.** My first pass
added a custom `<style>` block and a single `[theme]` block in
`.streamlit/config.toml`. Before shipping that, I checked Streamlit's own
bundled theming reference (`streamlit/.agents/skills/developing-with-streamlit/
references/theme.md`, installed alongside the package) rather than guess —
and it says explicitly: *"Users can switch between modes in the app
settings menu only if both `[theme.light]` and `[theme.dark]` are defined.
A custom theme with just `[theme]` locks the app to a single mode."* I had
done exactly that, which would have silently removed the built-in
light/dark switcher for every visitor — the opposite of what was asked. I
verified this two ways before trusting it: launching the app with
`--theme.base light` while the old single-`[theme]` config still forced
dark colors (confirming the lock), then rewriting into `[theme.light]` +
`[theme.dark]` sections and confirming with Playwright that Streamlit's
native "⋮ → Settings" menu now actually shows System/Light/Dark, and that
clicking each one re-renders correctly.

**Prefer the framework's theming over hand-rolled CSS.** The same reference
doc is explicit that colors/fonts/borders belong in `config.toml`, not
custom CSS, because it survives Streamlit upgrades and doesn't fight the
native switcher. I'd initially reached for a CSS block to reproduce card
styling that `st.container(border=True)` already does natively, and to
color a footer that `st.divider()` + `st.caption()` with a markdown link
already themes correctly on their own. I cut both. What's left as custom
CSS is only the two things Streamlit has no built-in equivalent for at
all — pill-shaped stage badges and the horizontal step tracker — and even
those read `st.context.theme.type` at render time to pick their colors, so
they track whichever theme the visitor currently has active instead of
assuming one.

**A real, if minor, bug the UI polish pass surfaced**: the original
`Regenerate`/`Download` interactions exposed that Streamlit widgets don't
reset when you clear their `session_state` key — the frontend keeps its
own value until the widget's `key` itself changes. "Start over" looked
like it worked (the state was cleared) but the textarea kept showing old
text. Caught this by actually driving the app with Playwright rather than
trusting the code by inspection, and fixed it by versioning the widget's
key (`description_text_{generation}`) and bumping the counter on reset —
the standard pattern for forcing a Streamlit widget to truly reinitialize.
The same pass also caught (and fixed) a latent issue where clicking
**Download** — itself just another widget interaction that reruns the
whole script — would have made the entire Result section vanish, because
it lived in a local variable instead of `st.session_state`.

## Round three: testing the real audio path, and a connection-test button

Up to this point every test — including all the Playwright runs — had gone
through the text-paste input, never the actual audio transcription code.
That's a gap: `LocalWhisperTranscriber` is the part of the app most likely
to break in ways a text-only test can't catch (wrong ffmpeg invocation,
temp files not cleaned up, a model that fails to load). So before adding
anything new, I closed it: installed `ffmpeg` and `espeak-ng` in this
sandbox, synthesized a real 18-second spoken app description with
espeak-ng, converted it to mp3, and ran it through the actual transcriber
code. That surfaced a real, sandbox-specific limitation rather than a code
bug: `faster-whisper` downloads its model from Hugging Face Hub on first
use, and this sandbox's network policy blocks huggingface.co outright.
Everything *around* that call — ffmpeg extracting audio from a synthetic
test video, joining Whisper's segment objects into one transcript, cleaning
up the temp `.wav` file afterward — checks out and is now a permanent test
(mocking only the model itself, since that's the one piece that needs a
network call this environment won't allow). Streamlit Community Cloud has
normal outbound internet access, so the actual model download should work
fine there; that's the one part of this project I can't verify from inside
this sandbox and said so plainly rather than claiming a live run.

## Round four: the Cloud deployment actually failed, and why

The optimistic close of round three ("should work fine there") turned out
to be wrong, or at least unconfirmed — the user reported recording and
transcription still not working on the live Streamlit Cloud app. Two things
came out of actually chasing that down instead of re-guessing:

First, a fake-microphone Playwright test (Chromium's
`--use-file-for-fake-audio-capture`, fed a real espeak-ng-synthesized
recording) against a locally-run copy of the app showed that recording
itself works correctly — the widget captures real audio, `Generate` becomes
enabled, and the pipeline reaches the transcription call. The failure was
always one step later, at Whisper model load, exactly matching the
sandbox's own `huggingface.co` block — the same failure mode, different
cause. This mattered because it ruled out "the recording UI is broken" and
pointed squarely at the local-Whisper dependency chain instead.

Second, the user's actual Streamlit Cloud build log revealed the deployed
app was running a different branch than the one already patched for the
`libgomp1`/error-handling fix — so that fix had never shipped. The log also
only covered the build/startup phase; the app's own error handling catches
transcription failures and shows them in the browser without ever writing
to the server log, so the log couldn't show the live failure either way.

Given a system-library dependency (`libgomp1`), a first-use network
download, and a resource-constrained free tier all had to line up correctly
for local Whisper to work at all — and a branch mismatch had already
undone one fix once — the more robust move was removing that whole
dependency chain rather than continuing to patch it. `GroqWhisperTranscriber`
replaced `LocalWhisperTranscriber`: one hosted HTTP call, no local model, no
ffmpeg, no first-use download. It trades "genuinely on-device" for "one API
key the deployer sets once," which is a better trade for a shared Cloud
deployment than for a personal machine.

**Adding a "Test connection" button** for the OpenRouter key turned up a
second real bug the same way the UI-polish round did: the handler called
`st.error(...)` inside an `except` block and then unconditionally called
`st.rerun()` right after it — which interrupts the script immediately, so
the error text never actually reached the browser before being wiped by
the rerun it triggered. Caught this by clicking the button with Playwright
using a deliberately bad key and noticing the failure *badge* appeared but
the detailed message didn't. Fixed by storing the message in
`session_state` (the same pattern already used for `result`) and rendering
it on every run instead of only inside the failing branch. **Update:** this
button, along with the rest of the per-visitor key/model sidebar UI, was
later removed entirely when the deployer switched to one shared,
deployer-supplied `OPENROUTER_API_KEY` — there's no visitor-facing
connection to test anymore, since visitors never enter a key.

## Round five: a real bug hiding behind what looked like a key problem

Once both keys (Groq, OpenRouter) were confirmed working, the app still
failed — this time with a raw dict dump: `Unexpected OpenRouter response
shape: {'id': ..., 'error': {'message': 'Upstream error from Nvidia:
Service temporarily overloaded', 'code': 502, ...}}`. Nemotron's free-tier
backend was genuinely overloaded, but that's not what made this worth
fixing: `OpenRouterProvider.complete()` only checked `resp.status_code` for
failures, and OpenRouter had returned this as an HTTP 200 with the error
embedded in the JSON body instead of a real completion — a legitimate
provider-proxying pattern the code simply didn't handle, so a real,
retryable upstream outage surfaced as an opaque "unexpected shape" message
indistinguishable from an actual code bug. Fixed by checking for an
`"error"` key in the parsed body regardless of status code, and covered
with a test that reproduces the exact response shape from the report
rather than a synthetic one, so a regression here would be caught by a
concrete example, not just a plausible-looking mock.

## What I'd do next with more time

- Let the Architect propose more than one screen/entity and have the
  Developer emit a small multi-page app instead of a single file.
- Stream the Developer's output into the live preview as it's written
  (OpenRouter's endpoint supports streaming) instead of waiting for the
  full reply.
- Add a second, independent QA pass that actually executes the generated
  JS in a headless browser (the way this project's own Playwright test
  drives the app) rather than relying on a model reading its own code.
- Give the deployer (not visitors — that sidebar UI is gone now) a small
  admin-only way to check the configured `OPENROUTER_API_KEY`/model still
  work, since the old visitor-facing "Test connection" button no longer
  exists.
