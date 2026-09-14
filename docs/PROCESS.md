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
   below). The deployer then switched `DEFAULT_MODEL` three more times —
   `thinkingmachines/inkling:free`, `google/gemma-4-31b-it:free`, then back
   to a different Nemotron variant, `nvidia/nemotron-3-super-120b-a12b:free`
   — none of which I could verify from this sandbox (`openrouter.ai` stayed
   blocked throughout this project), so all were taken as given rather than
   checked. The common symptom across every one of them ("stopping mid
   process") turned out not to be about which model was picked at all — see
   round six. **Update:** after round six's fix (an explicit
   `MAX_OUTPUT_TOKENS`, raised again from 8000 to 16000, plus a clear error
   instead of silent truncation) still hit the same free-tier output cap on
   Nemotron-super, the deployer switched `DEFAULT_MODEL` a fifth time —
   back to the original `inclusionai/ling-3.0-flash-vl:free` from item 4
   above. I flagged plainly that this reverts to a model this project
   originally moved away from for not being coding-focused (see point 5
   below), which is an orthogonal concern to output length — a model can
   be bad at coding and still never truncate, or good at coding and still
   hit a small free-tier cap. Still the deployer's call to make; I
   implemented it as asked.
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
  without that risk. **Update:** even a capped loop turned out to be the
  wrong default for complex tasks in practice — see round ten. The app now
  defaults to one pass; the capped-loop capability itself is still there
  and still tested, just not used by default anymore.

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

## Round six: every model "stopping mid process" was one root cause

After three more model swaps, the deployer reported the same symptom every
time: the pipeline "stops mid process." Swapping models again wouldn't have
told us anything — the pattern itself, recurring identically across
unrelated models, was the actual signal. Re-reading `OpenRouterProvider.
complete()` turned up the real cause: the request never set `max_tokens`
at all, and the response was never checked for truncation. A free-tier
model that falls back to a small provider-side output default (or applies
one on its own) would silently hand back a cut-off reply — not an error,
just incomplete content — and the code would happily hand that straight to
QA and the live preview. The Developer stage is the one most exposed to
this, since it writes by far the largest output (a complete self-contained
HTML/CSS/JS file) of any of the five agents. Fixed two ways: an explicit
`MAX_OUTPUT_TOKENS` on every request so the app itself is never the
limiting factor, and a check on the response's `finish_reason` — `"length"`
means truncation, and now raises a clear, specific error instead of
silently propagating broken HTML. This doesn't guarantee every free model
has a large enough window for this app's biggest prompt, but it turns a
silent, confusing failure into a diagnosable one.

## Round seven: the deployer was right, and the diagnostic proved it

Round six's error message asserted "this model's free-tier output limit is
too small" without ever showing evidence for it — and the deployer pushed
back, correctly, that this looked like the app's own cap rather than
anything the model or provider imposed. Rather than argue the point a
third time, I added the actual data: OpenRouter's response carries a
`usage.completion_tokens` figure showing exactly how many tokens the model
produced before `finish_reason: "length"` fired. The next failure showed
`completion_tokens` at precisely 16000 — the exact value `MAX_OUTPUT_TOKENS`
was set to. That's conclusive: for this model, the app's own cap was the
real bottleneck, not some smaller provider-side ceiling, and my working
theory going into that exchange was wrong. Raised `MAX_OUTPUT_TOKENS` to
32000 accordingly, and left the reasoning about *why* in the code comment
so this doesn't need re-litigating from scratch next time.

One caveat worth stating plainly rather than discovering the hard way: if
a future failure again shows `completion_tokens` landing exactly on
whatever the current cap is, that stops being evidence of "raise the cap
more" and starts being evidence of a model that doesn't converge on a
single, complete file — no ceiling is high enough for a model that
rambles indefinitely. That would call for tightening the Developer
prompt's constraints or picking a different model, not another token-limit
increase.

## Round eight: raising the cap was still the wrong instinct

The deployer's actual ask, three times over, was to remove the app's
output cap entirely — not to have it raised, whatever the number. Round
seven's fix (32000 instead of 16000) technically responded to the new
evidence but still defaulted to "pick a bigger number," which wasn't what
was asked for and risked looking like it was quietly reinterpreting a
direct instruction. Removed `MAX_OUTPUT_TOKENS` and the `max_tokens` field
from the request entirely — this app now asks OpenRouter for nothing but a
completion, full stop.

The `finish_reason: "length"` check stays, deliberately: it isn't a limit,
it's visibility into whatever limit the model or provider applies on its
own, which exists whether or not this app sends anything. Removing that
too would silently resurrect round six's original bug (a truncated reply
accepted as if it were complete) — the one part of "remove limits" that
would make failures *harder* to diagnose rather than fewer to hit, so it
was kept and the reasoning said so plainly rather than assumed obvious.

## Round nine: once it worked, a UI/scope pass

With the pipeline confirmed working end to end, the deployer asked for four
changes at once: drop the Paste-text input (audio/video only now), let the
Developer write in whatever language actually fits the described app
instead of always HTML, remove the sidebar entirely, and cap uploads/
recordings at 19.5MB — the point they'd found Groq's API actually stops
accepting files, tighter than the commonly-quoted 25MB figure.

The language change touched more than the Architect's prompt. Once the
Developer isn't guaranteed to write HTML, a field literally named `.html`
on `PipelineResult` becomes actively misleading — kept it accurate by
renaming to `.code` and adding `language`/`file_extension` to
`ArchitectureDoc`, threading both through the Developer/QA prompts (the
injection-safety instruction generalized from "no innerHTML string-concat"
to "no unsafe string-concat into shell/SQL/markup, whatever the language"),
and the UI (live preview only makes sense for actual HTML; anything else
renders as syntax-highlighted source via `st.code`, and the download
filename/mime type follow the chosen extension). Deliberately kept this to
one file per app rather than reaching for a multi-file/zip-download
architecture — the request asked for language flexibility, not a bigger
project structure, and that's a separate design decision if it's ever
wanted.

"Remove the sidebar completely" was literal, but the "How it works"
explainer inside it was real onboarding value, not sidebar-specific
content — moved it into a main-body expander rather than deleting it
outright, and dropped only the parts that were genuinely sidebar-specific
(the persistent branding caption, the theme-switch hint that duplicates
Streamlit's own always-visible menu).

The 19.5MB figure came from the deployer's own observed usage, not a spec
I could verify (`openrouter.ai`/Groq's docs aside, the *practical* cutoff a
real API enforces isn't always the documented one) — implemented as given,
in two layers: `.streamlit/config.toml`'s `maxUploadSize` (an int, so set
to 20 as a coarse client-side guard) plus an exact byte-level check in
`app.py` that covers both the uploader and `st.audio_input` precisely at
19.5MB, with a clear message naming the actual file size. Verified with a
real Playwright run against a 19.7MB fake file (between the two
thresholds, to prove the app's own check fires specifically, not just
Streamlit's coarser gate) — confirmed the exact error text renders
correctly, since a live model call still isn't reachable from this
sandbox for the language-selection half of this round.

## Round ten: the retry loop stopped being an asset on complex tasks

The QA→Developer feedback loop was this project's founding architectural
idea (round two, above) — the whole reason it's five agents instead of one
prompt. But the deployer reported it "not looking good" on complex tasks
and asked for "a solid one time run... finished in one go" instead. I
asked which loop they meant before touching anything, since the app has
two things that could plausibly be called a loop (the QA/Developer retry,
or plain request timeouts) and they behave completely differently — turned
out to be the retry loop specifically.

The retry loop's premise was that a second Developer pass, armed with
QA's specific findings, produces something strictly better than the first.
That holds when the *reason* for a re-try is a real, fixable gap in the
code. It stops holding once you account for everything this project's own
debugging rounds already surfaced: every extra round-trip through a
free-tier model is another chance to hit a rate limit (round three), an
upstream provider outage (round five), or output truncation (rounds six
through eight) — failure modes that have nothing to do with code quality
and every chance of getting *worse*, not better, on a longer or more
complex app description. A loop that's supposed to fix problems but can
just as easily introduce new ones on the exact inputs it's meant to help
with isn't earning its complexity by default.

Changed `Orchestrator`'s default `max_qa_iterations` from 2 to 1 rather
than deleting the loop: the mechanism is real, still fully exercised by
`tests/test_pipeline.py` (which pins `max_qa_iterations=2` explicitly), and
a deployer who wants the old behavior back can still ask for it by passing
that argument. QA still runs once and reports its findings — that
information wasn't the problem, the automatic retry on top of it was — and
`Regenerate` remains the manual equivalent of a second attempt, just
initiated by a person instead of the pipeline deciding on its own.

## Round eleven: switching providers, not just models

OpenRouter's shared-key rate limit (round ten's own change made this worse
in one sense — a single failed attempt now costs the same, but there's no
retry to spread across, so hitting the limit mid-run is more visible) kept
recurring under real testing regardless of which OpenRouter model was
picked, since the limit is per-key, not per-model. The deployer's fix
wasn't another model swap — it was a different provider entirely:
[AIHubMix](https://aihubmix.com), with its own free-standing API key.

I couldn't verify AIHubMix's API shape myself — `aihubmix.com` is
unreachable from this sandbox, same as every other external API this
project has touched. Rather than guess at endpoint URLs, auth header
format, or response shape (guessing wrong here fails silently or
confusingly, exactly the kind of mistake this project has spent many
rounds trying to avoid), I asked the deployer for a real example. They
provided AIHubMix's own quick-start snippet: an OpenAI-compatible
`POST https://aihubmix.com/v1/chat/completions` with a Bearer token and
the same `{"model": ..., "messages": [...]}` body shape OpenRouter uses.
That one snippet answered the request shape; the model id for Ling on
AIHubMix's catalog was inferred from the model's page URL
(`aihubmix.com/model/ling-3.0-flash-free` → `ling-3.0-flash-free`) rather
than confirmed directly, which I said plainly rather than presenting as
verified — if a real run rejects that model id, that's the first thing to
check.

Since AIHubMix turned out to be shaped exactly like OpenRouter's endpoint,
implementing `AIHubMixProvider` as a second copy of `OpenRouterProvider`'s
~50 lines of request/error-handling would have duplicated every fix this
project already made there (the embedded-error-object check, the
truncation-with-evidence message, the masked-key 401 logging) — and any
future fix to one would silently not apply to the other. Factored the
shared logic into one function
(`_complete_via_openai_compatible_api`, parameterized by URL, key, model,
and provider name for error messages) that both provider classes call;
`OpenRouterProvider`'s own tests kept passing unchanged after the
refactor, which is the cheapest evidence that behavior didn't shift.
`app.py` now constructs `AIHubMixProvider` by default; `OpenRouterProvider`
stays fully implemented and is what `cli.py` still uses, so switching back
is a one-line change, not lost work. **Update:** that one-line reversal is
exactly what happened next — AIHubMix's free `ling-3.0-flash-free` came
back `400 no_available_channel` (confirmed via a second real snippet the
deployer ran directly against AIHubMix, which ruled out a wrong model id
or a request-shape bug on this app's side — it's a capacity/availability
issue on AIHubMix's end for that specific free model), and the deployer
asked to revert to OpenRouter. Reverted `app.py` to `OpenRouterProvider`
with `DEFAULT_MODEL` unchanged (`inclusionai/ling-3.0-flash-vl:free`, the
last one actually configured there) — worth being honest that nothing in
this session's transcript shows clear evidence any specific OpenRouter
model completed a full run successfully; every one hit a distinct failure
(an upstream outage, silent or detected truncation, or the shared-key rate
limit). "Revert to whichever worked best" doesn't have a documented answer
to revert to, so I said that plainly rather than picking one and
presenting it as a confirmed good choice.

## Round twelve: accounts, a real Tester, and a database walked back to size

The deployer asked for three things together: generated apps should get
basic registration/login "memory" when the concept calls for it; the
QA→Developer retry loop should come back, capped at two iterations; and a
new Testing stage should exercise that login flow with synthetic users, in
the same retry cycle, so the output "looks like a working app that can be
deployed at any moment." Two things were worth settling before writing any
code, so I asked rather than guessed: whether Testing should actually
*execute* the generated app or just have a model reason about it (chose
real headless-browser execution for HTML output, LLM-only reasoning
elsewhere — this sandbox already has Playwright/Chromium, and there's no
safe way to run arbitrary generated Python here), and whether every app
should get accounts or only ones that imply them (the Architect decides
per-app, same pattern already used for language choice).

My first pass at the plan had accounts live in an in-memory JS
object — gone the moment the page reloads. The deployer pushed back on
seeing that: the point of the exercise was a genuinely deployable app, not
one that forgets every user on refresh, but they also explicitly didn't
want a real backend/database effort — keep it small. That reframed the
persistence choice rather than the rest of the plan: HTML output now
stores accounts in `localStorage` (the same mechanism already used for
the primary entity's data, so no new API); Python output uses the
standard-library `sqlite3` module, a genuine small embedded database with
zero new dependencies. Both keep the "one self-contained file, no build
step, no third-party deps" constraint intact.

Making the browser test actually mean something took a fixed contract:
without predictable element ids, a headless browser has no reliable way
to find arbitrary LLM-authored form fields. `pipeline/auth_contract.py`
holds those ids (and the exact prompt wording) once, imported by both
`DeveloperAgent`'s prompt and `pipeline/browser_tester.py`'s Playwright
driver — so the instructions given to the model and the selectors used to
grade it can never drift apart. The driver's sequence deliberately proves
persistence, not just in-page behavior: register a synthetic user, reload
the page, then check a wrong password is rejected and the correct one
isn't — reload is the step that would fail if `localStorage` weren't
actually being used. Chromium runs with `--no-sandbox` (needed to launch
as root in a container) and a route handler that aborts anything that
isn't a `file://` request, so a hallucinated "call an API" in generated JS
can't reach the network during the test.

Playwright ended up a dev/test-only dependency (`requirements-dev.txt`),
not part of the deployed app's `requirements.txt` — a bare Streamlit
Community Cloud deployment has no way to fetch a ~100MB browser binary at
build time, so `browser_tester.run_browser_auth_test` detects both "the
package isn't installed" and "the package is installed but no browser
binary exists" and degrades to `TestReport(executed=False, passed=True,
...)` in either case rather than crashing the pipeline. That means the
live deployment's Testing stage will show "skipped" for HTML apps too
until a browser is set up there separately, which this round didn't
attempt (fragile, slow cold starts, not asked for) — worth saying plainly
rather than implying the deployed app gets the same coverage this
sandbox's test suite does. The test suite itself checks whether a real,
launchable browser is actually present (not just whether the package
imports) and skips the browser-executing tests cleanly when it isn't,
so it stays honest in either kind of environment instead of assuming this
sandbox's setup everywhere.

`ProjectManagerAgent.decide` now takes both the QA and Testing reports, and
`Orchestrator.max_qa_iterations` went back to a default of 2 — a conscious
reversal of Round ten's "one solid pass" default, explicitly requested
this time in exchange for a real chance to fix what Testing catches.

As with every provider/model change this session, none of this proves a
*real* OpenRouter model reliably follows the id contract or judges
`has_auth` correctly per app — that needs a live run this sandbox still
can't make. What's verified here is the machinery: two hand-written HTML
fixtures (one correct, one that accepts any password) prove the
browser-driven pass/fail signal is trustworthy, and a full orchestrator
run using a scripted mock model proves the wiring drives a real
browser pass end to end, not just the unit-level function in isolation.

## Round thirteen: a name, and scrubbing the UI of implementation detail

The deployer asked to rename the app to **ATA System** (ATA — Audio to
Application System), with that full-name context living only here on
GitHub — the Streamlit app itself should show just "ATA System," no
abbreviation spelled out — and to remove any mention of API keys or which
platforms the pipeline sources things from from the Streamlit app's UI.

The rename itself was partial: I don't have a tool that can rename the
actual GitHub repository (its slug stays `Description-to-code-generator`
for now — no `list_*`/`create_*`/`update_*` tool in this session's GitHub
toolset touches repository settings), so this round renamed everything
that's actually mine to change — the README title, this file's narrative
going forward, and the Streamlit UI's title/page title. The repo's own
`Settings → General → Repository name` field is a one-click manual step
the deployer still needs to do if they want the URL itself to change;
GitHub redirects the old URL afterward, so nothing (including this
project's own README links, still pointing at the current slug) breaks in
the meantime.

Scrubbing the UI took more than swapping the two obvious "powered by
OpenRouter/Groq" sentences (the hero's "How it works" text and the
footer). Both `_friendly_llm_error` and the transcription path's error
handling had a fallback that returned the raw exception text verbatim
when no specific branch matched — which meant an unanticipated failure
(an upstream provider's own error body, a masked-key log line, whatever
`AIHubMixProvider` or `GroqWhisperTranscriber` happen to say internally)
could still leak a platform name straight to a visitor's screen despite
every *known* case being rewritten. Fixed both fallbacks to return a
generic, non-leaking message instead of the raw string — the actual gap,
not just the messages I happened to already know about. Added a parallel
`_friendly_transcription_error` (same shape as the existing
`_friendly_llm_error`) since `GroqWhisperTranscriber` raises `Groq`-branded
`TranscriptionError` text directly; it passes through the messages `app.py`
already writes itself unchanged (they're already generic) and only
rewrites ones containing "Groq" or "API key." Internal identifiers —
env var names like `GROQ_API_KEY`, class names like `OpenRouterProvider`,
code comments — were left alone: they're plumbing a visitor never sees,
not "context in the app," and renaming them would be a much larger,
unrequested refactor for no visible benefit.

## Round fourteen: renaming QA, and making Testing actually test the app

Three things came in together: rename "QA" to "Code Review" in the app's
own UI; make the Testing stage "explicitly test all the functionalities
in depth," not just the login flow it covered as of Round twelve; and
either switch every generated app to HTML+JS so logins can use
`localStorage`, or explain why not. I asked before touching code, since
both open questions reshape most of the file list: always HTML+JS
(dropping the Python/other-language choice from Round nine) — confirmed;
and "in depth" meaning real browser execution of the primary entity's
full add/edit/delete/filter flow, not a deeper LLM prompt — also
confirmed, over a cheaper LLM-reasoning-only alternative that wouldn't
have been meaningfully stronger than what Code Review already does.

Committing to always-HTML is what actually made the deeper testing ask
tractable: with one guaranteed output shape, Testing needs exactly one
real execution path instead of two (a browser for HTML, a weaker LLM
guess for everything else) — so `TesterAgent`, the LLM-only fallback for
non-HTML output, came out entirely, along with the Python/sqlite3 branch
of the auth prompt it existed to justify. The Architect's prompt dropped
the language-choice paragraph and the `language`/`file_extension` JSON
keys; the dataclass fields stay (so `app.py`/`cli.py`'s generic
mime-type/filename/preview code needs no changes — it just reads a field
that now always happens to be `"html"`).

The harder design problem was generalizing the login test's fixed-id
trick to something as open-ended as "every app's fields and actions."
Login has one fixed shape; a task tracker's fields aren't fixed at all.
The answer, `pipeline/crud_contract.py`, derives ids from the actual
`Requirements` object instead of hardcoding them: `field-<slugified-name>`
per non-boolean field, one `#add-form`/`#add-submit` pair that does double
duty for both create and edit (clicking an item's `data-action="edit"`
just repopulates the same form), and a `#filter-<name>` select — but only
for filters targeting a `select`-typed field with 2+ options, since that's
the only shape a script can set and verify without guessing at the app's
own UI conventions for something like a free-text search box. Boolean
fields (a "completed" flag) are left out of the add-form entirely — real
apps essentially never let you create something pre-completed — and
aren't real-execution-tested; Code Review's text judgment still covers
both of these gaps, same layered-testing philosophy as the auth work.

Ordering mattered once accounts and CRUD had to share one browser session:
a plausible `has_auth` app design gates its entity UI behind login, so the
combined driver (`browser_tester.run_browser_functional_test`, replacing
the auth-only `run_browser_auth_test`) logs in first when the app has
accounts and only then drives the CRUD checks on that same page. Designing
that ordering surfaced a free win: inserting one extra registration
attempt (same username, a second password) between the original
registration and the reload-then-login checks means a broken
duplicate-username rule shows up as the *existing* wrong-password
assertion failing — no new assertion, no new contract id needed to catch
a real, previously-untested bug class.

`max_qa_iterations` went from 2 to 3. Real browser execution costs no LLM
requests at all, so the deployer's now-larger daily request budget is
better spent on Developer retries — Testing got stricter (login *and*
full CRUD), so more generated apps will need that second retry to
actually clear it, and a full run's worst-case LLM call count barely
moved (Testing used to occasionally cost one extra LLM call for non-HTML
apps' fallback; now it costs zero, always).

"QA" → "Code Review" was scoped the same way the ATA rename was: the
stage tracker label, the "How it works" bullet, and the results panel
caption changed; `QAReviewerAgent`, `QAReport`, the internal `"qa"` stage
key, and the prompt text sent to the model stayed — that's plumbing a
visitor never sees. What *did* change is what the prompt asks the model to
check, since Code Review now grades apps against the same concrete CRUD
contract ids the browser driver enforces, not just an abstract
description of what QA should look for.

As with every model-facing change this session, none of this proves a
*real* OpenRouter model reliably follows the now-longer, more specific
Developer contract — that needs a live run this sandbox still can't make.
What's verified here is the machinery: hand-written correct and
single-bug-injected fixtures (a broken delete handler, a login that
accepts any password, a registration that silently overwrites) prove the
pass/fail signal is trustworthy for each piece independently, and a full
orchestrator run with a scripted mock model proves the combined
auth-then-CRUD wiring drives one real browser pass end to end.

## Round fifteen: a CLAUDE.md, and auditing the app against it

The deployer wrote a `CLAUDE.md` stating this project's Goal, Scope, and
Process plainly, and asked for a rework of the app using it as the
reference — not a vague "improve things," a specific instruction to check
the app against three concrete sentences. That's what this round actually
was: an audit against each one, producing three fixes, not a rewrite.

**Process** ("no particular restriction or shape") was already
diagnosed as broken by then: the deployer had asked for "a code debugging
application" and gotten a rent-management app back. Reading
`RequirementsAnalystAgent`'s prompt found why — it said to "identify the
single primary entity being managed... its fields, the actions... any
filters," as if every app manages a records list. A debugging aid doesn't.
Forced to invent one, the model fell back on a generic disconnected
template. The fix: the prompt now branches — an entity for a records-list
app, a new freeform `Requirements.features` list for anything else, a mix
when genuinely true, and an explicit instruction never to invent an entity
to fill the shape. `DeveloperAgent`/`QAReviewerAgent` stopped assuming an
entity exists too. This also exposed and fixed a second, purely mechanical
bug in `browser_tester.py`: Testing's CRUD phase ran unconditionally, so
an app with correctly *no* entity (because the Developer was correctly
never told to build one) was failing Testing outright for lacking a form
it was never supposed to have. Apps with neither an entity nor accounts
now get a generic smoke test instead — load the page for real, fail on
any uncaught JS error, confirm visible content rendered. Honestly scoped:
this proves a page didn't silently fail to load, not that a bespoke
feature like "step through code line by line" is correct.

**Scope** ("every functionality working in the preview... exactly what
they're getting") turned up a bug nobody had reported yet — found by
re-reading that sentence literally and asking "does the preview actually
guarantee that?" It doesn't: `st.components.v1.html` renders the app in a
`srcdoc` iframe with same-origin access to the Streamlit app itself (its
own docstring says so), and same origin means one shared `localStorage`.
Generate a task tracker, then regenerate a totally different app that
happens to also use `localStorage.setItem("tasks", ...)` — a very common
key — and the second preview can show, or corrupt, data from the first.
That's the opposite of "exactly what they're getting." This is the first
finding this session that came from actually running the app rather than
reading its source: a two-line Streamlit probe plus a Playwright script
confirmed the leak directly, and confirmed the fix (a namespaced
`localStorage` shim). It also caught something reading the code would
never have shown: `window.localStorage = shim` **silently does nothing**
in Chromium — no error, just no effect — and `Object.defineProperty`
is required instead. Both findings are checked into
`pipeline/preview.py`'s `isolate_local_storage`, unit-tested with real
Playwright, and wired into `app.py` so only the *preview* copy is
namespaced (per a fresh id each generation) — the download button still
serves the real, unmodified file, since a real deployed app should have
real, permanent storage.

**Goal** ("an application the user can actually use") was at quiet risk
independent of any bug report: `st.components.v1.html`'s own deprecation
notice says removal after 2026-06-01, which has already passed by the
time of this round, and `requirements.txt` pins no Streamlit upper bound.
Migrated to `st.iframe`, the documented replacement — same same-origin
behavior (so the storage fix above is still necessary after migrating),
confirmed directly from its docstring in the installed version rather
than assumed from changelog text.

## Round sixteen: a real generated app, a broken preview, and what it revealed

The deployer pasted a real generated rent-management app whose preview
looked genuinely broken — sidebar nav text overlapping the main heading.
Reproduced it directly rather than guessing: saved the exact HTML,
rendered it through the exact same `st.iframe` + `st.columns(2)` layout
`app.py` uses, and it broke identically. Root cause: the generated app's
own `@media(max-width:768px)` CSS shrinks its sidebar to 60px, and
Streamlit's half-width column rendered the preview iframe at ~344px wide
at every window size tested — well under that breakpoint — so the app's
mobile layout activated, and its nav labels (full words, no
`overflow`/`text-overflow` handling) spilled out of the 60px box into the
main content. Checked whether this was a regression from Round fifteen's
`st.iframe` migration before assuming so: it wasn't — the old
`components.v1.html` reproduced the identical 344px width in the same
layout. The actual fix was removing the half-width squeeze: the preview
now renders full page width, confirmed empirically to stay above 768px
at every realistic window size tried (900–1400px). `DeveloperAgent`'s
prompt also now asks for any responsive breakpoint to be verified, not
just written, since a real deployment can still legitimately be viewed
at a narrow width.

Investigating that app also turned up a second, unrelated bug: two of its
functions ended with a stray extra `);`, an unbalanced-bracket syntax
error that silently kills an *entire* inline `<script>` block — none of
that app's interactivity would have worked at all. This is exactly the
class of bug the Testing stage's real browser execution exists to catch
(driving `#add-form` would have failed outright). Since the fixed
elements matched our CRUD contract precisely, this app should have gone
through Testing — which raises a real, previously-only-theoretical
question directly: is Testing actually running on the live Streamlit
Cloud deployment, or has it been silently reporting "skipped" there this
whole time because Playwright isn't installed on that deployment (a
known, documented gap — see Round twelve)? If the latter, Code Review's
text-only review has been the *only* check running in production, and a
single stray bracket is a very easy thing for a model reading code,
rather than executing it, to miss. Its prompt now explicitly asks it to
mentally balance every bracket in every `<script>` block first — a cheap,
always-useful change regardless of the answer. Whether to also attempt
running Playwright on the actual Streamlit Cloud deployment (fragile,
flagged as out of scope back in Round twelve) is a decision for the
deployer, not something to change unilaterally while fixing a layout bug.

## Round seventeen: compiling every edge case this session surfaced and running them

The deployer asked directly: take everything found so far and actually
run it against the app, rather than treating each finding as closed once
diagnosed. Compiled a list from this session's own history — non-entity
apps, multi-entity apps, id-collision-prone field names, malformed
`localStorage`-shim inputs, filter lists with an ineligible field first,
select fields with duplicate option values — and ran each one directly
against the real pipeline code, not just discussed them. Two were
already correctly handled (a multi-filter list with an ineligible first
entry correctly falls through to the next eligible one; a select field
with duplicate option values gracefully skips the filter check instead
of crashing, via existing defensive code). Two were real, newly-found
bugs, both fixed:

**The CRUD test's own methodology had a false-negative gap.** Running
Round sixteen's actual buggy rent-management app through
`run_browser_functional_test` (with `PLAYWRIGHT_CHROMIUM_EXECUTABLE` set
so it could really execute) didn't fail with the JS-syntax-error signal
expected — it failed on a Playwright timeout trying to fill `#field-name`,
because that app's Tenants section is hidden behind a sidebar-nav click,
and the CRUD phase has always assumed the primary entity's form is
visible on load. Isolating the two issues (temporarily making the
Tenants screen the default one) confirmed the pageerror-based syntax
check *does* work correctly on its own — but the visibility assumption
is a real, previously-undocumented gap that would make a genuinely
well-built multi-screen app (exactly what "build to the best of your
ability" now encourages) fail Testing for a reason that has nothing to
do with a real bug. Fixed two ways: `crud_contract.developer_prompt_block`
now explicitly requires `#add-form`/`#item-list` to be visible without
navigation, and `browser_tester.py`'s auth and CRUD phases now check
`is_visible()` before attempting to interact, turning what used to be an
opaque 5-second-timeout crash into an immediate, specific, actionable
note either way.

**`isolate_local_storage` searched for a substring that isn't reliably
unique.** Built a deliberately adversarial case — a script containing the
literal text `"<body class=nope>fake</body>"` inside a JS string, before
the real `<body>` tag — and confirmed the shim landed inside that fake
match, corrupting the script. The fix removes the search entirely: the
shim is now unconditionally prepended before the whole document (even
before `<!doctype html>`), which a direct check confirmed still executes
before any other inline script in document order — simpler than the
substring search it replaced, and structurally can't be fooled by
content that merely looks like the tag it was looking for.

One finding was flagged but not fixed: `crud_contract.field_id` slugifies
"Due Date," "due_date," and "due-date" to the identical `field-due-date`
— if the Requirements Analyst ever extracted two distinct fields whose
names differ only in separator/case, their ids would collide. Left as a
documented, low-priority gap rather than adding disambiguation logic:
it requires the model to propose two near-duplicate field names for the
same entity, which is unlikely, and the failure mode if it did happen is
a wrong-field read in a test, not a crash.

## What I'd do next with more time

- Let the Architect propose more than one screen/entity and have the
  Developer emit a small multi-page app instead of a single file.
- Stream the Developer's output into the live preview as it's written
  (OpenRouter's endpoint supports streaming) instead of waiting for the
  full reply.
- Set up a real browser on the actual Streamlit Cloud deployment (not just
  this dev sandbox) so the Testing stage's headless-browser checks run
  there too, instead of always reporting "skipped."
- Give the deployer (not visitors — that sidebar UI is gone now) a small
  admin-only way to check the configured `OPENROUTER_API_KEY`/model still
  work, since the old visitor-facing "Test connection" button no longer
  exists.
- Browser-test the filters and actions this round left to Code Review's
  text judgment alone (boolean-field filters, "complete" toggles, search/
  sort) — each needs its own answer to "what's the unambiguous testable
  UI convention for this," the same problem `first_testable_filter`
  solved narrowly for `select`-typed filters.
- Give the open-ended app tier (Round fifteen's smoke test) something
  deeper than "it loaded" — e.g. an optional self-test hook convention the
  Developer can expose for apps whose functionality can't be derived into
  a fixed contract, without going back to trusting an LLM's unverified
  opinion of its own code.
- Disambiguate `crud_contract.field_id` when two distinct field names
  slugify to the same id (Round seventeen) — track ids already used per
  entity and append a suffix on collision, in both the Developer prompt
  and the browser driver so they'd still agree.
- Extend real-execution testing to a second/third entity when
  `Requirements.entities` has more than one (Round seventeen's rent-
  management app had four) — right now only `entities[0]` gets any
  fixed-id contract or browser coverage; the rest rely on Code Review's
  text judgment alone, the same layered-testing gap as boolean filters.
