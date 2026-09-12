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
   **OpenRouter** (default model: `inclusionai/ling-3.0-flash-vl:free`),
   not Claude. I verified that model exists at the URL I was given, but I
   could not inspect its actual input modalities or current rate limits —
   `openrouter.ai` is blocked by this sandbox's own network policy for both
   `curl` and a web-fetch tool — so I said that plainly rather than
   asserting specifics I couldn't check, and made the model id a one-line
   config value instead of a hardcoded assumption.
5. **Transcription backend.** "VL" in a model name means Vision-Language
   (text + images), not audio — so the free chat model above can't
   transcribe a recording. Rather than reach for a paid ASR API, I moved
   transcription to `faster-whisper` running locally: free, offline, needs
   no key, and — as a side effect of being a real backend instead of a
   static page — it can handle genuine audio/video file uploads, which the
   Artifact path couldn't.
6. **One key, not several.** Since transcription never leaves the machine
   and every agent goes through the same OpenRouter gateway, the whole app
   needs exactly one external API key, used uniformly — not a different key
   per capability.
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
one) so that Demo mode and the test suite both exercise a *real* two-pass
loop rather than asserting it happened.

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
- **Local Whisper over a cloud transcription API**: slower on first run
  (model download, CPU inference) but genuinely free, keyless, and keeps
  a user's recording on their own machine.
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

## What I'd do next with more time

- Let the Architect propose more than one screen/entity and have the
  Developer emit a small multi-page app instead of a single file.
- Stream the Developer's output into the live preview as it's written
  (OpenRouter's endpoint supports streaming) instead of waiting for the
  full reply.
- Add a second, independent QA pass that actually executes the generated
  JS in a headless browser (the way this project's own Playwright test
  drives the app) rather than relying on a model reading its own code.
- Let a user pick from a short list of known-good free OpenRouter models
  in the sidebar instead of typing a model id by hand, refreshed from
  OpenRouter's models API at deploy time.
