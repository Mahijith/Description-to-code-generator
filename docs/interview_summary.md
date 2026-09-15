# Project Summary: AI-Driven Description-to-Application Pipeline

**Brainstorming & Process**

**Problem framing.** The assignment: turn an audio/video description into
a working prototype. Rather than one prompt-to-code pass, the problem was
reframed as: can a small team of specialized AI agents, each owning one
part of the SDLC, mirror how a real team builds a draft — including
catching its own mistakes before shipping? That question drove the two
decisions that shaped everything after: a multi-agent pipeline, and a
genuine feedback loop rather than a straight line.

**Architecture, iterated under real constraints.** A first three-stage
design (transcribe → spec → generate) collapsed "understand," "decide,"
and "verify" into one ungoverned step, so it was restructured into five
agents — Project Manager, Requirements Analyst, Architect, Developer,
Code Reviewer — with review findings routed back to the Developer for a
bounded number of fixes before sign-off. A hosting path via a Claude
Artifact was evaluated and rejected after reading the platform's own
type definitions showed no audio support — a real constraint, stated
plainly rather than built around silently. Every agent was routed through
one provider-agnostic interface so the default model could be swapped
with a one-line change; that paid off directly when the default model
was changed six times across a rate-limit issue, an upstream outage, and
repeated truncation bugs, each swap fully isolated. A later bug report —
"asked for a debugging tool, got a rent-management app" — traced to a
prompt that forced every description into an entity/fields/actions
shape; fixed by adding a freeform `features` path so non-records-list
apps build to their actual described shape instead of a forced default.

**Testing infrastructure — the core original contribution.** Replaced
"an LLM's opinion of its own code" with real, automated execution. A
fixed element-id contract derives predictable ids from each app's actual
requirements and is shared verbatim between the Developer's prompt and a
headless-Chromium test driver, so instruction and verification can never
drift apart. That driver exercises registration/login (with reload-based
persistence and duplicate-account checks), full add/edit/delete/filter
flows, or a generic load-and-render smoke test — whichever tier a given
app actually needs. Running a real flagged app through this harness
surfaced a genuine methodology gap (the driver assumed a form was
visible on load, failing a well-built multi-screen app with an opaque
timeout) and a live-preview data-isolation defect, found by testing the
stated scope requirement literally: the preview's HTML component shares
one browser-storage origin across generations, so a new prototype could
silently inherit an old one's data. Verified empirically with a
disposable Streamlit + Playwright probe — which also caught that a plain
JS property reassignment silently no-ops in Chromium — before shipping a
namespaced-storage fix scoped to the preview only.

**Debugging methodology.** When five consecutive model swaps produced an
identical "stops mid-process" symptom, the repetition itself was treated
as the signal, not the models: traced to a missing truncation check, not
model choice. When a proposed fix was challenged as solving the wrong
problem, the response was to instrument the provider's own token-usage
data rather than argue from assertion — the resulting evidence settled
the disagreement and corrected an initially wrong working theory.

**Input validation, iterated toward a dedicated agent.** Off-topic or
scope-less recordings (a greeting, an unrelated remark, a bare "build me
an app" with nothing specified) were initially screened by one rule
folded into the requirements-extraction prompt. Live testing showed this
wasn't reliable — the rule competed for attention against ~40 lines of
unrelated extraction instructions. The fix generalized the check into
one explicit test ("does this name an actual goal to build toward?") and
then relocated it entirely: a dedicated Scope Gate agent now runs first,
before any other stage, with that decision as its only job. Each pipeline
stage owns exactly one responsibility, and the gate blocks wasted work
before it starts.

**Tools and technique.** An agentic coding assistant (Claude Code) drove
implementation and, critically, empirical verification — disposable
servers and Playwright sessions confirming real behavior, not just
source-reading. Backends evaluated: OpenRouter and AIHubMix for
generation, Groq's hosted Whisper API for transcription (replacing an
on-device approach after diagnosing a missing system library on the
target host). Prompts were hardened to require an exact JSON shape with
tolerant fallback parsing (no JSON-mode guarantee on free-tier models),
explicit safety instructions matched by a Code Review checklist item,
and conditional construction so an agent's instructions expand or
contract to what a given app actually needs.

**Outcome.** A working, tested six-agent pipeline producing one
self-contained, runnable file per description, real-execution-tested at
whichever functional tier applies, with every design trade-off and
defect diagnosis documented for traceability.
