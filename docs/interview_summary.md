# Brainstorming & Process Summary

The system turns a spoken description into a working application through six agents that each handle one stage of the process: a Scope Gate, a Project Manager, a Requirements Analyst, an Architect, a Developer, and a Code Reviewer, followed by an automated Testing stage. Each stage passes its output to the next, and the Developer's result is verified and refined for up to three passes before the final application is produced — a single, self-contained file that runs immediately in a browser, with no server or build step required.

The Scope Gate runs first, before any other stage. It reads the transcript and makes one decision: does it describe an actual goal or idea for an application? Only when it does does the pipeline continue to requirements extraction and build; if the recording doesn't describe anything to build, the person is told so directly, rather than having the pipeline generate an unrelated result.

Testing is carried out by actually running the generated application in a real browser, not by asking a model to review its own code. Depending on what the application needs, the test drives it through the relevant flow: registering and logging in a user for apps with accounts, or adding, editing, filtering, and deleting a record for apps that manage data, confirming each step produces the expected result on screen. This is what allows the three-pass build cycle to catch and correct real issues before the application is handed over, producing an executable result each time.

Development was done using Claude Code as the primary AI tool, working iteratively: each stage's prompt was refined by testing it against real generated output and observing the result, rather than designing it once and assuming it would hold. The model backend (OpenRouter, running free-tier language models) and the transcription backend (Groq's hosted Whisper API) were each selected the same way — evaluated directly, then adopted once they performed reliably for their part of the pipeline.

The result is a complete, working pipeline: a recording goes in, and a functioning, ready-to-run application comes out.
