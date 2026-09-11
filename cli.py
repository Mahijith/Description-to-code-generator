#!/usr/bin/env python3
"""Headless runner for the pipeline — no Streamlit required.

Usage:
    python cli.py run examples/sample_transcript.txt --out examples/output --mock
    OPENROUTER_API_KEY=sk-... python cli.py run my_recording.mp4 --out output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline.llm import DEFAULT_MODEL, MockLLMProvider, OpenRouterProvider
from pipeline.orchestrator import Orchestrator
from pipeline.secrets import Secrets
from pipeline.transcribe import make_transcriber_for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline on one input file")
    run_p.add_argument("input", help="Path to a .txt/.md transcript, or an audio/video file")
    run_p.add_argument("--out", default="output", help="Output directory")
    run_p.add_argument("--mock", action="store_true", help="Use MockLLMProvider (no API key/network needed)")
    run_p.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id to use (ignored with --mock)")

    args = parser.parse_args()
    if args.command != "run":
        parser.print_help()
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Transcribing {input_path} ...")
    transcriber = make_transcriber_for(input_path)
    transcript = transcriber.transcribe(input_path)
    (out_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    print(f"      {len(transcript)} characters")

    if args.mock:
        llm = MockLLMProvider()
        print("[2/3] Running pipeline with MockLLMProvider (offline demo mode) ...")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("error: set OPENROUTER_API_KEY, or pass --mock to run offline", file=sys.stderr)
            return 1
        llm = OpenRouterProvider(Secrets(api_key), model=args.model)
        print(f"[2/3] Running pipeline via OpenRouter ({args.model}) ...")

    def on_stage(stage: str, status: str) -> None:
        print(f"      - {stage}: {status}")

    orchestrator = Orchestrator(llm)
    result = orchestrator.run(transcript, on_stage=on_stage)

    print("[3/3] Writing output ...")
    (out_dir / "brief.json").write_text(json.dumps(result.brief.to_dict(), indent=2))
    (out_dir / "requirements.json").write_text(json.dumps(result.requirements.to_dict(), indent=2))
    (out_dir / "architecture.json").write_text(json.dumps(result.architecture.to_dict(), indent=2))
    for i, qa in enumerate(result.qa_reports, start=1):
        (out_dir / f"qa_report_iteration_{i}.json").write_text(json.dumps(qa.to_dict(), indent=2))
    (out_dir / "summary.txt").write_text(result.summary, encoding="utf-8")
    (out_dir / "prompt_log.json").write_text(json.dumps(result.prompt_log, indent=2))

    proto_dir = out_dir / "prototype"
    proto_dir.mkdir(exist_ok=True)
    (proto_dir / "index.html").write_text(result.html, encoding="utf-8")

    print(f"Done in {result.iterations} iteration(s). Output written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
