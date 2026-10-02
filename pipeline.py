"""Paper to prototype: read an arXiv paper, plan a toy version, write it in PyTorch, run it."""

import base64
import os
import re
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import anthropic

import prompts

MODEL = "claude-opus-5-5"
ARXIV_ID = re.compile(r"^(?:arxiv:|https?://arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?/?$", re.I)
FENCED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.S)


def parse_arxiv_id(ref):
    match = ARXIV_ID.match(ref.strip())
    return match.group(1) if match else None


def extract_code(reply):
    """Pull the script out of a ```python fence. A reply cut off mid-fence still counts."""
    blocks = FENCED.findall(reply)
    return max(blocks, key=len) if blocks else reply


def check_result(exit_code, output):
    """A run only passes if it exits cleanly and the script's own check printed SUCCESS."""
    if exit_code is None:
        return False, "Timed out"
    if exit_code != 0:
        return False, f"Exited with code {exit_code}"
    if not any(line.startswith("SUCCESS:") for line in output.splitlines()):
        return False, "Finished without printing a SUCCESS line"
    return True, "Exited cleanly and printed SUCCESS"


def run_script(path, timeout):
    """Run a Python file, yielding a log event per line of output.

    Returns (exit_code, output). exit_code is None if the script was killed for running too long.
    """
    proc = subprocess.Popen(
        [sys.executable, path.name], cwd=path.parent,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, kill)
    timer.start()
    output = []
    try:
        for line in proc.stdout:
            output.append(line)
            yield {"type": "log", "text": line.rstrip("\n")}
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:  # the browser went away mid-run
            proc.kill()

    if timed_out.is_set():
        note = f"Killed after {timeout} seconds"
        output.append(note + "\n")
        yield {"type": "log", "text": note}
        return None, "".join(output)
    return proc.returncode, "".join(output)


def download_pdf(arxiv_id, dest):
    request = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "paper-to-prototype"})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def ask_claude(system, content, effort):
    """Stream Claude's reply as text events. Returns the whole reply."""
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=os.environ.get("MODEL", MODEL),
        max_tokens=64000,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": effort},
        # If the model turns the request down for safety reasons, the API retries it on another model.
        fallbacks="default",
        betas=["server-side-fallback-2026-07-01"],
    ) as stream:
        for text in stream.text_stream:
            yield {"type": "text", "text": text}
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request.")
    return "".join(block.text for block in message.content if block.type == "text")


def run_pipeline(ref, runs_dir, ask=ask_claude, download=download_pdf, max_fixes=3, timeout=180):
    """Take one paper from PDF to a working script, yielding events for the page as it goes."""
    arxiv_id = parse_arxiv_id(ref)
    if not arxiv_id:
        yield {"type": "error", "message": "That doesn't look like an arXiv link or ID, e.g. 1706.03762."}
        return

    folder = Path(runs_dir) / arxiv_id
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / "paper.pdf"
    if not pdf.exists():
        try:
            download(arxiv_id, pdf)
        except OSError as e:
            yield {"type": "error", "message": f"Couldn't download the PDF from arXiv: {e}"}
            return

    yield {"type": "start", "step": "extract"}
    paper = {"type": "document", "source": {
        "type": "base64", "media_type": "application/pdf",
        "data": base64.standard_b64encode(pdf.read_bytes()).decode(),
    }}
    method = yield from ask(prompts.EXTRACT, [paper, {"type": "text", "text": "Extract the method."}], "medium")
    (folder / "method.md").write_text(method)
    yield {"type": "output", "step": "extract", "text": method}

    yield {"type": "start", "step": "plan"}
    plan = yield from ask(prompts.PLAN, f"<method>\n{method}\n</method>", "medium")
    (folder / "plan.md").write_text(plan)
    yield {"type": "output", "step": "plan", "text": plan}

    yield {"type": "start", "step": "code"}
    reply = yield from ask(prompts.CODE, f"<method>\n{method}\n</method>\n\n<plan>\n{plan}\n</plan>", "high")
    code = extract_code(reply)
    script = folder / "prototype.py"
    script.write_text(code)
    yield {"type": "output", "step": "code", "text": code}

    yield {"type": "start", "step": "run"}
    attempts = max_fixes + 1
    for attempt in range(1, attempts + 1):
        yield {"type": "attempt", "n": attempt, "of": attempts}
        exit_code, output = yield from run_script(script, timeout)
        (folder / f"run_{attempt}.log").write_text(output)
        ok, reason = check_result(exit_code, output)
        yield {"type": "result", "ok": ok, "reason": reason}
        if ok or attempt == attempts:
            break

        # Keep the failed version, then send it back with the end of its output (where the traceback is).
        (folder / f"prototype_{attempt}.py").write_text(code)
        yield {"type": "fix", "attempt": attempt}
        last_lines = "\n".join(output.splitlines()[-80:])
        request = f"<script>\n{code}\n</script>\n\n<output>\n{last_lines}\n</output>\n\n{reason}."
        reply = yield from ask(prompts.FIX, request, "high")
        code = extract_code(reply)
        script.write_text(code)
        yield {"type": "output", "step": "code", "text": code}

    yield {"type": "done", "ok": ok, "attempts": attempt, "folder": str(folder)}
