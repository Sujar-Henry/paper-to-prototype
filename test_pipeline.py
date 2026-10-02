import base64
import time

import pytest

from pipeline import check_result, extract_code, parse_arxiv_id, run_pipeline, run_script


def drain(gen):
    """Collect everything a generator yields, plus what it returns."""
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value


@pytest.mark.parametrize("ref", [
    "1706.03762",
    "arXiv:1706.03762",
    "https://arxiv.org/abs/1706.03762",
    "https://arxiv.org/pdf/1706.03762",
    "https://arxiv.org/pdf/1706.03762.pdf",
    "  https://arxiv.org/abs/1706.03762/  ",
])
def test_parse_arxiv_id(ref):
    assert parse_arxiv_id(ref) == "1706.03762"


def test_parse_arxiv_id_keeps_version():
    assert parse_arxiv_id("https://arxiv.org/abs/1710.09412v2") == "1710.09412v2"


@pytest.mark.parametrize("ref", ["", "attention is all you need", "https://example.com/1706.03762"])
def test_parse_arxiv_id_rejects_other_input(ref):
    assert parse_arxiv_id(ref) is None


def test_extract_code_from_fenced_reply():
    reply = "Here you go:\n```python\nprint('hi')\n```\nGood luck."
    assert extract_code(reply) == "print('hi')\n"


def test_extract_code_from_reply_cut_off_mid_fence():
    assert extract_code("```python\nimport torch\nx = 1\n") == "import torch\nx = 1\n"


def test_extract_code_without_fence_returns_reply():
    assert extract_code("print('hi')\n") == "print('hi')\n"


def test_check_result_passes_on_clean_exit_with_success_line():
    assert check_result(0, "step 1\nSUCCESS: accuracy 0.99\n") == (True, "Exited cleanly and printed SUCCESS")


def test_check_result_needs_success_line():
    ok, reason = check_result(0, "step 1\nloss 0.1\n")
    assert not ok
    assert "SUCCESS" in reason


def test_check_result_fails_on_error_exit():
    assert check_result(1, "Traceback ...\nSUCCESS: fake\n") == (False, "Exited with code 1")


def test_check_result_fails_on_timeout():
    assert check_result(None, "step 1\n") == (False, "Timed out")


def test_run_script_streams_output_and_exit_code(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("print('a')\nprint('b')\nraise SystemExit(3)\n")
    events, (exit_code, output) = drain(run_script(script, timeout=10))
    assert [e["text"] for e in events] == ["a", "b"]
    assert exit_code == 3
    assert output == "a\nb\n"


def test_run_script_captures_tracebacks(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("raise ValueError('bad shape')\n")
    _, (exit_code, output) = drain(run_script(script, timeout=10))
    assert exit_code == 1
    assert "ValueError: bad shape" in output


def test_run_script_kills_it_after_timeout(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("import time\nprint('started')\ntime.sleep(30)\n")
    start = time.monotonic()
    events, (exit_code, output) = drain(run_script(script, timeout=1))
    assert time.monotonic() - start < 10
    assert exit_code is None
    assert events[0]["text"] == "started"
    assert "Killed after 1 seconds" in output


PDF = b"%PDF-1.4 not really a paper"
BROKEN = "```python\nraise RuntimeError('shape mismatch')\n```"
WORKING = "```python\nprint('step 1/1  loss 0.10')\nprint('SUCCESS: accuracy 1.00 > 0.90')\n```"


def fake_download(arxiv_id, dest):
    dest.write_bytes(PDF)


def fake_claude(*replies):
    """Stands in for ask_claude, answering each call with the next reply and remembering what it was sent."""
    def ask(system, content, effort):
        ask.calls.append(content)
        reply = replies[len(ask.calls) - 1]
        yield {"type": "text", "text": reply}
        return reply
    ask.calls = []
    return ask


def test_pipeline_fixes_a_broken_script(tmp_path):
    ask = fake_claude("## Core idea\nattention", "## Toy task\nreversal", BROKEN, WORKING)
    events = list(run_pipeline("https://arxiv.org/abs/1706.03762", tmp_path, ask=ask, download=fake_download))

    folder = tmp_path / "1706.03762"
    assert events[-1] == {"type": "done", "ok": True, "attempts": 2, "folder": str(folder)}
    assert [e["ok"] for e in events if e["type"] == "result"] == [False, True]
    assert "shape mismatch" in ask.calls[3]  # the traceback went back to Claude
    assert (folder / "method.md").read_text() == "## Core idea\nattention"
    assert (folder / "plan.md").read_text() == "## Toy task\nreversal"
    assert (folder / "prototype.py").read_text() == extract_code(WORKING)
    assert (folder / "prototype_1.py").read_text() == extract_code(BROKEN)
    assert "shape mismatch" in (folder / "run_1.log").read_text()
    assert "SUCCESS" in (folder / "run_2.log").read_text()


def test_pipeline_sends_the_pdf_to_claude(tmp_path):
    ask = fake_claude("method", "plan", WORKING)
    list(run_pipeline("1706.03762", tmp_path, ask=ask, download=fake_download))
    document = ask.calls[0][0]
    assert document["type"] == "document"
    assert base64.b64decode(document["source"]["data"]) == PDF


def test_pipeline_gives_up_after_three_fixes(tmp_path):
    ask = fake_claude("method", "plan", BROKEN, BROKEN, BROKEN, BROKEN)
    events = list(run_pipeline("1706.03762", tmp_path, ask=ask, download=fake_download))
    assert events[-1]["ok"] is False
    assert events[-1]["attempts"] == 4
    assert len(ask.calls) == 6


def test_pipeline_rejects_input_that_is_not_an_arxiv_link(tmp_path):
    events = list(run_pipeline("attention is all you need", tmp_path, ask=fake_claude(), download=fake_download))
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "arXiv" in events[0]["message"]


def test_pipeline_reports_a_failed_download(tmp_path):
    def offline(arxiv_id, dest):
        raise OSError("no network")

    events = list(run_pipeline("1706.03762", tmp_path, ask=fake_claude(), download=offline))
    assert events[-1]["type"] == "error"
    assert "no network" in events[-1]["message"]
