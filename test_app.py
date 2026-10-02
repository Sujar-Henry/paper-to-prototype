import json

import anthropic
import httpx2
from fastapi.testclient import TestClient

import app


def get_events(paper):
    response = TestClient(app.app).get("/run", params={"paper": paper})
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_run_without_an_api_key_explains_where_to_put_it(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    [event] = get_events("1706.03762")
    assert event["type"] == "error"
    assert "ANTHROPIC_API_KEY" in event["message"]
    assert ".env" in event["message"]


def test_run_with_a_rejected_api_key_says_so(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-wrong")

    def rejected(paper, runs_dir):
        yield {"type": "start", "step": "extract"}
        response = httpx2.Response(401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
        raise anthropic.AuthenticationError("API key is invalid.", response=response, body=None)

    monkeypatch.setattr(app, "run_pipeline", rejected)
    events = get_events("1706.03762")
    assert events[-1]["type"] == "error"
    assert "didn't accept the API key" in events[-1]["message"]
