"""Serves the page and streams each run to it. Start with `python app.py`."""

import json
import os
from pathlib import Path

import anthropic
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from pipeline import run_pipeline

load_dotenv()
HERE = Path(__file__).parent
app = FastAPI()


def sse(event):
    return f"data: {json.dumps(event)}\n\n"


def error(message):
    return sse({"type": "error", "message": message})


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/run")
def run(paper: str):
    def events():
        if not os.environ.get("ANTHROPIC_API_KEY"):
            yield error("No API key found. Add ANTHROPIC_API_KEY to .env (see .env.example), then restart app.py.")
            return
        try:
            for event in run_pipeline(paper, HERE / "runs"):
                yield sse(event)
        except anthropic.AuthenticationError:
            yield error("Anthropic didn't accept the API key in .env. Check it, then restart app.py.")
        except Exception as e:
            yield error(str(e))

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Open http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
