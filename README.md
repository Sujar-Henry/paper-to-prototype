# Paper to prototype

Give it an arXiv paper and it tries to build a small, working version of the paper's main idea in PyTorch.

It works in four steps, and you can watch each one in the browser:

1. Claude reads the PDF and writes down how the method works, including the architecture and the loss.
2. It plans a toy version that trains on a laptop CPU in under a minute and checks its own result at the end.
3. It writes that as one PyTorch script.
4. The script runs on your machine. If it crashes or misses its target, the error goes back to Claude for a fix, up to three times.

A run only counts as working if the script exits cleanly and its own check prints a `SUCCESS:` line.

## Setup

You need Python 3.10 or newer and an [Anthropic API key](https://console.anthropic.com/).

Clone the repo, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add your key to .env
python app.py
```

Open http://127.0.0.1:8000 and paste a link like `https://arxiv.org/abs/1706.03762`.
Opening `http://127.0.0.1:8000/?paper=1706.03762` starts a run straight away.

PyTorch is a big download. If you already have it installed, `python3 -m venv --system-site-packages .venv` lets the venv reuse it.

## What gets saved

Each paper gets a folder in `runs/`, named after its arXiv ID:

- `paper.pdf`
- `method.md` and `plan.md`, from steps 1 and 2
- `prototype.py`, the final script
- `prototype_1.py`, `prototype_2.py`, ... for versions that failed
- `run_1.log`, `run_2.log`, ... with the output of each attempt

## Settings

These go in `.env`:

- `ANTHROPIC_API_KEY` (required)
- `MODEL`, which defaults to `claude-opus-5-5`. `claude-sonnet-5-5` is cheaper.
- `PORT`, which defaults to 8000

## Before you run it

The generated script runs as a normal Python process on your computer. Claude is told to use only torch and numpy and to leave your files and the network alone, but nothing enforces that. Treat it like running code from someone you don't know.

## Tests

```bash
pytest
```

The tests swap Claude for canned replies, so they don't need a key or a network connection.

## Files

- `pipeline.py` has the four steps and the run-and-fix loop
- `prompts.py` is what each step asks Claude
- `app.py` is a small FastAPI server that streams a run to the page
- `index.html` is the page
