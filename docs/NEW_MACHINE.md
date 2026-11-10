# Continuing on a new computer

Everything is on GitHub except three things git never carries:
- **your secrets** (`.env`);
- **the downloaded market data** (`data/`, about 365 MB);
- **the notes Claude keeps on one computer.** What matters in them is in this repo: here, in
  [`CLAUDE.md`](../CLAUDE.md) and in [`SESSION_NOTES.md`](SESSION_NOTES.md).

This page gets a new computer to where the old one was, in about 15 minutes plus the data.

## 1. What is where

| on GitHub (`talalz94/Trading-Analysis`, public) | not on GitHub | how to get it back |
|---|---|---|
| all code, tests, notebooks and their configs, docs, the Exness indicators | `.env` | step 4: copy it over by hand |
| `requirements.txt`, `.env.example` (key NAMES only) | `data/` | section 3: copy the folder, or download again |
| | `experiments/results/`, `reports/` | made again by running the experiment or notebook |
| | `alerts/state/` | lives on the alerts server; the laptop does not need it |

The repo is **public**: anyone can read the code and the docs, never the secrets. That is why
`.env` and the server's details stay off it.

## 2. Install the tools

1. **Python 3.13** from python.org. On Windows, tick "Add python.exe to PATH".
2. **Git** from git-scm.com.
3. Optional: **VS Code**, and **Claude Code** if you work with Claude (section 6).

## 3. Get the project

```bash
git clone https://github.com/talalz94/Trading-Analysis.git quant
cd quant
git checkout research/scenario-framework-2026-07-30
```

All the work since July 2026 is on that branch, and the alerts server pulls it. `main` was
brought level with it on 11 Nov 2026, so a plain clone is current too. Keep working on the
research branch.

**The data (`data/`).** The quickest way is to copy the `data` folder from the old computer into
`quant/` (zip it: about 365 MB). Or let it download again:
- **Market candles** download by themselves the first time a notebook or `get_ohlcv(...)` asks for
  them, and are kept in `data/` after that. True spot gold back to 2005:
  `python scripts/fetch_gold_dukascopy.py` (slow, resumable).
- **The economic calendar** (docs/NEWS.md, "Getting it running"). The last command needs the
  FRED key from step 4:
  ```bash
  python -m quant.news import-archive
  python -m quant.news update
  python -m quant.news official --validate
  ```

## 4. Python packages and secrets

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell;  macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt     # research, notebooks, alerts, tests
pip install -e .                    # the quant package itself
copy .env.example .env              # macOS / Linux: cp .env.example .env
```

Then open `.env` and fill in the values. Copy them from the old computer's `.env`, or from the
server's `~/quant/.env`, through a password manager. Never send them through git, email or chat.

| key | what it is for | if left empty |
|---|---|---|
| `FRED_API_KEY` | US release dates and the official US numbers (free) | no US actuals in research; the calendar still works |
| `ALERT_SMTP_USER` / `ALERT_SMTP_PASSWORD` / `ALERT_EMAIL_TO` | sending alert emails (a Gmail APP password) | only matters if this computer sends alerts, and it should not (section 5) |
| `NTFY_TOPIC` | the push copy of each alert; acts like a password | same |
| `JBLANKED_API_KEY`, `BLS_API_KEY` | optional extras (see `.env.example`) | nothing breaks |
| `BINANCE_API_KEY` / `_SECRET` | not needed: market data uses Binance's public API | nothing breaks |

## 5. Jupyter, and the check

```bash
python -m ipykernel install --user --name python3 --display-name "Python 3 (3.13 - quant)"
python -m pytest tests/
```

The first command makes the notebooks run on this project's Python (CLAUDE.md, section 9,
explains why that matters).

**What the test run should show.** Measured on 11 Nov 2026, on a fresh clone in a fresh
environment, with no `data/` and no `.env`: **1,330 pass, 27 skip, nothing fails.**
- 9 skips happen on every computer: the Exness gap rows, which run in their own test file.
- 18 skips say why: 16 need the cached BTC candles, and 2 need the imported calendar.
- Once `data/` is copied over: 1,348 pass and 9 skip, as on the old computer.

That run also caught the one package `requirements.txt` was missing (`anywidget`, which the live
charts need), so a test failing here means a real problem to report, not noise.

Start with `notebooks/lab.ipynb` (one strategy on one window) and `notebooks/evaluate.ipynb` (the
same strategy over 19 quarters). [`NOTEBOOK_GUIDE.md`](NOTEBOOK_GUIDE.md) explains the layout.

## 6. The live pieces (they do not move with the laptop)

- **The email alerts run on the Google Cloud server** (the VM named `quant-alerts`), not on a
  laptop.
  - **Do not start the alerts monitor, or its Windows scheduled task, on the new computer while
    the server runs:** every alert would arrive twice.
  - Reach the server from any browser: Google Cloud console > Compute Engine > VM instances >
    **SSH**.
  - Day to day, and updating with a way back: [`ALERTS_CLOUD.md`](ALERTS_CLOUD.md), "Day to day".
- **Deploying a change:** commit and push from the laptop first. Then, on the server, follow
  "Updating, with a way back" in ALERTS_CLOUD.md:
  1. back up;
  2. `git pull` + `bash scripts/install_alerts_linux.sh`;
  3. copy `alerts/alerts_config_trendlines.py` over `alerts/alerts_config.py` (the monitor reads
     only `alerts_config.py`) and check that it loads;
  4. check the status and the log;
  5. keep the rollback line at hand.
- **The calendar on the laptop:** for research only. Install the Windows task that keeps it
  fresh with `powershell -ExecutionPolicy Bypass -File scripts\install_news_task.ps1`. This one
  is safe: it sends nothing.
- **The Exness indicators** are pasted into Exness's own editor from `exness/`, so nothing on the
  laptop runs them. Which file does what, and the paste order: [`EXNESS.md`](EXNESS.md).

## 7. Continuing with Claude

- **Claude Code reads [`CLAUDE.md`](../CLAUDE.md) by itself:** the map of the project and the
  working agreements (section 14).
- **[`SESSION_NOTES.md`](SESSION_NOTES.md) is the running log,** newest at the bottom.
- **[`WHAT_WORKS.md`](WHAT_WORKS.md)** is the verdict on every strategy idea already tested.
- **How the work goes:**
  - explanations step by step, on one real trade or release, no jargon;
  - configs short;
  - no setting changed unless you asked;
  - every server update comes with the backup, check and rollback lines above, and says whether
    the alerts are affected.

**Open as of 11 Nov 2026:**
1. **Exness:** "Too many concurrent indicator runs" with five indicators on one chart. The test to
   run first: replace "100 SMA Custom" with Exness's built-in Moving Average (100, Simple), then
   add the other four again.
   - If they all run, that is the fix.
   - If not, send the "Show error" text. The plan is then one indicator that combines Trendlines
     MTF, Structure and Bias.
2. **A new FRED key**, whenever it arrives. Put it in the server's `~/quant/.env` and in the
   laptop's `.env`, then `sudo systemctl restart quant-alerts` on the server.
3. **Parked strategy ideas.** Do not start them unasked: SESSION_NOTES.md, "Parked ideas".
