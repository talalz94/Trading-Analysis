# Alerts in the cloud: Google Cloud's free tier, step by step

This runs the alerts monitor on a free Google Cloud server, so it keeps working while your PC is
off. Budget about 40 minutes. Steps 1–4 happen in the browser; from step 5 you type into a terminal
on the server itself, and an install script does the heavy lifting.

## Is it really free?

Google has two different offers. Only the second is the one this guide uses:

| offer | what | how long |
|---|---|---|
| Free Trial | $300 credit for any service | 90 days |
| **Free Tier** | **one `e2-micro` server (2 shared vCPUs, 1 GB RAM) + a 30 GB standard disk + 1 GB of outbound traffic a month, in Oregon, Iowa or South Carolina** | **no end date** |

There are four conditions to keep it free:
- **Activate the full account** (step 2). A trial that simply ends switches billing off and you
  lose the server. A full account keeps the Free Tier and charges only for usage above it.
- **Use exactly the settings in step 3:** an `e2-micro` in one of the three US regions, with a
  *Standard* disk. Google's default disk type, *Balanced*, is charged.
- **Watch the server's public IP address.** Older Google documentation says it is free on the
  free server, but at least one source says Google charges it even there: about $3.65 a month at
  worst. Step 11 checks your bill so you know for certain.
- **Google can change the offer** with 30 days' notice. The budget alert in step 2 emails you if
  anything ever costs money.

What the monitor needs fits easily:
- **Memory:** it peaks at about 130 MB.
- **Traffic:** by my estimate a few hundred MB a month at most, and downloads don't count.

**The one real limitation: the free server is in the US, and Binance refuses US servers.** The
monitor then tries Binance's market-data address, which gives identical candles (checked 71 of
71), and switches to it by itself. Step 7 tells you whether that works. If Binance refuses that
too, BTC comes from Dukascopy. Gold and USDJPY come from Dukascopy anyway, so they are never
affected.

**Before you start:** don't run the monitor on your PC as well, or every alert arrives twice. If
you installed the Windows task, remove it:
`powershell -ExecutionPolicy Bypass -File scripts\install_alerts_task.ps1 -Uninstall`

---

## Step 0: put the code on GitHub (on your PC, once)

The server downloads the code from GitHub, and the alerts code lives on a branch that is not on
GitHub yet:

```bash
git push -u origin research/scenario-framework-2026-07-30
```

The repo is public. Your `.env` (the email password), `data/` and `alerts/state/` are git-ignored
and never uploaded.

## Step 1: create the Google Cloud account (10 min)

1. Open **https://console.cloud.google.com/freetrial** and sign in with your Gmail.
2. **Country:** choose the country your card is billed in, which for your Dubai card is
   **United Arab Emirates**. A billing account's country cannot be changed later. Accept the
   terms, then **Continue**.
3. **Account type:** *Individual*. Enter your name and the **billing address exactly as your bank
   has it**.
4. Enter the card and click **Start free**. Google may place a small temporary hold of about $1,
   which is released. Don't use a VPN during sign-up, since that is a common reason for refusals.
5. You land in the Google Cloud console, with a project called *My First Project*.

## Step 2: keep it free after 90 days, and add a safety net (5 min)

1. Click **Activate full account** in the banner at the top. If there is no banner, go to
   ☰ → **Billing** → **Overview**. Do it now so you can't forget: you keep the $300 credit, keep
   the Free Tier, and are charged only for usage above it.
2. Go to ☰ → **Billing** → **Budgets & alerts** → **Create budget**:
   - name `safety`, all projects;
   - under *Credits*, **untick "Promotions and others"**, so the alert reacts to real usage even
     while the trial credit would cover it;
   - amount: *Specified amount*, **1** (USD);
   - thresholds: 50%, 90% and 100% of actual spend, emailed to you. Click **Finish**.

## Step 3: create the free server (5 min)

1. Go to ☰ → **Compute Engine** → **VM instances**. The first time, click **Enable** for the
   *Compute Engine API* and wait a minute.
2. Click **Create instance**.
3. Under **Machine configuration**:
   - **Name:** `quant-alerts`
   - **Region:** `us-central1 (Iowa)`. Only us-central1, us-west1 and us-east1 are free. Leave
     the zone as it is.
   - **General purpose**, series **E2**, machine type **e2-micro** (2 vCPU, 1 GB memory).
   - Leave *VM provisioning model* on **Standard**, not Spot.
   - The price estimate on the right (about $7 a month) does **not** subtract the Free Tier.
     Ignore it, as long as the region, type and disk match this guide.
4. Under **OS and storage**, click **Change**:
   - Operating system **Ubuntu**, version **Ubuntu 24.04 LTS** (x86/64). Pick the plain one, not
     *Minimal* or *Pro*.
   - **Boot disk type: Standard persistent disk.** This one matters: the default, *Balanced*,
     is charged.
   - **Size: 30** GB. Click **Select**.
5. Under **Data protection**, choose **No backups**.
6. Under **Networking**, leave the defaults. The server gets a public IP for outgoing traffic.
   No firewall boxes need ticking, because nothing connects *in* to the monitor.
7. Under **Observability**, untick **Install Ops Agent**. That saves memory on a 1 GB machine.
8. Click **Create** and wait for the green tick.

If the zone says e2-micro is unavailable, pick another zone, or another of the three free
regions.

## Step 4: open a terminal on the server

In the VM instances list, click **SSH** on the `quant-alerts` row and allow it. A terminal opens in
your browser; there are no key files to manage. **Everything below is typed there.** The prompt
looks like `yourname@quant-alerts:~$`.

## Step 5: prepare the server (3 min)

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install git python3-venv
```

If a purple screen asks which services to restart, press Enter. Then add 1 GB of swap, a safety
margin for a 1 GB machine:

```bash
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 6: get the code

```bash
git clone -b research/scenario-framework-2026-07-30 https://github.com/talalz94/Trading-Analysis.git quant
cd quant
```

## Step 7: does Binance answer this server?

```bash
curl -s -o /dev/null -w "main host: %{http_code}\n" "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1"
curl -s -o /dev/null -w "data host: %{http_code}\n" "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1"
```

- **Either prints `200`:** nothing to do. The monitor uses whichever host answers, by itself, and
  the BTC alerts are Binance's own candles.
- **Both print `451` or `403`:** Binance refuses this server. Run `nano alerts/alerts_config.py`
  and change the BTC line to `"BTC": dict(symbol="BTC/USD", source="dukascopy"),`. Save with
  **Ctrl+O, Enter, Ctrl+X**.

  Dukascopy's BTC trades around the clock and sits within about 0.03% of Binance's price. Its
  candle-pattern alerts do not always match Binance's: over 20 days, 60 of 108 hourly alerts
  matched.

## Step 8: email settings

```bash
nano .env
```

Type these three lines with your own values (the same as on your PC), then save with **Ctrl+O,
Enter, Ctrl+X**:

```
ALERT_SMTP_USER=you@gmail.com
ALERT_SMTP_PASSWORD=yourappsixteenletters
ALERT_EMAIL_TO=you@gmail.com
```

For news **results** by email, add the same `FRED_API_KEY=...` line as on your PC: it gives the
US actuals (from the BLS, FRED and the Fed) and official US dates beyond this week. The server
fetches the last 150 days of numbers on its first run (about a minute) and only what is new after
that. `JBLANKED_API_KEY=...` adds other currencies' actuals, but only with paid credits.

Tip: create a **second** Gmail App Password called *quant alerts cloud*. You can then cut the
server off without touching your PC.

## Step 9: install and start

```bash
bash scripts/install_alerts_linux.sh
```

The script:
1. creates a Python environment in `.venv`;
2. installs only what the monitor needs (`requirements-alerts.txt`) — the first run takes a few
   minutes;
3. checks the config loads;
4. registers the monitor as a service that **starts at every boot and restarts 30 s after any
   crash**;
5. starts it.

It ends with **`quant-alerts is RUNNING`**.

## Step 10: check it works

```bash
.venv/bin/python -m quant.alerts test-email   # an email should arrive
.venv/bin/python -m quant.alerts check        # what WOULD have fired in the last 48 h; sends nothing
.venv/bin/python -m quant.alerts status       # "monitor running", and a heartbeat under 2 min old
```

Then close the browser terminal. **The server keeps running.**

## Step 11: the next day, check the bill

Go to ☰ → **Billing** → **Reports**, group by **SKU**, for the last 1–2 days. Everything should be
**$0.00** or covered by credits.

If you see **"External IP Charge on a Standard VM"**, Google is charging the public IP at about
$3.65 a month. Tell me, and we'll decide whether to keep it or change the setup.

---

## Day to day

Open the terminal with the **SSH** button (step 4), then:

| to | type |
|---|---|
| change alerts | `nano ~/quant/alerts/alerts_config.py` and save. It is used within a minute, with no restart. Run `check` first. |
| see what it is doing | `cd ~/quant && .venv/bin/python -m quant.alerts status` |
| update to newer code | see **Updating, with a way back** below |
| stop / start | `sudo systemctl stop quant-alerts` / `sudo systemctl start quant-alerts` |
| startup errors | `journalctl -u quant-alerts -n 50 --no-pager` |
| remove it | `bash scripts/install_alerts_linux.sh --uninstall` |

### Updating, with a way back

Record what runs now, update, and keep the command that restores it:

```bash
cd ~/quant && git rev-parse --short HEAD > ~/code_before.txt && cp alerts/alerts_config.py ~/alerts_config_before.py
git pull && bash scripts/install_alerts_linux.sh
cp alerts/alerts_config_trendlines.py alerts/alerts_config.py      # only if you run that config
.venv/bin/python -c "from quant.alerts.config import load_config; load_config(); print('config OK')"
.venv/bin/python -m quant.alerts status | head -6
```

`git pull` works with your edited `alerts/alerts_config.py` as long as the update does not change
that file itself (it changes `alerts_config_trendlines.py`, which you copy over it). To go back:

```bash
cd ~/quant && git checkout $(cat ~/code_before.txt) && cp ~/alerts_config_before.py alerts/alerts_config.py && bash scripts/install_alerts_linux.sh
```

and later `git checkout research/scenario-framework-2026-07-30` to return to the latest.

### Push notifications on your phone and PC (ntfy)

Every alert email can also pop up on your phone and PC (docs/ALERTS.md, "Push notifications"). The
topic name works like a password, so it goes only in `.env`:

```bash
cd ~/quant && sed -i -e '$a\' .env && echo "NTFY_TOPIC=alerts-$(openssl rand -hex 12)" >> .env && grep NTFY_TOPIC .env
```

Subscribe to that topic in the ntfy app on your phone and at https://ntfy.sh/app on your PC, then
`.venv/bin/python -m quant.alerts test-push` sends a test, and `push="ntfy"` in `DELIVERY` turns
it on for every email (`sudo systemctl restart quant-alerts` once, so the monitor reads `.env`).

### The headline probe (a 7-day test of news sources)

`experiments/headline_probe.py` records 12 free headline sources (FinancialJuice RSS and Telegram,
@WalterBloomberg, investingLive, Google News searches, the Fed, State Department, White House) for
7 days, as its own service beside the alerts: its own database, capped at 250 MB and 10% CPU.

```bash
bash scripts/install_headline_probe.sh                          # tests each source, then starts it
.venv/bin/python experiments/headline_probe.py --status         # what it has recorded
PYTHONPATH=. .venv/bin/python experiments/headline_report.py    # speed / coverage / noise / reliability
bash scripts/install_headline_probe.sh --uninstall              # stop and remove it
```

The State Department's press releases refuse Google Cloud's addresses (403); the probe retries them
every 15 minutes and the report lists them as blocked.

### Switch to the trendline alerts (breaks at once, the rest every 5 minutes at most)

`alerts/alerts_config_trendlines.py` watches trendlines on 1m-4h (and their early breaks), wedges
and triangles, and news reminders from a day ahead. A break is emailed as soon as its candle
closes; everything else at most once every 5 minutes, with tiered subjects and pictures. It needs
the code of 23 Oct 2026 or later (older code refuses `priority="instant"`), so update first,
then copy it over your config (the old one is kept):

```bash
cd ~/quant && git stash && git pull && git stash pop && bash scripts/install_alerts_linux.sh
cp alerts/alerts_config.py alerts/alerts_config_before_trendlines.py
cp alerts/alerts_config_trendlines.py alerts/alerts_config.py
.venv/bin/python -c "from quant.alerts.config import load_config; load_config(); print('config OK')"
.venv/bin/python -m quant.alerts status
```

Back to the old alerts: `cp alerts/alerts_config_before_trendlines.py alerts/alerts_config.py`.

### Forecasts your PC missed

Forex Factory's feed only ever shows the current week, so a week in which nothing fetched it loses
its forecasts for good. The server never sleeps, and the monitor logs the feed every 30 minutes, so
its log has every week. To fill your PC's research calendar from it:

1. In the SSH window, click the gear (⚙) → **Download file**, and enter
   `/home/<your user name>/quant/alerts/state/news/feed_log.parquet`
   (`echo ~` in the terminal prints the first part).
2. On your PC: `python -m quant.news import-log C:\Users\<you>\Downloads\feed_log.parquet`

Importing twice changes nothing, so do it whenever you like.

## If something goes wrong

- **The install stops at "No email settings":** step 8 was skipped or the password line is empty.
- **`status` shows a feed error for BTC:** Binance refuses both hosts. Do step 7's switch.
- **The install says "did not stay up", and the journal shows "another monitor ... is already
  running":** a second copy holds the lock -- typically a user-level service from a setup done
  without sudo, which keeps running old code and an old config. `systemctl --user disable --now
  quant-alerts`, then run the install again (it now does this itself).
- **The SSH button keeps failing:** wait a minute after creating the server, then try again.
  Stop/Start on the VM page also helps.
- **A charge appears:** check the server is e2-micro, in us-central1, us-west1 or us-east1, on a
  *Standard* disk, and that there are no snapshots.

## Other providers

From step 5 on, everything works on **any Ubuntu 24.04 server**. Only how you get a terminal
differs:

- **Oracle Cloud Always Free** (2 Arm CPUs / 12 GB, a Dubai or EU region, so no Binance problem).
  - Create a VM.Standard.A1.Flex or VM.Standard.E2.1.Micro server and download its SSH key.
  - Upgrade to Pay As You Go, or Oracle reclaims "idle" free servers.
  - Log in with `ssh -i <key> ubuntu@<ip>`.
  - Oracle's sign-up refuses many cards. It refused this one from Pakistan.
- **A paid server that logs you in as `root`** (Vultr, Hetzner, DigitalOcean...). The install
  script refuses root, so first create a normal user:
  `adduser alerts && usermod -aG sudo alerts && su - alerts`, then continue from step 5.
