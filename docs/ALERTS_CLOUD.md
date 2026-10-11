# Alerts in the cloud: Oracle Cloud's free tier, step by step

This runs the alerts monitor on a free Oracle Cloud server, so it works while your PC is off.
Budget about 45 minutes, most of it waiting for Oracle. Everything after step 4 is typed on the
server, and the server's install script does the heavy lifting.

**Before you start:** don't run the monitor on your PC as well, or every alert arrives twice. If
you installed the Windows task, remove it first:
`powershell -ExecutionPolicy Bypass -File scripts\install_alerts_task.ps1 -Uninstall`

---

## Step 0: put the code on GitHub (on your PC, once)

The server downloads the code from GitHub, and the alerts code lives on the branch
`research/scenario-framework-2026-07-30`, which is not on GitHub yet. Push it:

```bash
git push -u origin research/scenario-framework-2026-07-30
```

The repo is **public**. Your `.env` (the email password), `data/` and `alerts/state/` are
git-ignored and never uploaded. The alerts config is uploaded; it holds no secrets.

## Step 1: create the Oracle account (10 min, then wait for an email)

1. Go to **https://signup.cloud.oracle.com**. Enter your country, name and email, then click the
   verification link Oracle emails you.
2. Choose a password and a **Cloud Account Name**. You log in with it, so write it down.
3. **Home Region.** This one matters:
   - It is permanent, and free servers can only be created in it.
   - **Do not pick a US region:** Binance refuses data to US servers.
   - Germany Central (Frankfurt) or UAE East (Dubai) are reasonable choices. Step 7 tests
     whichever you pick.
4. Enter your address and mobile number.
5. Enter a card:
   - It must be a **credit card, or a debit card that works like one**. PIN-only debit,
     prepaid and virtual cards are refused.
   - Oracle places a small temporary hold, released in 3–5 days, and charges nothing.
6. Accept the terms and start the free trial. Wait for the **"Your account is ready"** email. That
   is usually minutes, sometimes a few hours.

## Step 2: make the free server safe from "idle" shutdown (5 min)

Oracle may stop a free server that sits under 20% use for a week, and the monitor uses far less.
The fix users report is to upgrade to **Pay As You Go**. Oracle says Always Free resources are not
charged after the upgrade, only usage above the free limits.

1. Sign in at **https://cloud.oracle.com**: Cloud Account Name, then your email and password.
2. Click **Upgrade** in the banner at the top of the console. If there is no banner, use
   ☰ → **Billing & Cost Management** → **Upgrade and Manage Payment**.
3. Choose **Pay As You Go**, then **Individual**, confirm the card, and **Upgrade your account**.
   Approval can take minutes or a few days; carry on with step 3 meanwhile.
4. **Safety net** (recommended): ☰ → **Billing & Cost Management** → **Budgets** →
   **Create Budget**:
   - schedule *Monthly*, amount **1** (USD);
   - alert rule: actual spend above **1%**, sent to your email.

   If anything ever starts to cost money, you hear about it immediately.

## Step 3: create the server (5 min)

1. ☰ → **Compute** → **Instances** → **Create instance**.
2. **Name:** `quant-alerts`.
3. **Shape:** under *Image and shape* click **Change shape**, then **Ampere**, then
   **VM.Standard.A1.Flex**. Set **1 OCPU** and **6 GB** memory, then **Select shape**. That is
   well inside the free allowance and marked *Always Free-eligible*.
4. **Image:** click **Change image**, then **Canonical Ubuntu**, then **24.04**, then **Select
   image**. For an Ampere shape the console picks the ARM build itself.
5. **Networking:** leave the defaults: a new virtual cloud network, a new **public** subnet, and
   **Automatically assign public IPv4 address** switched on.
6. **SSH keys:** choose **Generate a key pair for me** and click **Download private key**. *Without
   this file you can never log in*, so keep it.
7. **Boot volume:** leave the default. Click **Create**.
8. Wait until the state turns **RUNNING**, then copy the **Public IP address** from the instance
   page.

**If it says "Out of capacity for shape VM.Standard.A1.Flex":** free ARM servers are often
sold out in busy regions. Do one of these:
- pick another *Availability domain* under *Placement*;
- try again later;
- use the small AMD server instead: **Change shape** → **Specialty and previous generation**
  → **VM.Standard.E2.1.Micro**. It is also Always Free, and its 1 GB is plenty: the monitor peaks
  at about 130 MB.

## Step 4: connect from your PC (PowerShell)

Move the key you downloaded into place, and make it readable by you alone. Windows' `ssh` refuses
a key other users could read.

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh" | Out-Null
Move-Item "$env:USERPROFILE\Downloads\ssh-key-*.key" "$env:USERPROFILE\.ssh\oracle-quant.key"
icacls "$env:USERPROFILE\.ssh\oracle-quant.key" /inheritance:r /grant:r "$($env:USERNAME):R"
```

Then log in, replacing `YOUR_PUBLIC_IP` with the IP from step 3:

```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle-quant.key" ubuntu@YOUR_PUBLIC_IP
```

Type `yes` the first time. The prompt becomes `ubuntu@quant-alerts:~$`, and **everything below is
typed there**.

## Step 5: prepare the server

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install git python3-venv
```

If a purple screen asks which services to restart, press Enter.

## Step 6: get the code

```bash
git clone -b research/scenario-framework-2026-07-30 https://github.com/talalz94/Trading-Analysis.git quant
cd quant
```

## Step 7: does Binance answer this server?

```bash
curl -s "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1"; echo
```

- **Starts with `[[`:** Binance answers this server, and there is nothing to change.
- **Says "restricted location", or prints nothing:** Binance blocks this server. Watch BTC through
  Dukascopy instead:

  ```bash
  nano alerts/alerts_config.py
  ```

  Change the BTC line to `"BTC": dict(symbol="BTC/USD", source="dukascopy"),`, then save with
  **Ctrl+O, Enter, Ctrl+X**.

  Dukascopy's BTC trades around the clock, within about 0.03% of Binance's price, and its 1h and 4h
  candles are the same colour 96–98% of the time. Pattern alerts still differ: over 20 days only
  60 of Binance's 108 hourly candle alerts also fired on Dukascopy.

## Step 8: email settings

```bash
nano .env
```

Type these three lines with your own values (the same as your PC's `.env`), then save with
**Ctrl+O, Enter, Ctrl+X**:

```
ALERT_SMTP_USER=you@gmail.com
ALERT_SMTP_PASSWORD=yourappsixteenletters
ALERT_EMAIL_TO=you@gmail.com
```

Tip: create a **second** Gmail App Password called *quant alerts cloud*. You can then cut the
server off later without touching your PC.

## Step 9: install and start

```bash
bash scripts/install_alerts_linux.sh
```

The script:
1. creates a Python environment in `.venv`;
2. installs only what the monitor needs (`requirements-alerts.txt`) — the first run takes a few
   minutes;
3. checks the config loads;
4. registers the monitor as a system service that **starts at every boot and restarts 30 s after
   any crash**;
5. starts it.

It ends with **`quant-alerts is RUNNING`**.

## Step 10: check it works

```bash
.venv/bin/python -m quant.alerts test-email   # an email should arrive
.venv/bin/python -m quant.alerts check        # what WOULD have fired in the last 48 h; sends nothing
.venv/bin/python -m quant.alerts status       # "monitor running", and a heartbeat under 2 min old
```

`tail -f alerts/state/alerts.log` shows it working; press Ctrl+C to stop watching. Then type
`exit`. **The server keeps running after you disconnect.**

---

## Day to day

| to | do (after `ssh` in, step 4) |
|---|---|
| change alerts | `nano ~/quant/alerts/alerts_config.py` and save. It is used within a minute, with no restart. Run `check` first. |
| see what it is doing | `cd ~/quant && .venv/bin/python -m quant.alerts status` |
| update to newer code | `cd ~/quant && git stash && git pull && git stash pop && bash scripts/install_alerts_linux.sh` (stash keeps your config edits) |
| stop / start | `sudo systemctl stop quant-alerts` / `sudo systemctl start quant-alerts` |
| startup errors | `journalctl -u quant-alerts -n 50 --no-pager` |
| remove it | `bash scripts/install_alerts_linux.sh --uninstall` |

## If something goes wrong

- **`ssh` says "UNPROTECTED PRIVATE KEY FILE" or "Permission denied (publickey)":** re-run the
  `icacls` line from step 4. Check the user is `ubuntu`, which it is for Ubuntu images.
- **`ssh` hangs:** check the IP. Oracle's default network already allows SSH (port 22).
- **The install stops at "No email settings":** step 8 was skipped or the password line is empty.
- **`status` shows a feed error for BTC:** Binance is blocking this server. Do step 7's switch.
- **Oracle emails that your instance is idle and will be stopped:** step 2's upgrade has not gone
  through. Check *Billing & Cost Management*.
