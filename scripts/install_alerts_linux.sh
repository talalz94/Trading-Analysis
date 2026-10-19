#!/usr/bin/env bash
# Install (or update, or remove) the quant alerts monitor as a systemd service on a Linux server --
# the cloud counterpart of scripts/install_alerts_task.ps1. Written for Ubuntu 24.04, e.g. Google
# Cloud's free e2-micro (docs/ALERTS_CLOUD.md walks through all of it).
#
#     bash scripts/install_alerts_linux.sh              install, or update after a git pull
#     bash scripts/install_alerts_linux.sh --uninstall  stop it and remove the service
#
# Run it from anywhere as the user who should own the monitor (not root): it finds the repo from
# its own location, keeps everything inside it (.venv, alerts/state), and uses sudo only to write
# the service file. systemd then starts the monitor at boot and restarts it 30 s after any exit,
# so the scheduled-task watchdog of the Windows install is built in here.
set -euo pipefail

NAME=quant-alerts
UNIT=/etc/systemd/system/$NAME.service
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A user-level copy (systemctl --user), left from a setup done without sudo, holds the monitor's
# lock: this service then starts, finds "another monitor running" and exits every 30 s, while the
# other copy runs whatever code and config it loaded. Seen on the GCP server, 19 Oct 2026.
user_copy_off() {
    if systemctl --user is-enabled --quiet "$NAME" 2>/dev/null        || systemctl --user is-active --quiet "$NAME" 2>/dev/null; then
        echo "a user-level $NAME service is also installed; turning it off so only one monitor runs"
        systemctl --user disable --now "$NAME" 2>/dev/null || true
    fi
}

if [[ "${1:-}" == "--uninstall" ]]; then
    user_copy_off
    sudo systemctl disable --now "$NAME" 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl daemon-reload
    echo "Stopped and removed $NAME. The code, .env and alerts/state are left in $REPO."
    exit 0
fi

if [[ "$(id -u)" == "0" ]]; then
    echo "Run this as your normal user (e.g. ubuntu), not root." >&2
    exit 1
fi
cd "$REPO"

if [[ ! -f .env ]] || ! grep -q '^ALERT_SMTP_PASSWORD=.' .env; then
    echo "No email settings: create $REPO/.env with ALERT_SMTP_USER, ALERT_SMTP_PASSWORD and" >&2
    echo "ALERT_EMAIL_TO first (docs/ALERTS_CLOUD.md, step 8)." >&2
    exit 1
fi
chmod 600 .env                                   # the App Password: readable by you alone

echo "== Python environment (.venv) =="
# `python3 -m venv --help` works on Ubuntu even without python3-venv -- only CREATING fails, and a
# failed attempt leaves a .venv with no pip in it that the next run would trust. So: no pip, no venv.
if [[ ! -x .venv/bin/pip ]]; then
    rm -rf .venv
    if ! python3 -m venv .venv; then
        rm -rf .venv
        echo "Could not create a Python environment. First run:  sudo apt -y install python3-venv" >&2
        exit 1
    fi
fi
.venv/bin/python -m pip install --quiet --disable-pip-version-check --upgrade pip
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements-alerts.txt
# imports load AND the config is valid (`list` alone never reads the config)
.venv/bin/python -c "from quant.alerts.config import load_config; load_config()"
echo "packages installed; alerts/alerts_config.py loads"

user_copy_off

echo "== systemd service ($UNIT) =="
sudo tee "$UNIT" > /dev/null <<EOF
[Unit]
Description=quant alerts monitor ($REPO/alerts/alerts_config.py)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO
Environment=PYTHONUNBUFFERED=1
ExecStart=$REPO/.venv/bin/python -m quant.alerts run
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --quiet "$NAME"
sudo systemctl restart "$NAME"                   # restart, so an update loads the new code
sleep 5
if systemctl is-active --quiet "$NAME"; then
    echo "$NAME is RUNNING, and will start at every boot."
else
    echo "$NAME did not stay up -- its last lines:" >&2
    sudo journalctl -u "$NAME" -n 20 --no-pager >&2
    exit 1
fi
cat <<EOF

Check it:      .venv/bin/python -m quant.alerts status
Its log:       tail -f alerts/state/alerts.log      (startup errors: journalctl -u $NAME -n 50)
Remove it:     bash scripts/install_alerts_linux.sh --uninstall
EOF
