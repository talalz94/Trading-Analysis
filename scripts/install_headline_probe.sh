#!/usr/bin/env bash
# Install (or remove) the HEADLINE PROBE -- a 7-day test of free news sources -- as its own
# systemd service on the alerts server (experiments/headline_probe.py says what it records).
#
#     bash scripts/install_headline_probe.sh               check every source once, then start it
#     bash scripts/install_headline_probe.sh --uninstall   stop it and remove the service
#
# It runs beside quant-alerts and touches none of its files: its own database under
# experiments/results/headline_probe/, the same .venv (nothing new to install), and systemd caps
# of 250 MB and 10% of a CPU so it can never crowd the monitor on a 1 GB e2-micro. It stops by
# itself 7 days after its FIRST start; a reboot or restart inside that resumes, never extends.
set -euo pipefail

NAME=headline-probe
UNIT=/etc/systemd/system/$NAME.service
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--uninstall" ]]; then
    sudo systemctl disable --now "$NAME" 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl daemon-reload
    echo "Stopped and removed $NAME. What it recorded stays in experiments/results/headline_probe/."
    exit 0
fi

if [[ "$(id -u)" == "0" ]]; then
    echo "Run this as your normal user, not root." >&2
    exit 1
fi
cd "$REPO"
if [[ ! -x .venv/bin/python ]]; then
    echo "No .venv here: install the alerts first (bash scripts/install_alerts_linux.sh)." >&2
    exit 1
fi

echo "== every source, once =="
PYTHONPATH="$REPO" .venv/bin/python experiments/headline_probe.py --once

echo "== systemd service ($UNIT) =="
sudo tee "$UNIT" > /dev/null <<EOF
[Unit]
Description=quant headline probe (a 7-day test of free news sources)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO
Environment=PYTHONUNBUFFERED=1
ExecStart=$REPO/.venv/bin/python experiments/headline_probe.py --days 7
Restart=on-failure
RestartSec=60
MemoryMax=250M
CPUQuota=10%
Nice=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --quiet "$NAME"
sudo systemctl restart "$NAME"
sleep 5
if systemctl is-active --quiet "$NAME"; then
    echo "$NAME is RUNNING: 7 days from its first start, then it stops by itself."
else
    echo "$NAME did not stay up -- its last lines:" >&2
    sudo journalctl -u "$NAME" -n 20 --no-pager >&2
    exit 1
fi
cat <<EOF

Check it:      .venv/bin/python experiments/headline_probe.py --status
Its log:       journalctl -u $NAME -n 30 --no-pager
The result:    PYTHONPATH=. .venv/bin/python experiments/headline_report.py
Remove it:     bash scripts/install_headline_probe.sh --uninstall
EOF
