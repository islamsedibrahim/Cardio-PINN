"""Bridge to the Echocardiology (UltrasoundAI) dashboard.

* ``fetch_echo_lvef``: latest echo LVEF -> twin personalisation target
* ``post_report``: digital-twin report -> dashboard "Digital Twin" view
"""

from __future__ import annotations

import json
import urllib.request


def fetch_echo_lvef(base_url: str, timeout=5.0):
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/echo/latest", timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data.get("lvef")


def post_report(base_url: str, report_path: str, timeout=15.0):
    with open(report_path, "rb") as f:
        body = f.read()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/twin/report", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("success", False)
