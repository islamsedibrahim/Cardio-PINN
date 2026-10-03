"""Install / open CardioSolv from Isaac Sim's Script Editor (Window > Script Editor > paste > Run).

Finds the extension wherever the zip was unpacked (Downloads, Documents, Isaac Sim's kit/exts, ...,
including the extra folder level that Windows "Extract All" creates), copies the newest version to
Documents/Kit/shared/exts/cardiosolv.digitaltwin (a standard search path), enables it and opens the window.
Set SOURCE to the unzipped folder if it lives somewhere else.
"""

import glob
import os
import re
import shutil
import sys

import omni.kit.app

SOURCE = ""  # optional: folder where you unzipped cardiosolv.digitaltwin-<version>.zip
EXT_ID = "cardiosolv.digitaltwin"
HOME = os.path.expanduser("~")
TARGET_ROOT = os.path.join(HOME, "Documents", "Kit", "shared", "exts")


def _version(toml):
    m = re.search(r'^version\s*=\s*"([^"]+)"', open(toml, encoding="utf-8").read(), re.M)
    return tuple(int(x) for x in re.findall(r"\d+", m.group(1))) if m else (0,)


def _candidates():
    roots = [SOURCE] if SOURCE else []
    roots += [os.path.join(HOME, "Downloads"), os.path.join(HOME, "Documents"), os.path.join(HOME, "Desktop"),
              TARGET_ROOT, os.path.dirname(omni.kit.app.get_app().get_app_filename() or "")]
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    roots += [exe_dir, os.path.dirname(exe_dir)]
    found = set()
    for r in filter(None, roots):
        for depth in ("", "*", "*/*", "*/*/*", "*/*/*/*"):
            found.update(glob.glob(os.path.join(r, depth, EXT_ID, "config", "extension.toml")))
    return sorted(found, key=_version, reverse=True)


cands = _candidates()
if not cands:
    raise FileNotFoundError("cardiosolv.digitaltwin/config/extension.toml not found: set SOURCE to the folder "
                            "where you unzipped cardiosolv.digitaltwin-<version>.zip")
src = os.path.dirname(os.path.dirname(cands[0]))
dst = os.path.join(TARGET_ROOT, EXT_ID)
mgr = omni.kit.app.get_app().get_extension_manager()
if os.path.abspath(src) != os.path.abspath(dst):
    if mgr.is_extension_enabled(EXT_ID):
        mgr.set_extension_enabled_immediate(EXT_ID, False)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"[CardioSolv] installed {src} -> {dst}")
mgr.add_path(TARGET_ROOT)
mgr.refresh_registry() if hasattr(mgr, "refresh_registry") else None
mgr.set_extension_enabled_immediate(EXT_ID, True)
if not mgr.is_extension_enabled(EXT_ID):
    raise RuntimeError("CardioSolv did not start: Window > Console, filter 'CardioSolv' for the error")
print("[CardioSolv] enabled - window docked on the right; reopen it from the CardioSolv menu. "
      "Tick AUTOLOAD in Window > Extensions to start it with Isaac Sim.")
