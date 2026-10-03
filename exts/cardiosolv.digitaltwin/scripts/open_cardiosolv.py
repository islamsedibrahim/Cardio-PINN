"""Install / open CardioSolv from Isaac Sim's Script Editor (Window > Script Editor > paste > Run).

1. Finds every unzipped copy of the extension (Downloads, Documents, Desktop, Isaac Sim's folders; also
   the extra folder level that Windows "Extract All" creates) and installs the NEWEST one as
   Documents/Kit/shared/exts/cardiosolv.digitaltwin (a default Isaac Sim search folder).
2. Unregisters the other copies (two copies of one extension make Kit fail with KeyError in
   fast_importer.remove_sys_path) and lists them so you can delete them (or set CLEAN = True).
3. Turns on AUTOLOAD, then enables the extension now if no CardioSolv copy is running yet; otherwise
   asks you to restart Isaac Sim, so an already-loaded copy is never swapped while running.
"""

import glob
import os
import re
import shutil
import sys

import carb.settings
import omni.kit.app

SOURCE = ""  # optional: the folder where you unzipped cardiosolv.digitaltwin-<version>.zip
CLEAN = False  # True: delete the other (duplicate) copies after installing
EXT_ID = "cardiosolv.digitaltwin"
HOME = os.path.expanduser("~")
TARGET_ROOT = os.path.join(HOME, "Documents", "Kit", "shared", "exts")
TARGET = os.path.join(TARGET_ROOT, EXT_ID)


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def version_of(ext_dir):
    toml = os.path.join(ext_dir, "config", "extension.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', open(toml, encoding="utf-8").read(), re.M)
    return tuple(int(x) for x in re.findall(r"\d+", m.group(1))) if m else (0,)


def find_copies():
    roots = [SOURCE] if SOURCE else []
    roots += [os.path.join(HOME, d) for d in ("Downloads", "Documents", "Desktop")]
    exe = os.path.dirname(os.path.abspath(sys.executable))
    roots += [exe, os.path.dirname(exe), os.path.dirname(os.path.dirname(exe))]
    found = {}
    for r in filter(os.path.isdir, roots):
        for depth in ("", "*", "*/*", "*/*/*", "*/*/*/*"):
            for toml in glob.glob(os.path.join(r, depth, EXT_ID, "config", "extension.toml")):
                d = os.path.dirname(os.path.dirname(toml))
                found[norm(d)] = d
    return sorted(found.values(), key=version_of, reverse=True)


def running_path(mgr):
    try:
        for e in mgr.get_extensions():
            if e.get("name") == EXT_ID and e.get("enabled"):
                return e.get("path")
    except Exception:
        pass
    return None


def main():
    copies = find_copies()
    if not copies:
        raise FileNotFoundError(f"No {EXT_ID}/config/extension.toml found: set SOURCE to the folder where you unzipped "
                                f"the CardioSolv zip")
    newest = copies[0]
    print(f"[CardioSolv] newest copy: {newest}  (version {'.'.join(map(str, version_of(newest)))})")

    mgr = omni.kit.app.get_app().get_extension_manager()
    running = mgr.is_extension_enabled(EXT_ID) or any(m.startswith("cardiosolv.") for m in sys.modules)
    run_path = running_path(mgr)

    # 1. install the newest copy at the standard location (unless it already is that copy)
    if norm(newest) != norm(TARGET):
        if run_path and norm(run_path) == norm(TARGET):
            print("[CardioSolv] CardioSolv is running from the install folder: close Isaac Sim, run this script "
                  "again right after starting it.")
            return
        os.makedirs(TARGET_ROOT, exist_ok=True)
        shutil.rmtree(TARGET, ignore_errors=True)
        shutil.copytree(newest, TARGET, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"[CardioSolv] installed -> {TARGET}")

    # 2. unregister / list the other copies
    others = [c for c in copies if norm(c) != norm(TARGET)]
    for c in others:
        for path in (os.path.dirname(c), c):
            try:
                mgr.remove_path(path)
            except Exception:
                pass
    if others:
        print("[CardioSolv] other copies (delete these folders to avoid duplicate-extension errors):")
        for c in others:
            holder = os.path.dirname(c)
            print("    ", holder if os.path.basename(holder).startswith(EXT_ID) else c)
            if CLEAN:
                shutil.rmtree(holder if os.path.basename(holder).startswith(EXT_ID) else c, ignore_errors=True)
    mgr.add_path(TARGET_ROOT)

    # 3. autoload at startup (same setting the Extension Manager's AUTOLOAD toggle writes)
    try:
        s = carb.settings.get_settings()
        key = "/persistent/app/exts/enabled"
        enabled = list(s.get(key) or [])
        if EXT_ID not in enabled:
            s.set(key, enabled + [EXT_ID])
    except Exception as exc:
        print(f"[CardioSolv] could not set AUTOLOAD ({exc}); tick it in Window > Extensions")

    if running:
        print("[CardioSolv] a CardioSolv copy is already loaded in this session: RESTART Isaac Sim. It will start "
              f"from {TARGET} and open its window (CardioSolv menu).")
    else:
        mgr.set_extension_enabled_immediate(EXT_ID, True)
        if mgr.is_extension_enabled(EXT_ID):
            print("[CardioSolv] enabled - window docked on the right; reopen it from the CardioSolv menu.")
        else:
            print("[CardioSolv] not enabled: Window > Console, filter 'CardioSolv', and send the error.")



main()
