"""Open the CardioSolv panel from Isaac Sim's Script Editor (Window > Script Editor), no setup needed.

1. Unzip cardiosolv.digitaltwin-<version>.zip anywhere, e.g. C:/omni/exts or ~/omni/exts.
2. Set EXT_FOLDER below to the folder that CONTAINS the "cardiosolv.digitaltwin" folder.
3. Paste this file into the Script Editor and press Run (Ctrl+Enter).

It adds the folder to Kit's extension search paths, enables the extension and shows the window.
Next time, the Extension Manager lists "CardioSolv Digital Twin" (tick AUTOLOAD to keep it on).
"""

import os

import omni.kit.app

EXT_FOLDER = r"C:/omni/exts"  # <-- the folder containing cardiosolv.digitaltwin/
EXT_ID = "cardiosolv.digitaltwin"

folder = os.path.abspath(os.path.expanduser(EXT_FOLDER))
if os.path.isdir(os.path.join(folder, EXT_ID, EXT_ID)):  # zip unpacked one level deeper
    folder = os.path.join(folder, EXT_ID)
if not os.path.isfile(os.path.join(folder, EXT_ID, "config", "extension.toml")):
    raise FileNotFoundError(f"{folder}/{EXT_ID}/config/extension.toml not found: set EXT_FOLDER to the folder "
                            f"that contains the '{EXT_ID}' folder")
mgr = omni.kit.app.get_app().get_extension_manager()
mgr.add_path(folder)
mgr.set_extension_enabled_immediate(EXT_ID, True)
if not mgr.is_extension_enabled(EXT_ID):
    raise RuntimeError("CardioSolv did not start: open Window > Console and look for [CardioSolv] errors")
import cardiosolv.digitaltwin.extension  # noqa: E402,F401  (now importable)

print("[CardioSolv] enabled from", folder, "- the window is docked on the right (Window > CardioSolv Digital Twin)")
