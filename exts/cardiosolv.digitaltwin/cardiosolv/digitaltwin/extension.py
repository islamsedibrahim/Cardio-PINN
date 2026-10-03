import traceback

import carb
import omni.ext

MENU_PATH = "Window/CardioSolv Digital Twin"


class CardioSolvExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self._panel = None
        self._menus = []
        try:
            from .ui.panel import CardioSolvPanel

            self._panel = CardioSolvPanel(ext_id)
            self._panel.show()
        except Exception:
            # make a broken install visible instead of failing silently
            carb.log_error("[CardioSolv] the panel could not be created:\n" + traceback.format_exc())
        try:
            from omni.kit.menu.utils import MenuItemDescription, add_menu_items

            item = MenuItemDescription(name="CardioSolv Digital Twin", onclick_fn=self.show)
            for menu in ("Window", "CardioSolv"):
                add_menu_items([item], menu)
                self._menus.append(([item], menu))
        except Exception:
            carb.log_warn("[CardioSolv] menu entries unavailable:\n" + traceback.format_exc())
        print("[CardioSolv] Extension started: Window > CardioSolv Digital Twin")

    def show(self, *_):
        if self._panel is None:
            from .ui.panel import CardioSolvPanel

            self._panel = CardioSolvPanel(self._ext_id)
        self._panel.show()

    def on_shutdown(self):
        try:
            from omni.kit.menu.utils import remove_menu_items

            for items, menu in self._menus:
                remove_menu_items(items, menu)
        except Exception:
            pass
        self._menus = []
        if self._panel:
            self._panel.destroy()
            self._panel = None
        print("[CardioSolv] Extension stopped")
