import omni.ext

from .ui.panel import CardioSolvPanel

MENU_PATH = "Window/CardioSolv Digital Twin"


class CardioSolvExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self._panel = CardioSolvPanel(ext_id)
        self._panel.show()
        self._menu = None
        try:
            from omni.kit.menu.utils import MenuItemDescription, add_menu_items

            self._menu = [MenuItemDescription(name="CardioSolv Digital Twin", onclick_fn=self._panel.show)]
            add_menu_items(self._menu, "Window")
        except Exception:
            pass
        print("[CardioSolv] Extension started")

    def on_shutdown(self):
        if self._menu:
            try:
                from omni.kit.menu.utils import remove_menu_items

                remove_menu_items(self._menu, "Window")
            except Exception:
                pass
        if self._panel:
            self._panel.destroy()
            self._panel = None
        print("[CardioSolv] Extension stopped")
