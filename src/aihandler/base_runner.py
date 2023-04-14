import os
from PyQt6.QtCore import QObject, pyqtSignal
from aihandler.logger import logger
from aihandler.qtvar import TQDMVar, ImageVar
from aihandler.settings_manager import SettingsManager
from aihandler.settings import AIRUNNER_ENVIRONMENT, LOG_LEVEL
from aihandler.util import get_extensions_from_url, import_extension_class


class BaseRunner(QObject):
    _tqdm_var: TQDMVar
    tqdm_callback_signal = pyqtSignal()
    active_extensions = []

    @property
    def is_dev_env(self):
        return AIRUNNER_ENVIRONMENT == "dev"

    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.set_level(LOG_LEVEL)
        self.app = kwargs.get("app", None)
        self.settings_manager = SettingsManager(app=self.app)
        self._tqdm_var: TQDMVar = kwargs.get("tqdm_var", None)
        self._tqdm_callback = kwargs.get("tqdm_callback", None)
        self._image_var: ImageVar = kwargs.get("image_var", None)
        self._image_handler = kwargs.get("image_handler", None)
        self._error_handler = kwargs.get("error_handler", None)
        self._error_var = kwargs.get("error_var", None)
        self._message_var = kwargs.get("message_var", None)
        self._message_handler = kwargs.get("message_handler", None)
        self.tqdm_callback_signal = pyqtSignal(int, int, str, object, object)
        self.get_extensions_from_url()
        #self.initialize_active_extensions()

    def get_extensions_from_url(self):
        available_extensions = get_extensions_from_url(self)
        self.settings_manager.settings.available_extensions.set(available_extensions)

    def initialize_active_extensions(self):
        """
        Initialize extensions by loading them from the extensions_directory.
        These are extensions that have been activated by the user.
        Extensions can be activated by manually adding them to the extensions folder
        or by browsing for them in the extensions menu and activating them there.

        This method initializes active extensions.
        :return:
        """
        extensions = []
        available_extensions = self.settings_manager.settings.available_extensions.get()
        if available_extensions:
            for extension in available_extensions:
                if extension.enabled:
                    repo = extension.repo.get()
                    name = repo.split("/")[-1]
                    base_path = self.settings_manager.settings.model_base_path.get()
                    path = os.path.join(base_path, "extensions", name)
                    ExtensionClass = import_extension_class(repo, path, "stablediffusion.py", "RunnerExtension")
                    if ExtensionClass:
                        extensions.append(ExtensionClass(self.settings_manager))

        self.active_extensions = extensions
        print("*" * 100)
        print("EXTENSIONS LOADED: ", extensions)
        print("*" * 100)

    def set_message(self, message):
        if self._message_handler:
            self._message_handler(message)
        elif self._message_var:
            self._message_var.set(message)

    def error_handler(self, error):
        if self._error_handler:
            self._error_handler(error)
        elif self._error_var:
            self._error_var.set(str(error))

    def tqdm_callback(self, step, total, action, image=None, data=None):
        if self._tqdm_callback:
            self._tqdm_callback(step, total, action, image, data)
        else:
            self._tqdm_var.set({
                "step": step,
                "total": total,
                "action": action,
                "image": image,
                "data": data
            })