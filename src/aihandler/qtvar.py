from PyQt6.QtCore import QObject, pyqtSignal


class Var(QObject):
    my_signal = pyqtSignal()

    def __init__(self, app=None, default=None):
        super().__init__()
        self._app = app
        self.set(default, skip_save=True)

    def set(self, val, skip_save=False):
        self._my_variable = val
        self.emit()
        if self._app and not skip_save:
            try:
                self._app.save_settings()
            except Exception as e:
                print(e)

    def get(self):
        return self._my_variable

    def emit(self):
        if self._my_variable is None:
            return
        self.my_signal.emit(self._my_variable)


class TQDMVar(Var):
    step = pyqtSignal(int)
    total = pyqtSignal(int)
    action = pyqtSignal(str)
    image = pyqtSignal(object)
    data = pyqtSignal(object)
    my_signal = pyqtSignal(int, int, str, object, object)

    def set(self, val, skip_save=False):
        if val is None:
            return
        self.step = val["step"]
        self.total = val["total"]
        self.action = val["action"]
        self.image = val["image"]
        self.data = val["data"]
        self.emit()

    def emit(self):
        self.my_signal.emit(self.step, self.total, self.action, self.image, self.data)


class MessageHandlerVar(Var):
    message = pyqtSignal(dict)
    my_signal = pyqtSignal(dict)

    def set(self, val, skip_save=False):
        if val is None:
            return
        self.message = val
        self.emit()

    def emit(self):
        self.my_signal.emit({
            "response": self.message
        })


class ErrorHandlerVar(Var):
    message = pyqtSignal(str)
    my_signal = pyqtSignal(str)

    def set(self, val, skip_save=False):
        if val is None:
            return
        self.message = val
        self.emit()

    def emit(self):
        self.my_signal.emit(self.message)

class ImageVar(Var):
    image = pyqtSignal(object)
    data = pyqtSignal(object)
    nsfw_content_detected = pyqtSignal(bool)
    my_signal = pyqtSignal(object, object, bool)

    def set(self, val, skip_save=False):
        if val is None:
            return
        self.image = val["image"]
        self.data = val["data"]
        self.nsfw_content_detected = val["nsfw_content_detected"]
        self.emit()

    def emit(self):
        self.my_signal.emit(self.image, self.data, self.nsfw_content_detected)


class BooleanVar(Var):
    my_signal = pyqtSignal(bool)


class IntVar(Var):
    my_signal = pyqtSignal(int)


class FloatVar(Var):
    my_signal = pyqtSignal(float)


class StringVar(Var):
    my_signal = pyqtSignal(str)


class ListVar(Var):
    my_signal = pyqtSignal(list)


class DictVar(Var):
    my_signal = pyqtSignal(dict)


class DoubleVar(Var):
    my_signal = pyqtSignal(float)


class ExtensionVar(Var):
    name = StringVar("")
    description = StringVar("")
    repo = StringVar("")
    version = StringVar("")
    reviewed = BooleanVar(False)
    official = BooleanVar(False)
    enabled = BooleanVar(False)
    my_signal = pyqtSignal(str, bool)
    object = None

    def __init__(
        self,
        app=None,
        name="",
        description="",
        repo="",
        version="",
        reviewed=False,
        official=False,
        enabled=False
    ):
        super().__init__(app, None)
        self.name.set(name, skip_save=True)
        self.description.set(description, skip_save=True)
        self.repo.set(repo, skip_save=True)
        self.version.set(version, skip_save=True)
        self.reviewed.set(reviewed, skip_save=True)
        self.official.set(official, skip_save=True)
        self.enabled.set(enabled, skip_save=True)
        self.name.my_signal.connect(self.emit)

    def emit(self):
        self.my_signal.emit(self.name.get(), self.enabled.get() == True)
