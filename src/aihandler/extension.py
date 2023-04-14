class BaseStableDiffusionExtension:
    def __init__(self, settings_manager):
        self.settings_manager = settings_manager

    def call_pipe(self, runner, **kwargs):
        """
        Override this method to inject code into the Stable Diffusion runner just before the pipe is called.
        :param runner:
        :param kwargs:
        :return:
        """
        pass