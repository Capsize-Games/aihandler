import traceback
from aihandler.logger import logger
from aihandler.settings import AVAILABLE_SCHEDULERS_BY_ACTION


class SchedulerMixin:
    scheduler_name: str = "Euler a"
    schedulers: dict = {
        "DDIM": "DDIMScheduler",
        "DDIM Inverse": "DDIMInverseScheduler",
        "DDPM": "DDPMScheduler",
        "DEIS": "DEISMultistepScheduler",
        "DPM Discrete": "KDPM2DiscreteScheduler",
        "DPM Discrete a": "KDPM2AncestralDiscreteScheduler",
        "Euler a": "EulerAncestralDiscreteScheduler",
        "Euler": "EulerDiscreteScheduler",
        "Heun": "HeunDiscreteScheduler",
        "IPNM": "IPNDMScheduler",
        "LMS": "LMSDiscreteScheduler",
        "Multistep DPM": "DPMSolverMultistepScheduler",
        "PNDM": "PNDMScheduler",
        "DPM singlestep": "DPMSolverSinglestepScheduler",
        "RePaint": "RePaintScheduler",
        "Karras Variance exploding": "KarrasVeScheduler",
        "UniPC": "UniPCMultistepScheduler",
        "VE-SDE": "ScoreSdeVeScheduler",
        "VP-SDE": "ScoreSdeVpScheduler",
        "VQ Diffusion": " VQDiffusionScheduler",
    }
    registered_schedulers: dict = {}
    do_change_scheduler = False
    _scheduler = None
    current_scheduler_name = None

    @property
    def scheduler(self):
        return self.load_scheduler()

    def load_scheduler(self, force_scheduler_name=None, config=None):
        import diffusers
        if (
            not force_scheduler_name and
            self._scheduler and not self.do_change_scheduler and
            self.options.get(f"{self.action}_scheduler") == self.current_scheduler_name
        ):
            return self._scheduler

        self.current_scheduler_name = self.options.get(f"{self.action}_scheduler")

        if not self.model_path or self.model_path == "":
            traceback.print_stack()
            raise Exception("Chicken / egg problem, model path not set")

        scheduler_name = force_scheduler_name if force_scheduler_name else self.scheduler_name
        if not force_scheduler_name and scheduler_name not in AVAILABLE_SCHEDULERS_BY_ACTION[self.action]:
            scheduler_name = AVAILABLE_SCHEDULERS_BY_ACTION[self.action][0]
        scheduler_class_name = self.schedulers[scheduler_name]
        scheduler_class = getattr(diffusers, scheduler_class_name)
        kwargs = {
            "subfolder": "scheduler"
        }
        # check if self.scheduler_name contains ++
        if scheduler_name.startswith("DPM"):
            kwargs["lower_order_final"] = self.steps < 15
            if scheduler_name.find("++") != -1:
                kwargs["algorithm_type"] = "dpmsolver++"
            else:
                kwargs["algorithm_type"] = "dpmsolver"
        if self.current_model_branch:
            kwargs["variant"] = self.current_model_branch
        logger.info(f"Loading scheduler {self.scheduler_name} with kwargs {kwargs}")
        if not config and self._scheduler:
            config = self._scheduler.config
        if config:
            self._scheduler = scheduler_class.from_config(config)
        else:
            self._scheduler = scheduler_class.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
                use_auth_token=self.data["options"]["hf_token"],
                **kwargs
            )
        return self._scheduler

    def _change_scheduler(self):
        if not self.do_change_scheduler:
            return
        if self.model_path and self.model_path != "":
            self.pipe.scheduler = self.scheduler
            self.do_change_scheduler = False
        else:
            logger.warning("Unable to change scheduler, model_path is not set")

    def _prepare_scheduler(self):
        scheduler_name = self.options.get(f"{self.action}_scheduler", "euler_a")
        if self.scheduler_name != scheduler_name:
            logger.info("Prepare scheduler")
            self.set_message("Preparing scheduler...")
            self.scheduler_name = scheduler_name
            self.do_change_scheduler = True
        else:
            self.do_change_scheduler = False