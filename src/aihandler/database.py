import os
from aihandler.qtvar import Var, BooleanVar, IntVar, StringVar, DoubleVar, FloatVar, ListVar
from aihandler.settings import \
    DEFAULT_MODEL, \
    DEFAULT_SCHEDULER, \
    DEFAULT_CANVAS_COLOR, \
    DEFAULT_GRID_COLOR, \
    DEFAULT_WORKING_SIZE


USER = os.environ.get("USER", "")
default_model_path = os.path.join("/", "home", USER, "stablediffusion")


class PropertyBase:
    """
    A base class used for collections of properties that are stored in the database.
    This is an interface into the database with a tkinter var representation of
    the columns within the database. The TkVarMapperBase class is used to map
    the tkinter variable to the database column and update the database when
    the tkinter variable is changed.
    """
    settings = None
    def __init__(self, app):
        self.app = app
        self.mapped = {}

    def initialize(self):
        """
        Implement this class and initialize all tkinter variables within in.
        :return:
        """
        pass

    def read(self):
        """
        Implement this class - return the items from the database.
        :return:
        """
        pass


class RunAISettings(PropertyBase):
    """
    An interface class which stores all of the application settings.
    This class should be used to interact with the settings database from
    within the application.
    """
    namespace = ""

    def __getattr__(self, name):
        """
        when a property is called such as `steps` and it is not found, an
        AttributeError is raised. This function will catch that and try to find
        the property on this classed based on namespace. If it is not found, it
        will raise the AttributeError again.
        """
        try:
            super().__getattribute__(name)
        except AttributeError as e:
            # check if the property is on this class
            # check if name_spaced already in name
            if name.startswith(self.namespace):
                raise e
            else:
                name_spaced = f"{self.namespace}_{name}"
                if hasattr(self, name_spaced):
                    return getattr(self, name_spaced)
            raise e

    def initialize(self, settings=None):
        self.settings = settings
        app = self.app

        # general settings
        self.nsfw_filter = BooleanVar(app)
        self.nsfw_filter._default = True
        self.allow_hf_downloads = BooleanVar(app)
        self.pixel_mode = BooleanVar(app)
        self.canvas_color = StringVar(app, DEFAULT_CANVAS_COLOR)
        self.paste_at_working_size = BooleanVar(app)
        self.outpaint_on_paste = BooleanVar(app)
        self.fast_load_last_session = BooleanVar(app)
        self.available_embeddings = StringVar(app, "")
        self.do_settings_reset = BooleanVar(app)
        self.dark_mode_enabled = BooleanVar(app)
        self.resize_on_paste = BooleanVar(app)

        # toolkit
        self.primary_color = StringVar(app, "#ffffff")
        self.secondary_color = StringVar(app, "#000000")
        self.blur_radius = FloatVar(app, 0.0)
        self.cyan_red = IntVar(app, 0.0)
        self.magenta_green = IntVar(app, 0.0)
        self.yellow_blue = IntVar(app, 0.0)
        self.current_tool = StringVar(app, "")

        # stable diffusion memory options
        self.use_last_channels = BooleanVar(app, True)
        self.use_enable_sequential_cpu_offload = BooleanVar(app, False)
        self.use_attention_slicing = BooleanVar(app, True)
        self.use_tf32 = BooleanVar(app, True)
        self.use_cudnn_benchmark = BooleanVar(app, True)
        self.use_enable_vae_slicing = BooleanVar(app, True)
        self.use_xformers = BooleanVar(app, True)
        self.enable_model_cpu_offload = BooleanVar(app, False)

        # size settings
        self.working_width = IntVar(app, DEFAULT_WORKING_SIZE[0])
        self.working_height = IntVar(app, DEFAULT_WORKING_SIZE[1])

        # huggingface settings
        self.hf_api_key = StringVar(app, "")

        # model settings
        self.model_base_path = StringVar(app, default_model_path)

        self.mask_brush_size = IntVar(app, 10)

        self.line_color = StringVar(app, DEFAULT_GRID_COLOR)
        self.line_width = IntVar(app, 1)
        self.line_stipple = BooleanVar(app, True)
        self.size = IntVar(app, 64)
        self.show_grid = BooleanVar(app, True)
        self.snap_to_grid = BooleanVar(app, True)

        default_scale = 750
        default_strength = 50
        default_image_guidance_scale = 150

        self.txt2img_steps = IntVar(app, 20)
        self.txt2img_ddim_eta = DoubleVar(app, 0.5)
        self.txt2img_height = IntVar(app, 512)
        self.txt2img_width = IntVar(app, 512)
        self.txt2img_scale = DoubleVar(app, default_scale)
        self.txt2img_seed = IntVar(app, 42)
        self.txt2img_random_seed = BooleanVar(app)
        self.txt2img_model_var = StringVar(app, DEFAULT_MODEL)
        self.txt2img_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.txt2img_prompt_triggers = StringVar(app, "")
        self.txt2img_n_samples = IntVar(app, 1)

        self.img2img_steps = IntVar(app, 20)
        self.img2img_ddim_eta = DoubleVar(app, 0.0)
        self.img2img_height = IntVar(app, 512)
        self.img2img_width = IntVar(app, 512)
        self.img2img_scale = DoubleVar(app, default_scale)
        self.img2img_strength = DoubleVar(app, default_strength)
        self.img2img_seed = IntVar(app, 42)
        self.img2img_random_seed = BooleanVar(app)
        self.img2img_model_var = StringVar(app, DEFAULT_MODEL)
        self.img2img_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.img2img_prompt_triggers = StringVar(app, "")
        self.img2img_n_samples = IntVar(app, 1)

        self.riffusion_steps = IntVar(app, 20)
        self.riffusion_ddim_eta = DoubleVar(app, 0.5)
        self.riffusion_height = IntVar(app, 512)
        self.riffusion_width = IntVar(app, 512)
        self.riffusion_scale = DoubleVar(app, default_scale)
        self.riffusion_seed = IntVar(app, 42)
        self.riffusion_random_seed = BooleanVar(app)
        self.riffusion_model_var = StringVar(app, DEFAULT_MODEL)
        self.riffusion_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.riffusion_prompt_triggers = StringVar(app, "")
        self.riffusion_n_samples = IntVar(app, 1)

        self.pix2pix_steps = IntVar(app, 20)
        self.pix2pix_ddim_eta = DoubleVar(app, 0.5)
        self.pix2pix_height = IntVar(app, 512)
        self.pix2pix_width = IntVar(app, 512)
        self.pix2pix_scale = DoubleVar(app, default_scale)
        self.pix2pix_image_guidance_scale = DoubleVar(app, default_image_guidance_scale)
        self.pix2pix_strength = DoubleVar(app, default_strength)
        self.pix2pix_seed = IntVar(app, 42)
        self.pix2pix_random_seed = BooleanVar(app)
        self.pix2pix_model_var = StringVar(app, DEFAULT_MODEL)
        self.pix2pix_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.pix2pix_prompt_triggers = StringVar(app, "")
        self.pix2pix_n_samples = IntVar(app, 1)

        self.inpaint_steps = IntVar(app, 20)
        self.inpaint_ddim_eta = DoubleVar(app, 0.5)
        self.inpaint_height = IntVar(app, 512)
        self.inpaint_width = IntVar(app, 512)
        self.inpaint_scale = DoubleVar(app, default_scale)
        self.inpaint_seed = IntVar(app, 42)
        self.inpaint_random_seed = BooleanVar(app)
        self.inpaint_model_var = StringVar(app, "Stable Diffusion Inpaint V2")
        self.inpaint_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.inpaint_prompt_triggers = StringVar(app, "")
        self.inpaint_n_samples = IntVar(app, 1)

        self.outpaint_steps = IntVar(app, 20)
        self.outpaint_ddim_eta = DoubleVar(app, 0.5)
        self.outpaint_height = IntVar(app, 512)
        self.outpaint_width = IntVar(app, 512)
        self.outpaint_scale = DoubleVar(app, default_scale)
        self.outpaint_seed = IntVar(app, 42)
        self.outpaint_random_seed = BooleanVar(app)
        self.outpaint_model_var = StringVar(app, "Stable Diffusion Inpaint V2")
        self.outpaint_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.outpaint_prompt_triggers = StringVar(app, "")
        self.outpaint_n_samples = IntVar(app, 1)

        self.depth2img_steps = IntVar(app, 20)
        self.depth2img_ddim_eta = DoubleVar(app, 0.5)
        self.depth2img_height = IntVar(app, 512)
        self.depth2img_width = IntVar(app, 512)
        self.depth2img_scale = DoubleVar(app, default_scale)
        self.depth2img_seed = IntVar(app, 42)
        self.depth2img_random_seed = BooleanVar(app)
        self.depth2img_model_var = StringVar(app, "Stable Diffusion Inpaint V2")
        self.depth2img_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.depth2img_prompt_triggers = StringVar(app, "")
        self.depth2img_n_samples = IntVar(app, 1)
        self.depth2img_strength = DoubleVar(app, default_strength)

        self.superresolution_steps = IntVar(app, 20)
        self.superresolution_ddim_eta = DoubleVar(app, 0.5)
        self.superresolution_height = IntVar(app, 512)
        self.superresolution_width = IntVar(app, 512)
        self.superresolution_scale = DoubleVar(app, default_scale)
        self.superresolution_seed = IntVar(app, 42)
        self.superresolution_random_seed = BooleanVar(app)
        self.superresolution_model_var = StringVar(app, "Stable Diffusion Inpaint V2")
        self.superresolution_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.superresolution_prompt_triggers = StringVar(app, "")
        self.superresolution_n_samples = IntVar(app, 1)

        self.controlnet_steps = IntVar(app, 20)
        self.controlnet_ddim_eta = DoubleVar(app, 0.5)
        self.controlnet_height = IntVar(app, 512)
        self.controlnet_width = IntVar(app, 512)
        self.controlnet_scale = DoubleVar(app, default_scale)
        self.controlnet_seed = IntVar(app, 42)
        self.controlnet_random_seed = BooleanVar(app)
        self.controlnet_model_var = StringVar(app, "Stable Diffusion Inpaint V2")
        self.controlnet_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.controlnet_prompt_triggers = StringVar(app, "")
        self.controlnet_n_samples = IntVar(app, 1)
        self.controlnet_strength = DoubleVar(app, default_strength)

        self.txt2vid_steps = IntVar(app, 20)
        self.txt2vid_ddim_eta = DoubleVar(app, 0.5)
        self.txt2vid_height = IntVar(app, 512)
        self.txt2vid_width = IntVar(app, 512)
        self.txt2vid_scale = DoubleVar(app, default_scale)
        self.txt2vid_seed = IntVar(app, 42)
        self.txt2vid_random_seed = BooleanVar(app)
        self.txt2vid_model_var = StringVar(app, DEFAULT_MODEL)
        self.txt2vid_scheduler_var = StringVar(app, DEFAULT_SCHEDULER)
        self.txt2vid_prompt_triggers = StringVar(app, "")
        self.txt2vid_n_samples = IntVar(app, 1)

        self.available_extensions = ListVar(app, [])

    def set_namespace(self, namespace):
        self.namespace = namespace

    def reset_settings_to_default(self):
        self.initialize()

