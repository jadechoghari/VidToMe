from diffusers import DiffusionPipeline
from invert import Inverter
from generate import Generator
from utils import init_model, seed_everything, get_frame_ids

class VidToMePipeline(DiffusionPipeline):
    def __init__(self, device="cuda", sd_version="2.1", float_precision="fp16", height=512, width=512):
        # this will initlize the core pipeline components
        pipe, scheduler, model_key = init_model(device, sd_version, None, "none", float_precision)
        self.pipe = pipe
        self.scheduler = scheduler
        self.model_key = model_key
        self.device = device
        self.sd_version = sd_version
        self.float_precision = float_precision
        self.height = height
        self.width = width

    def __call__(self, video_path=None, video_prompt=None, edit_prompt=None, 
                 control_type="none", n_timesteps=50, guidance_scale=7.5, 
                 negative_prompt="ugly, blurry, low res", frame_range=None, 
                 use_lora=False, seed=123, local_merge_ratio=0.9, global_merge_ratio=0.8):
        
        # dynamic config built from user inputs
        config = self._build_config(video_path, video_prompt, edit_prompt, control_type, 
                                    n_timesteps, guidance_scale, negative_prompt, 
                                    frame_range, use_lora, seed, local_merge_ratio, global_merge_ratio)
        
        # seed for reproducibility - change as you need
        seed_everything(config['seed'])

        # inversion stage
        print("Start inversion!")
        inversion = Inverter(self.pipe, self.scheduler, config)
        inversion(config['input_path'], config['inversion']['save_path'])

        # generation stage
        print("Start generation!")
        generator = Generator(self.pipe, self.scheduler, config)
        frame_ids = get_frame_ids(config['generation']['frame_range'], None)
        generator(config['input_path'], config['generation']['latents_path'], 
                  config['generation']['output_path'], frame_ids=frame_ids)
        print(f"Output generated at: {config['generation']['output_path']}")

    def _build_config(self, video_path, video_prompt, edit_prompt, control_type, 
                      n_timesteps, guidance_scale, negative_prompt, frame_range, 
                      use_lora, seed, local_merge_ratio, global_merge_ratio):
        # constructing config dictionary from user prompts
        config = {
            'sd_version': self.sd_version,
            'input_path': video_path,
            'work_dir': "outputs/",
            'height': self.height,
            'width': self.width,
            'inversion': {
                'prompt': video_prompt or "Default video prompt.",
                'save_path': "outputs/latents",
                'steps': 50,
                'save_intermediate': False
            },
            'generation': {
                'control': control_type,
                'guidance_scale': guidance_scale,
                'n_timesteps': n_timesteps,
                'negative_prompt': negative_prompt,
                'prompt': edit_prompt or "Default edit prompt.",
                'latents_path': "outputs/latents",
                'output_path': "outputs/final",
                'frame_range': frame_range or [0, 32],
                'use_lora': use_lora,
                'local_merge_ratio': local_merge_ratio,
                'global_merge_ratio': global_merge_ratio
            },
            'seed': seed,
            'device': self.device,
            'float_precision': self.float_precision
        }
        return config

# Sample usage
pipeline = VidToMePipeline(device="cuda", sd_version="2.1", float_precision="fp16")
pipeline(video_path="path/to/video.mp4", video_prompt="A beautiful scene of a sunset", 
         edit_prompt="Make the sunset look more vibrant", control_type="depth", n_timesteps=50)
