import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

#%%
prompt = "Cats djing."
video_frames = pipe(prompt, num_inference_steps=20, height=480, width=720, num_frames=80, guidance_scale=16).frames
video_path = export_to_video(video_frames, output_video_path="C:\\Users\\Plutonium\\Desktop\\vid2.mp4")

# height=320, width=576,