import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# load pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# generate
prompt = "Spaceship Battle"
video_frames = pipe(prompt, num_inference_steps=100, num_frames=40).frames

# convent to video
video_path = export_to_video(video_frames, output_video_path="C:\\Users\\Plutonium\\Desktop\\vid.mp4")