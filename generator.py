#!/usr/bin/env python3
import torch
import sys
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

if len(sys.argv) < 2:
    print("Please provide a prompt.")
    sys.exit(1)

pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)  
pipeline.enable_attention_slicing()
pipeline.to("cpu")


prompt = sys.argv[1]
results = pipeline([prompt], num_inference_steps=20, height=1024, width=1024)
image = results.images[0]
image.save("my_image.png")