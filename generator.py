#!/usr/bin/env python3
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

# Todo. Yes I know that non positional arguments are better.
height = width = 512
if len(sys.argv) >= 3:
    height = int(sys.argv[2])
if len(sys.argv) >= 4:
    width = int(sys.argv[3])

if height % 16 != 0 or width % 16 != 0:
    print("Height and width must be divisible by 16.")
    sys.exit(1)

results = pipeline([prompt], height=height, width=width, num_inference_steps=20, num_images_per_prompt=4)
for idx, image in enumerate(results.images):
    image.save("my_image" + str(idx) + ".png")