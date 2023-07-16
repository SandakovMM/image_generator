#!/usr/bin/env python3
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from optparse import OptionParser, OptionValueError


def generate(prompt, height=512, width=512, output="result.png"):
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_attention_slicing()
    pipeline.to("cpu")

    results = pipeline([prompt], height=height, width=width,
                       num_inference_steps=20, num_images_per_prompt=1)
    results.images[0].save(output)


HELP_MESSAGE = """usage: %prog [options] prompt

This script generates images from a prompt using the Stable Diffusion model.
Default image size is 512x512. The image size must be divisible by 16.
"""


def main():
    opts = OptionParser(usage=HELP_MESSAGE)
    opts.add_option("--height", dest="height", type="int", default=512, help="height of the generated image")
    opts.add_option("--width", dest="width", type="int", default=512, help="width of the generated image")
    opts.add_option("-o", "--output", dest="output", type="string", default="result.png", help="output file name")

    (options, args) = opts.parse_args()

    if len(args) < 1:
        opts.error("prompt not specified")

    prompt = args[0]
    height = options.height
    width = options.width

    if height % 16 != 0 or width % 16 != 0:
        raise OptionValueError("height and width must be divisible by 16")

    generate(prompt, height, width, output=options.output)


if __name__ == "__main__":
    main()
