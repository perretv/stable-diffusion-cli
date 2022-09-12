import hashlib
import os

import click
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast


@click.command()
@click.option("--prompt", "-p", help="Prompt for stable-diffusion.", required=True)
@click.option(
    "--steps", "-s", default=51, help="Number of inference steps (default=%(default)s)."
)
def diffuse_cli(prompt: str, steps: int) -> None:
    """Diffuse a prompt using stable-diffusion.

    This is a click wrapped function that allows a call from the command line.

    Args:
        prompt (str): Prompt to diffuse.
        steps (int): Number of inference steps.
    """
    diffuse(prompt, steps)


def diffuse(prompt: str, steps: int) -> None:
    """Diffuse a prompt using stable-diffusion.

    Args:
        prompt (str): Prompt to diffuse.
        steps (int): Number of inference steps.
    """
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    )
    pipe = pipe.to(device)
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, num_inference_steps=steps)["sample"][0]
    token = hashlib.md5(prompt.encode()).hexdigest()  # nosec
    image.save(f"{token}.png")
