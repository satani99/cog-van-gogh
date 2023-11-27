# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File
import tempfile
import os
from typing import Any, List
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            "./Van-Gogh-diffusion", variant="fp16", torch_dtype=torch.float16, local_files_only=True).to("cuda")
        self.pipeline.enable_model_cpu_offload()

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        output_name: str = Input(description="Name for the output image", default="van_gogh_1"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        seed = int.from_bytes(os.urandom(2), "big")

        init_image = load_image(f"{image}")
        generator = torch.Generator("cuda").manual_seed(seed)

        prompt = "lvngvncnt, highly detailed, portrait"
        neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

        output = self.pipeline(prompt, image=init_image, strength=0.3, guidance_scale=6, generator=generator).images[0]
        output_paths = []

        output_path = f"/tmp/{output_name}.png"
        output.save(output_path)
        output_paths.append(Path(output_path))

        torch.cuda.empty_cache()

        return output_paths
        
