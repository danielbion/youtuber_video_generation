import os
import json
import time
import re
import gc
from pathlib import Path
from typing import List, Dict, Optional
import argparse

import torch
import requests
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLPipeline
from dotenv import load_dotenv


class ImageGeneratorConfig:
    def __init__(self):        
        self.SYSTEM_PROMPT = """
        You will take a given a paragraph and generate a prompt for an image generation model. 
        The prompt will be a list of keywords that describe the image you want to generate to capture the audience attention, including some of the following elements:
        Action: holding an umbrella, playing soccer, eating spaghetti, reading a book, etc.
        Landscape: in the park, at a restaurant, sitting next to a wooden window, etc.
        Subject (One or more): animals, food, cars, people, objects, etc. Be creative!
        Description of the subject: wearing a red shirt, blue hat, black hair, etc.
        Lighting: natural light, crepuscular rays, etc.        
        Mood: happy, sad, mysterious, etc. 
        Tips:
        Focus more on describing the actions than the subjects
        If you need to describe only a landscape, avoid describing the subject, else be detailed and specific.
        Avoid writting text or words on the image.
        Use multiple brackets () to increase its strength and [] to reduce.
        The response you give will always only be all the keywords you have chosen separated by a comma only. 
        IMPORTANT: The total prompt token limit is 60 tokens. 
        I will give actual paragraph i'm writing so you can use the context to generate a more accurate prompt:
        """
        
        self.STYLE_PREFIX = "(((Digital Painting, concept art, expansive))), "

        self.NEGATIVE_PROMPT = ("text, letters, too many feet, too many fingers, deformed, extra limbs, "
                               "twisted fingers, malformed hands, multiple heads, missing limb, cut-off, "
                               "over satured, grain, bad anatomy, poorly drawn face, mutation, mutated, "
                               "floating limbs, disconnected limbs, extra fingers, missing arms, mutated hands, missing legs")
        
        # Groq API
        self.GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
        self.MODEL = "deepseek-r1-distill-llama-70b"
        self.MAX_TOKENS_PROMPT = 700
        self.TEMPERATURE = 0.5
        self.DELAY_BETWEEN_CALLS = 10 # Respecting Groq rate limits

        self.MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"    
        self.IMAGE_HEIGHT = 512
        self.IMAGE_WIDTH = 896
        self.INFERENCE_STEPS = 40
        self.GUIDANCE_SCALE = 7
        self.SEED = 0
        
        # Paths
        self.TEXT_PATH = Path("assets/text.txt")
        self.OUTPUT_DIR = Path("output/images")
        self.PROMPTS_FILE = "prompts.jsonl"


class PromptGenerator:
    def __init__(self, config: ImageGeneratorConfig):
        self.config = config
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise EnvironmentError("Can't find GROQ_API_KEY in env file")
    
    def clean_response(self, response: str) -> str:        
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        return cleaned.strip().strip('"').replace("\n", " ").replace("  ", " ")
    
    def generate_prompt(self, paragraph: str, last_paragraph: str, next_paragraph: str) -> str:        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        text = f"This is the actual paragraph: \n{paragraph}"
        body = {
            "model": self.config.MODEL,
            "messages": [
                {"role": "system", "content": self.config.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "max_tokens": self.config.MAX_TOKENS_PROMPT,
            "temperature": self.config.TEMPERATURE,
        }
        
        try:
            response = requests.post(
                self.config.GROQ_API_URL, 
                headers=headers, 
                json=body, 
                timeout=60
            )
            response.raise_for_status()
            
            raw_content = response.json()["choices"][0]["message"]["content"]
            return self.clean_response(raw_content)
            
        except requests.RequestException as e:
            print(f"Error in Groq API: {e}")
            raise
        
class ImageGenerator:
    def __init__(self, config: ImageGeneratorConfig):
        self.config = config
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):        
        print(f"Loading model from: {self.device}")
        
        pipeline_class = StableDiffusionXLPipeline        
        self.pipe = pipeline_class.from_pretrained(
            self.config.MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            text_encoder_3=None,
            tokenizer_3=None,
        )
        
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
                    
    def generate_image(self, paragraph_id: int, prompt: str, output_dir: Path) -> Path:        
        full_prompt = self.config.STYLE_PREFIX + prompt
        
        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=self.config.NEGATIVE_PROMPT,
            num_inference_steps=self.config.INFERENCE_STEPS,
            guidance_scale=self.config.GUIDANCE_SCALE,
            generator=torch.manual_seed(self.config.SEED),
            max_sequence_length=512,
            height=self.config.IMAGE_HEIGHT,
            width=self.config.IMAGE_WIDTH
        ).images[0]
                
        output_path = output_dir / f"paragraph_{paragraph_id}.png"
        result.save(output_path)
        
        del result
        gc.collect()
        
        return output_path

class TextProcessor:
    @staticmethod
    def read_paragraphs(file_path: Path) -> List[str]:        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        text = file_path.read_text(encoding="utf-8")
        return [p.strip() for p in text.split("\n\n") if p.strip()]


class DataManager:  
    @staticmethod
    def save_prompt_data(file_path: Path, paragraph_id: int, prompt: str, paragraph_text: str):        
        record = {
            "paragraph": paragraph_id,
            "prompt": prompt,
            "paragraph_text": paragraph_text
        }
        
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    @staticmethod
    def load_prompts(file_path: Path) -> List[Dict]:        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {file_path}")
        
        prompts = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                prompts.append(json.loads(line.strip()))
        
        return prompts


class ImageGenerationPipeline:
    def __init__(self, config: ImageGeneratorConfig):
        self.config = config
        self.prompt_generator = PromptGenerator(config)
        self.image_generator = ImageGenerator(config)
        self.text_processor = TextProcessor()
        self.data_manager = DataManager()
    
    def run(self, 
            skip_prompt_generation: bool = False,
            skip_image_generation: bool = False):

        text_file =  self.config.TEXT_PATH
        output_dir = self.config.OUTPUT_DIR
        
        output_dir.mkdir(exist_ok=True)
        prompts_file = output_dir / self.config.PROMPTS_FILE
        
        if not skip_prompt_generation and prompts_file.exists():
            prompts_file.unlink()
        
        if not skip_image_generation:
            self.image_generator.load_model()
        
        paragraphs = self.text_processor.read_paragraphs(text_file)
        
        print(f"Processing {len(paragraphs)} paragraphs...")
        
        existing_prompts = []
        if skip_prompt_generation:
            existing_prompts = self.data_manager.load_prompts(prompts_file)
        
        for i, paragraph in enumerate(paragraphs, 1):
            print(f"Processing paragraph {i}/{len(paragraphs)}...")
            
            if skip_prompt_generation:
                if i <= len(existing_prompts):
                    prompt = existing_prompts[i-1]["prompt"]
                else:
                    print(f"Prompt for paragraph {i} not found!")
                    continue
            else:
                last_paragraph = paragraphs[i-1] if i > 1 else ""
                next_paragraph = paragraphs[i+1] if i < len(paragraphs)-1 else ""
                prompt = self.prompt_generator.generate_prompt(paragraph, last_paragraph, next_paragraph)
                self.data_manager.save_prompt_data(prompts_file, i, prompt, paragraph)
                print(f"Prompt saved: {prompt}")
                time.sleep(self.config.DELAY_BETWEEN_CALLS)
            
            # Gera imagem
            if not skip_image_generation:
                image_path = self.image_generator.generate_image(i, prompt, output_dir)
                print(f"Image saved: {image_path}")
        
        print(f"Done! Images saved in: {output_dir}")


def main():    
    parser = argparse.ArgumentParser(description="Generate images from text")
    parser.add_argument("--skip-prompts", default=False, action="store_true", help="Skip prompt generation (Reuse prompt file)")
    parser.add_argument("--skip-images", default=False, action="store_true", help="Skip image generation")
    args = parser.parse_args()
    
    load_dotenv()
    
    config = ImageGeneratorConfig()
    
    # Executa pipeline
    pipeline = ImageGenerationPipeline(config)
    pipeline.run(
        skip_prompt_generation=args.skip_prompts,
        skip_image_generation=args.skip_images        
    )


if __name__ == "__main__":
    main()
