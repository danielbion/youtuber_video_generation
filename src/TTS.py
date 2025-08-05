import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from pydub import AudioSegment
from typing import List, Optional
import argparse


class TextToSpeechProcessor:    
    def __init__(self, device: str = "cuda", audio_prompt_path: str = "baseline_voice.wav"):
        self.device = device
        self.audio_prompt_path = audio_prompt_path
        self.model = ChatterboxTTS.from_pretrained(device=device)
        
    def load_text_from_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def split_text_into_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    def analyze_paragraphs(self, paragraphs: List[str], max_chars: int = 400) -> None:
        for i, paragraph in enumerate(paragraphs):
            num_chars = len(paragraph)
            if num_chars > max_chars:
                print(f"Paragraph {i+1}: has {num_chars} characters")
                print(paragraph)
                print("-" * 50)
    
    def generate_audio_files(self, paragraphs: List[str], output_dir: str = "output/") -> None:
        os.makedirs(output_dir, exist_ok=True)
        n = len(paragraphs)
        
        for i, paragraph in enumerate(paragraphs):
            print(f"Processing paragraph {i+1}/{n}...")
            
            wav = self.model.generate(paragraph, audio_prompt_path=self.audio_prompt_path, exaggeration=0.5, cfg_weight=0.5)
            output_file = os.path.join(output_dir, f"paragraph_{i+1}.wav")
            ta.save(output_file, wav, self.model.sr)
            
            if i == n-1:
                audio = AudioSegment.from_wav(output_file)
                audio += AudioSegment.silent(duration=1000)
                audio.export(output_file, format="wav", bitrate="128k")
    
    def combine_audio_files(self, input_dir: str = "output/", 
                            output_file: str = "final_output.mp3",
                            silence_duration: int = 200) -> None:
            audio_files = sorted(
                [f for f in os.listdir(input_dir) if f.endswith('.wav')],
                key=lambda x: int(x.split('_')[1].split('.')[0])
            )
            
            if not audio_files:
                print("None audio file!")
                return
            
            print(f"Combining {len(audio_files)} audio files...")
            
            silence = AudioSegment.silent(duration=silence_duration)
            combined = AudioSegment.empty()
            
            for idx, audio_file in enumerate(audio_files):
                audio_path = os.path.join(input_dir, audio_file)
                audio = AudioSegment.from_wav(audio_path)
                combined += audio
                
                if idx < len(audio_files) - 1:
                    combined += silence
            
            file_format = output_file.split('.')[-1]
            combined.export(output_file, format=file_format, bitrate="128k")

def main():
    parser = argparse.ArgumentParser(description="ChatterboxTTS text to speech")
    parser.add_argument("--input", "-i", default="assets/text.txt", help="Input file")
    parser.add_argument("--output-dir", "-o", default="output/audios", help="Output dir")
    parser.add_argument("--final-output", "-f", default="output/audios/final_output.mp3", help="Output file")
    parser.add_argument("--device", "-d", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--audio-prompt", "-a", default="assets/baseline_voice.wav", help="Baseline voice")
    parser.add_argument("--silence", "-s", type=int, default=200, help="Silence duration (ms)")
    parser.add_argument("--analyze-only", default=False, action="store_true", help="Paragraph size analyzer")
    
    args = parser.parse_args()
    
    processor = TextToSpeechProcessor(device=args.device, audio_prompt_path=args.audio_prompt)
    
    text = processor.load_text_from_file(args.input)
    paragraphs = processor.split_text_into_paragraphs(text)
    
    processor.analyze_paragraphs(paragraphs)
    
    if args.analyze_only:        
        return
    
    processor.generate_audio_files(paragraphs, args.output_dir)
    
    processor.combine_audio_files(args.output_dir, args.final_output, args.silence)

if __name__ == "__main__":
    main()