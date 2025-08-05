# YouTube Video Generator

An automated pipeline for generating YouTube videos from text using AI-powered text-to-speech, image generation, and video composition.

## Features

- **Text-to-Speech (TTS)**: Generate natural-sounding voiceovers using a custom voice baseline
- **Text-to-Image (TTI)**: Create relevant images for each text paragraph
- **Image-to-Video (ITV)**: Compose videos with zoom effects and synchronized audio
- **Automated Pipeline**: Complete video generation from text input to final video output

## Project Structure
```
youtuber-video-generator/
├── src/
│ ├── TTS.py # Text-to-speech generation
│ ├── TTI.py # Text-to-image generation
│ └── ITV.py # Image-to-video composition
├── assets/
│ ├── baseline_voice.wav # Voice baseline for TTS model
│ └── text.txt # Input text (max 500 chars per paragraph)
├── output/
│ ├── audio/ # Generated audio files
│ ├── images/ # Generated images
│ └── videos/ # Final video output
├── .env # Environment variables (API keys)
├── requirements.txt # Python dependencies
├── Makefile # Automation commands
└── README.md # This file
```

## Prerequisites

- Python 3.11.9
- Groq API key (Free models available: https://console.groq.com/)
- Required Python packages (see requirements.txt)

## Installation

1. **Clone the repository**:
```
git clone https://github.com/danielbion/youtuber_video_generation.git
cd youtuber-video-generation
```
2. **Install dependencies**:
```
pip install -r requirements.txt
```
## Configuration

### Environment Variables (.env)
```
GROQ_API_KEY=your_groq_api_key_here
```

### Text Input (assets/text.txt)
- Separate paragraphs with line breaks
- Keep paragraphs under 500 characters for optimal 20-second audio segments
- Each paragraph will generate one image and audio segment

### Voice Baseline (assets/baseline_voice.wav)
The baseline voice recording. I used this text:
```
Hello! My name is [Your Name], and I am recording this text to help create a custom text-to-speech voice.
I enjoy reading books, listening to music, and exploring new places.
Each day brings new opportunities to learn and grow.
The quick brown fox jumps over the lazy dog.
She sells seashells by the seashore.
Would you like a cup of coffee or tea?
Technology is changing the world in many exciting ways.
Artificial intelligence can help people solve complex problems.
It is important to speak clearly and at a steady pace.
Thank you for listening to my voice recording.
```

## Usage

### Quick Start (Full Pipeline)
```
make all
```

### Individual Steps

1. **Generate audio from text**:
`make audio` or `python src/TTS.py`

2. **Generate images from text**:
`make image` or `python src/TTI.py`

3. **Create video from images and audio**:
`make video` or `python src/ITV.py`

## How It Works

1. **Text Processing**: The system reads `assets/text.txt` and splits it into paragraphs
2. **Audio Generation**: Each paragraph is converted to speech using the custom voice baseline
3. **Image Generation**: Relevant images are created for each text paragraph
4. **Video Composition**: Images are animated with zoom effects and synchronized with audio
5. **Final Output**: A complete video file is generated in the `output/videos/` directory

## Customization

### Audio Settings
- Modify TTS parameters in `src/TTS.py`
- Adjust voice baseline in `assets/baseline_voice.wav`

### Image Generation
- Customize system prompt in `src/TTI.py`
- Adjust image dimensions and style parameters



