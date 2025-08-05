.PHONY: help install audio image image-prompt image-file video all

help:
	@echo "Commands:"
	@echo "  install  - Create folders and install dependencies"
	@echo "  audio  - Generate audio files"
	@echo "  images - Generate images"
	@echo "  video  - Create final video"
	@echo "  all    - Generate complete video (audio + images + video)"
	@echo "  clean  - Clean output folders"

install:
	@pip install -r requirements.txt

audio:
	@python src/TTS.py

image:
	@python src/TTI.py

image-prompt:
	@python src/TTI.py --skip-images

image-file:
	@python src/TTI.py --skip-prompts

video:
	@python src/ITV.py

all: audio image video
	@echo "Video generation complete!"
