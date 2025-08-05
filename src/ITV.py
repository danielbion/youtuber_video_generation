import cv2
import math
import numpy as np
import subprocess
from mutagen import File as AudioFile
import os
import random
from tqdm import tqdm
from pathlib import Path

class Config:
    IMG_DIR = "output/images/"
    AUDIO_DIR = "output/audios/"
    SIZE = (896, 512)  # 16:9
    FPS = 24
    ZOOM = 0.20
    VIDEO_OUTPUT = "output/videos/tmp_video.mp4"
    FINAL_OUTPUT = "output/videos/final_video.mp4"
    AUDIO_INPUT = "output/final_output.mp3"


def ease(t):    
    return 0.5 - 0.5 * math.cos(math.pi * t)

def translate(tx=0, ty=0):    
    M = np.eye(3, dtype=np.float64)
    M[0,2], M[1,2] = tx, ty
    return M

def scale(s):    
    return np.diag([s, s, 1.0]).astype(np.float64)

def gen_zoom(img, zoom_in, frames, size):    
    h, w = img.shape[:2]
    anchor = np.array([w/2, h/2])
    center = np.array([size[0]/2, size[1]/2])
    
    for i in range(frames):
        a = ease(i/frames)
        z = 1 + Config.ZOOM*a if zoom_in else (1+Config.ZOOM) - Config.ZOOM*a
        M = translate(*center) @ scale(z) @ translate(*-anchor)
        yield cv2.warpPerspective(
            img, M, size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE
        )

def get_audio_duration(filename):
    info = AudioFile(filename)
    return info.info.length + 0.2

def merge_audio_video(video_input, audio_input, output):    
    cmd = [
        "ffmpeg",
        "-i", video_input,
        "-i", audio_input,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output
    ]
    subprocess.run(cmd, check=True)

def main():    
    output_dir = Path("output/videos")
    output_dir.mkdir(exist_ok=True)

    imgs = sorted(
        [f for f in os.listdir(Config.IMG_DIR) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    audios = sorted(
        [f for f in os.listdir(Config.AUDIO_DIR) if f.endswith('.wav')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
        
    assert len(imgs) == len(audios), "Number of images and audios do not match"
    
    durations = [get_audio_duration(Config.AUDIO_DIR + a) for a in audios]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(Config.VIDEO_OUTPUT, fourcc, Config.FPS, Config.SIZE)
    
    EFFECTS = [
        lambda im, f: gen_zoom(im, True, f, Config.SIZE),
        lambda im, f: gen_zoom(im, False, f, Config.SIZE)
    ]
    
    for img_path, sec in tqdm(zip(imgs, durations), total = len(imgs), desc="Generating video"):
        frames = max(1, int(round(sec * Config.FPS)))
        img = cv2.imread(os.path.join(Config.IMG_DIR, img_path))
        motion = random.choice(EFFECTS)(img, frames)
        for frame in motion:
            writer.write(frame)
    
    writer.release()
    
    merge_audio_video(Config.VIDEO_OUTPUT, Config.AUDIO_INPUT, Config.FINAL_OUTPUT)
    print(f"Done: {Config.FINAL_OUTPUT}")

if __name__ == "__main__":
    main()