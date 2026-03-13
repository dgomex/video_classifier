from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image


class FrameExtractor:
    """Extracts frames from a video file at a given interval."""

    def __init__(self, sample_every_n_frames: int = 60):
        """
        Args:
            sample_every_n_frames: Extract one frame every N frames.
                                   At 30fps, 60 = one frame every 2 seconds.
        """
        self.sample_every_n_frames = sample_every_n_frames

    def extract(self, video_path: str | Path) -> list[Image.Image]:
        """
        Extract sampled frames from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of PIL images.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames: list[Image.Image] = []
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % self.sample_every_n_frames == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))

                frame_index += 1
        finally:
            cap.release()

        logger.debug(
            f"Extracted {len(frames)} frames from {video_path.name} "
            f"({frame_index} total frames, 1 every {self.sample_every_n_frames})"
        )
        return frames
