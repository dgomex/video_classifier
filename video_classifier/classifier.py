from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .extractor import FrameExtractor


@dataclass
class ClassificationResult:
    category: str
    confidence: float                  # confidence of the winning category (0–1)
    all_scores: dict[str, float]       # avg score for every category
    frame_votes: list[str]             # per-frame winning category


# Default categories — override with your own
DEFAULT_CATEGORIES = [
    "hotels",
    "restaurants",
    "attractions",
    "experiences",
    "others",
]


class VideoClassifier:
    """
    Zero-shot video classifier using OpenAI CLIP.

    Usage:
        classifier = VideoClassifier()
        result = classifier.classify("my_video.mp4")
        print(result.category, result.confidence)
    """

    def __init__(
        self,
        categories: list[str] = DEFAULT_CATEGORIES,
        model_name: str = "openai/clip-vit-base-patch32",
        sample_every_n_frames: int = 60,
        prompt_template: str = "a photo of a {category}",
        device: str | None = None,
    ):
        """
        Args:
            categories:             List of category labels to classify against.
            model_name:             HuggingFace model identifier.
            sample_every_n_frames:  How often to sample frames (lower = slower but more accurate).
            prompt_template:        Text prompt template. {category} is replaced with each label.
            device:                 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        self.categories = categories
        self.prompt_template = prompt_template
        self.device = device or self._detect_device()

        logger.info(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.extractor = FrameExtractor(sample_every_n_frames=sample_every_n_frames)
        logger.info("Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, video_path: str | Path) -> ClassificationResult:
        """
        Classify a video file into one of the configured categories.

        Args:
            video_path: Path to the video file.

        Returns:
            ClassificationResult with the predicted category and scores.
        """
        frames = self.extractor.extract(video_path)
        if not frames:
            raise ValueError(f"No frames could be extracted from {video_path}")

        return self._classify_frames(frames)

    def classify_frames(self, frames: list[Image.Image]) -> ClassificationResult:
        """Classify a list of already-extracted PIL images."""
        if not frames:
            raise ValueError("frames list is empty")
        return self._classify_frames(frames)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify_frames(self, frames: list[Image.Image]) -> ClassificationResult:
        texts = [self.prompt_template.format(category=c) for c in self.categories]

        # Accumulate per-category scores across all frames
        accumulated = {c: 0.0 for c in self.categories}
        frame_votes: list[str] = []

        for frame in frames:
            scores = self._score_frame(frame, texts)
            winner = self.categories[scores.argmax().item()]
            frame_votes.append(winner)
            for i, cat in enumerate(self.categories):
                accumulated[cat] += scores[i].item()

        # Average scores across frames
        avg_scores = {c: v / len(frames) for c, v in accumulated.items()}

        # Majority vote for final prediction
        final_category = Counter(frame_votes).most_common(1)[0][0]
        confidence = avg_scores[final_category]

        logger.info(
            f"Classified as '{final_category}' "
            f"(confidence: {confidence:.2%}, frames: {len(frames)})"
        )
        return ClassificationResult(
            category=final_category,
            confidence=confidence,
            all_scores=avg_scores,
            frame_votes=frame_votes,
        )

    @torch.no_grad()
    def _score_frame(self, frame: Image.Image, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            images=frame,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits_per_image  # shape: (1, num_categories)
        probs = logits.softmax(dim=-1).squeeze(0)       # shape: (num_categories,)
        return probs

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
