from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from video_classifier.classifier import ClassificationResult, VideoClassifier
from video_classifier.extractor import FrameExtractor


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_dummy_frame(width=224, height=224) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------
# FrameExtractor
# ------------------------------------------------------------------

class TestFrameExtractor:
    def test_raises_on_missing_file(self):
        extractor = FrameExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent_video.mp4")


# ------------------------------------------------------------------
# VideoClassifier
# ------------------------------------------------------------------

class TestVideoClassifier:
    @pytest.fixture
    def classifier(self):
        return VideoClassifier(
            categories=["hotel", "restaurant", "experience"],
            sample_every_n_frames=30,
        )

    def test_classify_frames_returns_valid_category(self, classifier):
        frames = [make_dummy_frame() for _ in range(5)]
        result = classifier.classify_frames(frames)

        assert isinstance(result, ClassificationResult)
        assert result.category in classifier.categories
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.frame_votes) == 5
        assert set(result.all_scores.keys()) == set(classifier.categories)

    def test_classify_frames_raises_on_empty(self, classifier):
        with pytest.raises(ValueError):
            classifier.classify_frames([])

    def test_scores_sum_to_one(self, classifier):
        frames = [make_dummy_frame()]
        result = classifier.classify_frames(frames)
        total = sum(result.all_scores.values())
        assert abs(total - 1.0) < 1e-4

    def test_custom_categories(self):
        clf = VideoClassifier(categories=["beach", "mountain", "city"])
        frames = [make_dummy_frame()]
        result = clf.classify_frames(frames)
        assert result.category in ["beach", "mountain", "city"]
