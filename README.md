# Video Classifier

Zero-shot video classifier using OpenAI CLIP. No training required.

## Setup

```bash
# Install UV if you don't have it
curl -Ls https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

## Usage

### Python API

```python
from video_classifier import VideoClassifier

classifier = VideoClassifier(
    categories=["hotel", "restaurant", "experience", "attraction", "vacation rental"]
)

result = classifier.classify("my_video.mp4")
print(result.category)       # "hotel"
print(result.confidence)     # 0.87
print(result.all_scores)     # {"hotel": 0.87, "restaurant": 0.06, ...}
```

### CLI — single video

```bash
uv run classify my_video.mp4

# Custom categories
uv run classify my_video.mp4 -c hotel -c restaurant -c experience

# Sample more frames (slower, more accurate)
uv run classify my_video.mp4 --sample-every 30
```

### CLI — batch folder

```bash
uv run classify batch ./videos/

# Custom extensions
uv run classify batch ./videos/ --ext mp4,mov
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `categories` | hotel, restaurant... | Labels to classify against |
| `sample_every_n_frames` | 60 | 1 frame every N (at 30fps = every 2s) |
| `model_name` | clip-vit-base-patch32 | Any CLIP model on HuggingFace |
| `prompt_template` | "a photo of a {category}" | Text prompt sent to CLIP |

## Running Tests

```bash
uv run pytest
```
