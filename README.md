# Video Classifier using Ollama and Qwen2.5-VL-72B-Instruct

This project allows you to classify a video into a category using the Qwen2.5-VL-72B-Instruct LLM model via the Ollama framework. You provide a video and a list of categories (with descriptions), and the model returns the best-matching category.

## Features
- Upload a video file
- Send video frames and category descriptions to the LLM
- Receive classification result

## Requirements
- Python 3.11+
- Ollama (with Qwen2.5-VL-72B-Instruct model pulled)
- Packages: ollama, moviepy, opencv-python, requests

## Usage
1. Ensure Ollama is running and the qwen2.5vl:3b model is available:
   ```
   ollama pull qwen2.5vl:3b
   ollama serve
   ```
2. Run the classifier script:
   ```
   python main.py --video path_to_video.mp4 --categories categories.json
   ```
   - `--video`: Path to the video file
   - `--categories`: JSON file with categories and descriptions

## Example `categories.json`
```
[
  {"name": "Sports", "description": "Videos related to sporting events or activities."},
  {"name": "News", "description": "News broadcasts or reports."},
  {"name": "Documentary", "description": "Informative or educational content."}
]
```

## Output
The script prints the predicted category for the video.

---

**Note:** This is a basic template. You may need to adjust frame extraction and prompt formatting for best results.