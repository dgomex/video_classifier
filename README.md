# Video Classifier using Ollama and Qwen2.5-VL-72B-Instruct

This project allows you to classify a video into a category using the Qwen2.5-VL-72B-Instruct LLM model via the Ollama framework. You provide a video and a list of categories (with descriptions), and the model returns the best-matching category.

## Features
- Upload a video file
- Send video frames and category descriptions to the LLM
- Receive classification result


## Requirements

- Python 3.11+
- Ollama Python client (cloud API)
- opencv-python

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Set your Ollama Cloud API key as an environment variable:
   - On Windows (PowerShell):
     ```
     $env:OLLAMA_API_KEY="your_api_key_here"
     ```
   - On Linux/macOS:
     ```
     export OLLAMA_API_KEY="your_api_key_here"
     ```

2. Run the classifier script:
   ```
   python main.py --video path_to_video.mp4 --categories categories.json
   ```

   Required:

   - `--video`: Path to the video file
   - `--categories`: JSON file with categories and descriptions

   Optional:

   - `--model`: Ollama model name (default: `gemma3:4b-cloud`)
   - `--num-frames`: Number of key frames sampled evenly across the video (default: `5`)

   Example with optional flags:

   ```
   python main.py --video path_to_video.mp4 --categories categories.json --model "gemma3:4b-cloud" --num-frames 10
   ```

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