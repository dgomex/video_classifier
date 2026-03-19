import argparse
import json
import cv2
import base64
import os
from ollama import Client
import os
from typing import List, Dict
from datetime import datetime
def print_with_time(message: str, start_time=None):
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    if start_time:
        elapsed = (now - start_time).total_seconds()
        print(f"[{timestamp}] (+{elapsed:.2f}s) {message}")
    else:
        print(f"[{timestamp}] {message}")

OLLAMA_CLOUD_HOST = "https://ollama.com"
MODEL_NAME = "qwen3-vl:235b-cloud"

def extract_key_frames(video_path: str, num_frames: int = 5) -> List[str]:
    """
    Extracts key frames from the video and returns them as base64-encoded JPEGs.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode('.jpg', frame)
        frames.append(base64.b64encode(buf).decode('utf-8'))
    cap.release()
    return frames


def build_prompt(categories: List[Dict[str, str]]) -> str:
    prompt = "You are a video classifier. Given the following video and a list of categories, classify the video into the most appropriate category.\n"
    prompt += "Categories:\n"
    for cat in categories:
        prompt += f"- {cat['name']}: {cat['description']}\n"
    prompt += "\nAnalyze the video and respond ONLY with the category name."
    return prompt


def query_ollama(frames: List[str], prompt: str, start_time=None) -> str:
    # Use Ollama cloud API with API key
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY environment variable not set.")
    print_with_time(f"Sending request to Ollama Cloud with model {MODEL_NAME}...", start_time)
    req_start = datetime.now()
    client = Client(
        host=OLLAMA_CLOUD_HOST,
        headers={'Authorization': f'Bearer {api_key}'}
    )
    # Prepare messages for chat
    messages = [
        {
            'role': 'user',
            'content': prompt,
            'images': frames
        }
    ]
    # Only get the first response part (no streaming)
    response = client.chat(MODEL_NAME, messages=messages, stream=False)
    print_with_time(f"Received response from Ollama Cloud", req_start)
    # The response is a dict with 'message' key
    return response['message']['content'] if 'message' in response and 'content' in response['message'] else str(response)


def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description="Classify a video using Qwen2.5-VL-72B-Instruct via Ollama.")
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--categories', required=True, help='Path to categories JSON file')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print_with_time(f"Video file not found: {args.video}", start_time)
        return
    if not os.path.exists(args.categories):
        print_with_time(f"Categories file not found: {args.categories}", start_time)
        return

    with open(args.categories, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    print_with_time("Extracting frames from video...", start_time)
    frame_start = datetime.now()
    frames = extract_key_frames(args.video, num_frames=30)
    print_with_time(f"Extracted {len(frames)} frames.", frame_start)

    prompt = build_prompt(categories)
    print_with_time("Querying LLM for classification...", start_time)
    result = query_ollama(frames, prompt, start_time)
    print_with_time("\nPredicted Category:", start_time)
    print(result.strip())

if __name__ == "__main__":
    main()
