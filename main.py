import argparse
import base64
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import duckdb
from ollama import Client

OLLAMA_CLOUD_HOST = "https://ollama.com"
DEFAULT_MODEL_NAME = "gemma3:4b-cloud"
DEFAULT_SHEET_NAME = "Sheet1"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

PRIORITY_KEYS = ("classified_at", "category", "place_name", "google_maps_url")


def print_with_time(message: str, start_time: Optional[datetime] = None) -> None:
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    if start_time:
        elapsed = (now - start_time).total_seconds()
        print(f"[{timestamp}] (+{elapsed:.2f}s) {message}")
    else:
        print(f"[{timestamp}] {message}")


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def excel_column_letters(n: int) -> str:
    """1-based column index to Excel-style letters (A, B, …, Z, AA, …)."""
    if n < 1:
        return "A"
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def extract_key_frames(video_path: str, num_frames: int = 5) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    n = min(num_frames, max(total_frames, 1))
    frame_indices = [int(i * total_frames / n) for i in range(n)]
    frames: List[str] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return frames


def image_to_base64_frames(image_path: str) -> List[str]:
    frame = cv2.imread(image_path)
    if frame is None:
        return []
    _, buf = cv2.imencode(".jpg", frame)
    return [base64.b64encode(buf).decode("utf-8")]


def build_prompt(
    categories: List[Dict[str, str]],
    media_description: Optional[str] = None,
) -> str:
    names = [str(c["name"]) for c in categories]
    names_csv = ", ".join(repr(n) for n in names)
    prompt = (
        "You are a media classifier. Given the following video or image and a list of categories, "
        "classify the content and infer location details when possible.\n"
    )
    prompt += "Categories (use category exactly as one of these names):\n"
    for cat in categories:
        prompt += f"- {cat['name']}: {cat['description']}\n"
    if media_description:
        prompt += (
            "\nThe following text describes this media from its source metadata. "
            "Use it together with what you see in the frames or image.\n"
            f"Description:\n{media_description}\n"
        )
    prompt += (
        "\nRespond with a single JSON object only (no markdown fences, no other text). "
        "Use these keys exactly:\n"
        '- "category": string, must be exactly one of: '
        f"{names_csv}\n"
        '- "place_name": string, the human-readable place name if you can infer it from the media '
        "or description; otherwise an empty string.\n"
        '- "google_maps_url": string, a full https URL that opens in a browser (e.g. '
        '"https://www.google.com/maps/search/?api=1&query=..." or '
        '"https://www.google.com/maps/place/...") pointing to that place; '
        "otherwise an empty string if unknown.\n"
        "Do not invent coordinates; base the link on the place you name when possible."
    )
    return prompt


def parse_classification_response(raw: str) -> Dict[str, str]:
    """Parse LLM JSON into fixed string fields for metadata and the sheet."""
    text = raw.strip()
    if not text:
        raise ValueError("Empty LLM response")

    def try_load_json(s: str) -> Optional[Dict[str, Any]]:
        s = s.strip()
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None

    obj = try_load_json(text)
    if obj is None and text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        obj = try_load_json("\n".join(lines))

    if obj is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = try_load_json(text[start : end + 1])

    if not isinstance(obj, dict):
        raise ValueError("LLM response is not a JSON object")

    def as_str(key: str) -> str:
        v = obj.get(key)
        if v is None:
            return ""
        return str(v).strip()

    category = as_str("category")
    if not category:
        raise ValueError('LLM JSON missing non-empty "category"')

    place_name = as_str("place_name")
    google_maps_url = as_str("google_maps_url")
    if google_maps_url and not (
        google_maps_url.startswith("https://") or google_maps_url.startswith("http://")
    ):
        raise ValueError(
            '"google_maps_url" must be empty or a full http(s) URL suitable for a browser'
        )

    return {
        "category": category,
        "place_name": place_name,
        "google_maps_url": google_maps_url,
    }


def query_ollama(
    frames: List[str],
    prompt: str,
    start_time: Optional[datetime] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY environment variable not set.")
    print_with_time(f"Sending request to Ollama Cloud with model {model_name}...", start_time)
    req_start = datetime.now()
    client = Client(
        host=OLLAMA_CLOUD_HOST,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    messages = [{"role": "user", "content": prompt, "images": frames}]
    response = client.chat(model_name, messages=messages, stream=False)
    print_with_time("Received response from Ollama Cloud", req_start)
    if "message" in response and "content" in response["message"]:
        return response["message"]["content"]
    return str(response)


def list_media_files(media_dir: str) -> List[str]:
    if not os.path.isdir(media_dir):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(media_dir)):
        path = os.path.join(media_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
            out.append(path)
    return out


def is_video_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def flatten_value_for_sheet(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def flatten_record_for_sheet(record: Dict[str, Any]) -> Dict[str, str]:
    return {k: flatten_value_for_sheet(v) for k, v in record.items()}


def ordered_column_names(flat: Dict[str, str], existing: List[str]) -> List[str]:
    seen = set(existing)
    merged = list(existing)
    priority_tail = [k for k in PRIORITY_KEYS if k in flat and k not in seen]
    for k in priority_tail:
        merged.append(k)
        seen.add(k)
    rest = sorted(k for k in flat.keys() if k not in seen)
    merged.extend(rest)
    return merged


def get_gsheet_credentials_path() -> str:
    path = os.environ.get("GSHEET_SERVICE_ACCOUNT_JSON") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", ""
    )
    path = path.strip()
    if not path:
        raise RuntimeError(
            "Set GSHEET_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS to your "
            "Google service account JSON key path."
        )
    if not os.path.isfile(path):
        raise RuntimeError(f"Google credentials file not found: {path}")
    return os.path.abspath(path)


def ensure_gsheets_extension(con: duckdb.DuckDBPyConnection) -> None:
    try:
        con.execute("LOAD gsheets;")
    except Exception:
        con.execute("INSTALL gsheets FROM community;")
        con.execute("LOAD gsheets;")


def ensure_gsheet_secret(con: duckdb.DuckDBPyConnection) -> None:
    cred_path = get_gsheet_credentials_path()
    lit = sql_string_literal(cred_path)
    con.execute(
        f"CREATE OR REPLACE SECRET gsheet_svc (TYPE gsheet, PROVIDER key_file, FILEPATH {lit});"
    )


def read_sheet_headers(con: duckdb.DuckDBPyConnection, spreadsheet_url: str, sheet_name: str) -> List[str]:
    u = sql_string_literal(spreadsheet_url)
    s = sql_string_literal(sheet_name)
    try:
        rel = con.sql(f"SELECT * FROM read_gsheet({u}, sheet = {s})")
        return list(rel.columns)
    except Exception:
        return []


def copy_temp_table_to_gsheet(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    spreadsheet_url: str,
    sheet_name: str,
    *,
    range_a1: Optional[str] = None,
    overwrite_range: Optional[bool] = None,
    overwrite_sheet: Optional[bool] = None,
) -> None:
    u = sql_string_literal(spreadsheet_url)
    sh = sql_string_literal(sheet_name)
    opts: List[str] = ["format gsheet", f"sheet {sh}"]
    if range_a1:
        opts.append(f"range {sql_string_literal(range_a1)}")
    if overwrite_range is not None:
        opts.append(f"overwrite_range {'true' if overwrite_range else 'false'}")
    if overwrite_sheet is not None:
        opts.append(f"overwrite_sheet {'true' if overwrite_sheet else 'false'}")
    opt_str = ", ".join(opts)
    con.execute(f"COPY {table_name} TO {u} ({opt_str});")


def write_header_row(
    con: duckdb.DuckDBPyConnection,
    spreadsheet_url: str,
    sheet_name: str,
    headers: List[str],
) -> None:
    con.execute("DROP TABLE IF EXISTS _gsheet_hdr;")
    col_defs = ", ".join(f'c{i} VARCHAR' for i in range(len(headers)))
    con.execute(f"CREATE TEMP TABLE _gsheet_hdr ({col_defs});")
    vals = ", ".join(sql_string_literal(h) for h in headers)
    con.execute(f"INSERT INTO _gsheet_hdr VALUES ({vals});")
    end = excel_column_letters(len(headers))
    rng = f"A1:{end}1"
    copy_temp_table_to_gsheet(
        con,
        "_gsheet_hdr",
        spreadsheet_url,
        sheet_name,
        range_a1=rng,
        overwrite_range=True,
    )
    con.execute("DROP TABLE IF EXISTS _gsheet_hdr;")


def append_data_row(
    con: duckdb.DuckDBPyConnection,
    spreadsheet_url: str,
    sheet_name: str,
    headers: List[str],
    flat: Dict[str, str],
) -> None:
    con.execute("DROP TABLE IF EXISTS _gsheet_row;")
    col_defs = ", ".join(f'c{i} VARCHAR' for i in range(len(headers)))
    con.execute(f"CREATE TEMP TABLE _gsheet_row ({col_defs});")
    vals = ", ".join(sql_string_literal(flat.get(h, "")) for h in headers)
    con.execute(f"INSERT INTO _gsheet_row VALUES ({vals});")
    copy_temp_table_to_gsheet(
        con,
        "_gsheet_row",
        spreadsheet_url,
        sheet_name,
        overwrite_sheet=False,
        overwrite_range=False,
    )
    con.execute("DROP TABLE IF EXISTS _gsheet_row;")


def append_flat_row_to_gsheet(
    con: duckdb.DuckDBPyConnection,
    spreadsheet_url: str,
    sheet_name: str,
    flat: Dict[str, str],
    headers_state: List[str],
) -> List[str]:
    """
    Merges columns with the sheet, updates row 1 if new columns appear, appends one data row.
    Returns updated header list for subsequent rows in this process.
    """
    sheet_headers = read_sheet_headers(con, spreadsheet_url, sheet_name)
    if sheet_headers:
        base = sheet_headers
    else:
        base = list(headers_state)

    merged = ordered_column_names(flat, base)
    if merged != base:
        write_header_row(con, spreadsheet_url, sheet_name, merged)

    append_data_row(con, spreadsheet_url, sheet_name, merged, flat)
    return merged


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def move_to_classified(root: str, media_path: str, metadata_path: str) -> None:
    dest_media_dir = os.path.join(root, "classified", "media")
    dest_meta_dir = os.path.join(root, "classified", "metadata")
    os.makedirs(dest_media_dir, exist_ok=True)
    os.makedirs(dest_meta_dir, exist_ok=True)
    name_m = os.path.basename(media_path)
    name_j = os.path.basename(metadata_path)
    shutil.move(media_path, os.path.join(dest_media_dir, name_m))
    shutil.move(metadata_path, os.path.join(dest_meta_dir, name_j))


def move_to_error(root: str, media_path: Optional[str], metadata_path: Optional[str]) -> None:
    err_dir = os.path.join(root, "error_classifying")
    os.makedirs(err_dir, exist_ok=True)
    if metadata_path and os.path.isfile(metadata_path):
        shutil.move(metadata_path, os.path.join(err_dir, os.path.basename(metadata_path)))
    if media_path and os.path.isfile(media_path):
        shutil.move(media_path, os.path.join(err_dir, os.path.basename(media_path)))


def classify_media_path(
    media_path: str,
    categories: List[Dict[str, str]],
    model_name: str,
    num_frames: int,
    start_time: datetime,
    media_description: Optional[str] = None,
) -> Dict[str, str]:
    prompt = build_prompt(categories, media_description=media_description)
    if is_video_path(media_path):
        print_with_time(f"Extracting frames from {media_path}...", start_time)
        frames = extract_key_frames(media_path, num_frames=num_frames)
        if not frames:
            raise RuntimeError("No frames could be read from video.")
    else:
        print_with_time(f"Loading image {media_path}...", start_time)
        frames = image_to_base64_frames(media_path)
        if not frames:
            raise RuntimeError("Could not read image file.")
    print_with_time("Querying LLM for classification...", start_time)
    raw = query_ollama(frames, prompt, start_time, model_name=model_name)
    return parse_classification_response(raw)


def main() -> None:
    start_time = datetime.now()
    parser = argparse.ArgumentParser(
        description="Classify videos and images under a directory; append metadata rows to Google Sheets."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Root directory with media/ and metadata/ subfolders",
    )
    parser.add_argument("--categories", required=True, help="Path to categories JSON file")
    parser.add_argument(
        "--spreadsheet-url",
        default=os.environ.get("GSHEET_SPREADSHEET_URL", ""),
        help="Google Spreadsheet URL or id (or set GSHEET_SPREADSHEET_URL)",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET_NAME,
        help=f"Worksheet tab name (default: {DEFAULT_SHEET_NAME})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Ollama model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of key frames for videos (default: 5)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print_with_time(f"Directory not found: {root}", start_time)
        return

    if not args.spreadsheet_url.strip():
        print_with_time(
            "Missing spreadsheet URL: pass --spreadsheet-url or set GSHEET_SPREADSHEET_URL.",
            start_time,
        )
        return

    if not os.path.isfile(args.categories):
        print_with_time(f"Categories file not found: {args.categories}", start_time)
        return

    with open(args.categories, "r", encoding="utf-8") as f:
        categories = json.load(f)

    media_dir = os.path.join(root, "media")
    metadata_dir = os.path.join(root, "metadata")
    media_files = list_media_files(media_dir)
    if not media_files:
        print_with_time(f"No media files found in {media_dir}", start_time)
        return

    spreadsheet_url = args.spreadsheet_url.strip()
    if not re.match(r"^https?://", spreadsheet_url):
        spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_url}/edit"

    con = duckdb.connect()
    try:
        ensure_gsheets_extension(con)
        ensure_gsheet_secret(con)
    except Exception as e:
        print_with_time(f"DuckDB / Google Sheets setup failed: {e}", start_time)
        return

    headers_state: List[str] = []

    for media_path in media_files:
        base = os.path.basename(media_path)
        stem, _ = os.path.splitext(base)
        meta_path = os.path.join(metadata_dir, f"{stem}.json")
        rel = os.path.relpath(media_path, root)

        if not os.path.isfile(meta_path):
            print_with_time(f"Missing metadata for {rel}; moving to error_classifying.", start_time)
            move_to_error(root, media_path, None)
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata_obj = json.load(f)
            if not isinstance(metadata_obj, dict):
                raise ValueError("metadata JSON root must be an object")

            desc_raw = metadata_obj.get("description")
            if isinstance(desc_raw, str):
                media_description = desc_raw.strip() or None
            elif desc_raw:
                media_description = str(desc_raw).strip() or None
            else:
                media_description = None

            llm_out = classify_media_path(
                media_path,
                categories,
                args.model,
                args.num_frames,
                start_time,
                media_description=media_description,
            )
            classified_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            enriched = {
                **metadata_obj,
                "classified_at": classified_at,
                **llm_out,
            }
            flat = flatten_record_for_sheet(enriched)

            headers_state = append_flat_row_to_gsheet(
                con, spreadsheet_url, args.sheet, flat, headers_state
            )
            print_with_time(
                f"Appended row for {rel} → category: {llm_out['category']}, "
                f"place: {llm_out['place_name'] or '(none)'}",
                start_time,
            )

            atomic_write_json(meta_path, enriched)
            move_to_classified(root, media_path, meta_path)
            print_with_time(f"Moved to classified/: {rel}", start_time)

        except Exception as e:
            print_with_time(f"Error for {rel}: {e}", start_time)
            move_to_error(root, media_path, meta_path if os.path.isfile(meta_path) else None)


if __name__ == "__main__":
    main()
