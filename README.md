# Video and image classifier (Ollama + Google Sheets)

Classify videos and images under a directory layout, merge results into paired JSON metadata, append one flattened row per item to Google Sheets via DuckDB’s community `gsheets` extension, then move files into `classified/` or `error_classifying/`.

## Directory layout

```text
<root>/
  media/                 # inputs only at this level (no subfolders scanned)
  metadata/              # <same basename as media>.json
  classified/media/      # successful media (after sheet append)
  classified/metadata/   # enriched JSON
  error_classifying/     # failed items (media ± json)
```

## Requirements

- Python 3.11+
- [DuckDB](https://duckdb.org/) Python package (see `requirements.txt`)
- Ollama Cloud API key
- Google Cloud service account JSON with access to the target spreadsheet (Sheets API enabled; share the spreadsheet with the service account email)

Install (use a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On first run DuckDB downloads the community `gsheets` extension (`INSTALL gsheets FROM community; LOAD gsheets;`).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_API_KEY` | Ollama Cloud API key (required for classification) |
| `GSHEET_SERVICE_ACCOUNT_JSON` or `GOOGLE_APPLICATION_CREDENTIALS` | Path to the Google service account JSON key file (required for Sheets writes) |

## Usage

```bash
export OLLAMA_API_KEY="..."
export GSHEET_SERVICE_ACCOUNT_JSON="/path/to/service-account.json"

python main.py \
  --dir /path/to/root \
  --categories categories.json \
  --spreadsheet-url 'https://docs.google.com/spreadsheets/d/<id>/edit' \
  --sheet Sheet1
```

You can pass the spreadsheet URL via `GSHEET_SPREADSHEET_URL` instead of `--spreadsheet-url`.

### Arguments

- `--dir` (required): Root directory containing `media/` and `metadata/`.
- `--categories` (required): JSON array of `{"name", "description"}` category objects.
- `--spreadsheet-url`: Google Spreadsheet URL or id (or env `GSHEET_SPREADSHEET_URL`).
- `--sheet`: Worksheet tab name (default: `Sheet1`). The tab must already exist.
- `--model`: Ollama model (default: `gemma3:4b-cloud`).
- `--num-frames`: Key frames sampled from videos (default: `5`).

## Behavior

1. For each file in `media/` with a supported video or image extension, the script expects `metadata/<basename>.json`.
2. After classification it adds `classified_at` (ISO 8601 UTC) and `category` (model output) to the metadata object.
3. A flattened row (one column per top-level key; nested values as JSON strings) is **appended** to the sheet (`overwrite_sheet` / `overwrite_range` false for data rows). If new columns appear, row 1 is updated in place for the header range only, then the row is appended.
4. Only after a successful Sheets append: enriched JSON is written and files are moved to `classified/media` and `classified/metadata`.
5. On errors (missing metadata, classification failure, Sheets failure), media (and JSON when present) go to `error_classifying/`.

## Example `categories.json`

```json
[
  {"name": "Sports", "description": "Videos related to sporting events or activities."},
  {"name": "News", "description": "News broadcasts or reports."},
  {"name": "Documentary", "description": "Informative or educational content."}
]
```

## Notes

- Google Sheets cells have size limits; very large JSON values may need trimming in your metadata.
- Share the spreadsheet with the service account email from the JSON (`client_email`).
