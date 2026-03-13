from pathlib import Path

import typer
from loguru import logger
from rich import print as rprint
from rich.table import Table

from .classifier import DEFAULT_CATEGORIES, VideoClassifier

app = typer.Typer(help="Zero-shot video classifier using CLIP")


@app.command()
def classify(
    video: Path = typer.Argument(..., help="Path to the video file"),
    categories: list[str] = typer.Option(
        DEFAULT_CATEGORIES, "--category", "-c", help="Category labels"
    ),
    sample_every: int = typer.Option(
        60, "--sample-every", "-s", help="Sample 1 frame every N frames"
    ),
    model: str = typer.Option(
        "openai/clip-vit-base-patch32", "--model", "-m", help="HuggingFace model name"
    ),
):
    """Classify a video into one of the given categories."""
    classifier = VideoClassifier(
        categories=categories,
        model_name=model,
        sample_every_n_frames=sample_every,
    )

    result = classifier.classify(video)

    # Print results table
    table = Table(title=f"Results for: {video.name}")
    table.add_column("Category", style="cyan")
    table.add_column("Avg Score", justify="right", style="green")
    table.add_column("Frame Votes", justify="right")

    vote_counts = {}
    for v in result.frame_votes:
        vote_counts[v] = vote_counts.get(v, 0) + 1

    for cat, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        is_winner = "✅ " if cat == result.category else ""
        table.add_row(
            f"{is_winner}{cat}",
            f"{score:.2%}",
            str(vote_counts.get(cat, 0)),
        )

    rprint(table)
    rprint(f"\n[bold green]Prediction:[/bold green] {result.category} ({result.confidence:.2%} confidence)")


@app.command()
def batch(
    folder: Path = typer.Argument(..., help="Folder containing video files"),
    categories: list[str] = typer.Option(
        DEFAULT_CATEGORIES, "--category", "-c"
    ),
    extensions: str = typer.Option("mp4,mov,avi,mkv", "--ext", help="Comma-separated extensions"),
    sample_every: int = typer.Option(60, "--sample-every", "-s"),
):
    """Classify all videos in a folder."""
    exts = {f".{e.strip().lower()}" for e in extensions.split(",")}
    videos = [f for f in folder.iterdir() if f.suffix.lower() in exts]

    if not videos:
        rprint(f"[red]No videos found in {folder}[/red]")
        raise typer.Exit(1)

    classifier = VideoClassifier(categories=categories, sample_every_n_frames=sample_every)

    results = []
    for video in videos:
        try:
            result = classifier.classify(video)
            results.append((video.name, result.category, result.confidence))
        except Exception as e:
            logger.error(f"Failed to classify {video.name}: {e}")
            results.append((video.name, "ERROR", 0.0))

    table = Table(title=f"Batch Results ({len(videos)} videos)")
    table.add_column("File", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Confidence", justify="right")

    for name, category, confidence in results:
        table.add_row(name, category, f"{confidence:.2%}" if confidence else "—")

    rprint(table)


if __name__ == "__main__":
    app()
