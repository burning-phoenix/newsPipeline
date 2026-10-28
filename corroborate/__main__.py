import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from corroborate.config import PipelineConfig
from corroborate.pipeline import Pipeline
from corroborate.stages.rendering import default_output_filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corroborated News Brief Pipeline",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show step-by-step progress (quiet by default)",
    )
    parser.add_argument(
        "--category", "-c",
        nargs="+",
        default=[],
        metavar="TAG",
        help="Filter feeds by category tags (default: all feeds)",
    )
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Require feeds to have ALL requested categories (default: ANY match)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_clusters",
        help="Analyze all clusters with 3+ sources (default: top 3 only)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        metavar="N",
        help="Number of top clusters to analyze (default: 3, ignored with --all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        metavar="N",
        help="Concurrent article fetches and LLM calls (default: 16)",
    )
    parser.add_argument(
        "--output", "-o",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Also save the brief to a file. Without PATH it is saved as "
             "outputs/MON_DAY_YEAR.md; a PATH is used exactly as given",
    )
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Paywall 403s are expected; the fetch stage logs each fallback itself.
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
    # trafilatura's pool is 1 connection per host; concurrent fetches to one
    # site make urllib3 warn about discarded connections. Harmless.
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    config = PipelineConfig.load()
    config.categories = args.category
    config.exclusive = args.exclusive
    config.all_clusters = args.all_clusters
    config.top_clusters = args.top
    config.max_workers = max(1, args.workers)
    if not config.api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env file", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(
            f"\nCorroborated News Pipeline — "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
            file=sys.stderr,
        )

    start = time.time()
    pipeline = Pipeline(config)
    document = pipeline.run()

    pipeline.log_usage()
    logging.getLogger(__name__).info("TOTAL TIME: %.1fs", time.time() - start)

    if not document:
        print("No briefs generated.", file=sys.stderr)
        sys.exit(0)

    print(document)

    if args.output is not None:
        path = Path(args.output) if args.output else (
            Path("outputs") / default_output_filename()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document + "\n", encoding="utf-8")
        print(f"Saved to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
