from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from service.classification_service import ClassificationService  # noqa: E402
from streaming.redis_stream import RedisStreamConsumer  # noqa: E402
from worker.classification_worker import ClassificationWorker, DEFAULT_OUTPUT_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BudgetMind Classification Worker (Member 5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to JSONL file where TaskProfiles are saved (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently available stream messages and exit instead of running continuously",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Initialize consumer (uses environment variables via RedisStreamConfig)
    consumer = RedisStreamConsumer()

    # Initialize in-process classification service (loads the 3 .pkl models once)
    service = ClassificationService()

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=args.output,
    )

    if args.once:
        count = worker.run_once()
        logging.getLogger("budgetmind.worker").info("Processed %d task(s)", count)
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
