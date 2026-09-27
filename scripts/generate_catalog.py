"""Regenerate config/channels.yaml from the synthetic assumption ranges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mediaplan_optimizer.catalog import generate_catalog, save_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assumptions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    destination = save_catalog(
        generate_catalog(seed=args.seed, assumptions_path=args.assumptions),
        path=args.output,
        seed=args.seed,
    )
    print(f"Wrote deterministic synthetic catalog to {destination}")


if __name__ == "__main__":
    main()
